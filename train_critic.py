import torch
import torch.nn as nn
import numpy as np
import os
import random
import wandb
from tqdm.auto import tqdm
from gymnasium.vector import SyncVectorEnv
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

from pusht_env import PushTEnv
from dataset import PushTStateDataset, CriticDataset
from network import CNNValueFunction, ConditionalUnet1D
from ddpo_utils_new import collect_trajectories_flat

def add_discounted_rewards_to_go_torch(
    trajectory_data, 
    final_rewards, 
    current_episode_idx_start, 
    num_diffusion_iters, 
    gamma_env=0.99, 
    gamma_latent=0.95,
    device="cpu",
):
    reward_map = {
        current_episode_idx_start + i: float(r)
        for i, r in enumerate(final_rewards)
    }

    episodes_group = defaultdict(list)
    for item in trajectory_data:
        episodes_group[item['episode_idx']].append(item)

    for ep_idx, items in episodes_group.items():
        final_r = reward_map.get(ep_idx, 0.0)
        total_items = len(items)

        if total_items % num_diffusion_iters != 0:
            print(f"Warning: Episode {ep_idx} has incomplete diffusion chains.")
            continue

        num_env_steps = total_items // num_diffusion_iters

        idx = torch.arange(total_items, device=device, dtype=torch.long)
        env_step_idx = idx // num_diffusion_iters
        steps_from_end = (num_env_steps - 1) - env_step_idx

        env_discount = (gamma_env ** steps_from_end.float())      # (N,)

        t_vals = torch.tensor(
            [it['t'] for it in items],
            device=device,
            dtype=torch.float32,
        )                                                          # (N,)
        latent_discount = gamma_latent ** t_vals                   # (N,)

        discounted = final_r * env_discount * latent_discount      # (N,)

        discounted_cpu = discounted.detach().cpu().tolist()
        for it, drtg in zip(items, discounted_cpu):
            it['discounted_reward_to_go'] = float(drtg)

    return trajectory_data

def create_dataset(
    policy,
    vec_env,
    noise_scheduler,
    dataset_size,
    obs_horizon,
    num_examples,
    action_horizon,
    pred_horizon,
    num_diffusion_iters,
    device,
    stats,
    save_path="no.pt"
    ):    
    num_collections = dataset_size // num_examples
    global_episode_idx = 0
    all_trajectory_data = []
    for i in tqdm(range(num_collections), desc="Collecting Data"): 
        
        # 1) Collect trajectories
        trajectory_data, returns = collect_trajectories_flat(
            env=vec_env,
            batch_size=vec_env.num_envs,
            model=policy,
            noise_scheduler=noise_scheduler,
            stats=stats,
            episode_idx=global_episode_idx, 
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            num_diffusion_iters=num_diffusion_iters,
            device=device
        )

        trajectory_data = add_discounted_rewards_to_go_torch(
            trajectory_data=trajectory_data,
            final_rewards=returns,
            current_episode_idx_start=global_episode_idx,
            num_diffusion_iters=num_diffusion_iters,
            gamma_env=0.99,
            gamma_latent=0.95
        )
        
        global_episode_idx += num_examples
        all_trajectory_data.extend(trajectory_data)

    # torch.save(all_trajectory_data, save_path)
    # print(f"Saved dataset with {len(all_trajectory_data)} samples to {save_path}")
    return all_trajectory_data

def train_value_network(
    network,
    optimizer,
    trajectory_data,
    num_epochs,
    batch_size,
    device
):
    network.train()
    criterion = nn.MSELoss()
    
    dataset = CriticDataset(trajectory_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True # Faster transfer to CUDA
    )
    
    epoch_losses = []

    for epoch in range(num_epochs):
        batch_losses = []
        # Progress bar for the batches
        pbar = tqdm(dataloader, desc=f"Value Train Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch in pbar:
            # Move batch to device
            latents = batch["latents"].to(device)
            timesteps = batch["t"].to(device)
            cond = batch["cond"].to(device)
            targets = batch["target"].to(device).unsqueeze(-1) # (B,) -> (B, 1)

            assert latents.shape[0] == targets.shape[0], f"Batch size mismatch: {latents.shape[0]} vs {targets.shape[0]}"

            pred_values = network(
                sample=latents, 
                timestep=timesteps, 
                global_cond=cond
            )

            loss = criterion(pred_values, targets)
            wandb.log({"critic_loss": loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            
        epoch_losses.append(np.mean(batch_losses))

    return np.mean(epoch_losses)

def main():
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    num_diffusion_iters = 15
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    bc_model_path = "bc_policy.pth"
    
    gamma_env = 0.99
    gamma_latent = 0.95

    parallel_envs = 256
    dataset_size = 1000
    num_collections = dataset_size // parallel_envs
    
    global_episode_idx = 0 

    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    stats = dataset.stats

    bc_model = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=5*obs_horizon
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    bc_model.load_state_dict(torch.load(bc_model_path))
    bc_model.eval()

    if os.path.exists("critic_dataset.pt"):
        print("Loading existing dataset...")
        trajectory_data = torch.load("critic_dataset.pt")
    else:
        print("Creating new dataset...")
        vec_env = SyncVectorEnv([lambda: PushTEnv() for _ in range(parallel_envs)])
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        trajectory_data = create_dataset(
            policy=bc_model,
            vec_env=vec_env,
            noise_scheduler=noise_scheduler,
            dataset_size=dataset_size,
            obs_horizon=obs_horizon,
            num_examples=parallel_envs,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
            num_diffusion_iters=num_diffusion_iters,
            stats=stats,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            save_path="critic_dataset.pt"
        )
    
    value_network = CNNValueFunction(
        input_dim=2,
        global_cond_dim=5*obs_horizon
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.AdamW(value_network.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb.init(project="pusht-critic")

    avg_loss = train_value_network(
        network=value_network,
        optimizer=optimizer,
        trajectory_data=trajectory_data,
        num_epochs=num_epochs,
        batch_size=256,
        device=device
    )

    torch.save(
        value_network.state_dict(),
        "critic_network.pth"
    )

if __name__ == "__main__":
    main()