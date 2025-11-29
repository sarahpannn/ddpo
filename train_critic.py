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
from torch.utils.data import Dataset, DataLoader, random_split

from collections import defaultdict

# --- Import your env/network definitions ---
from pusht_env import PushTEnv
from dataset import PushTStateDataset, CriticDataset
from network import CNNValueFunction, ConditionalUnet1D, StateIndependentValueWrapper
from ddpo_utils_new import collect_trajectories_flat

from critic_utils import attach_mc_to_trajectories, attach_gae_to_trajectories

def create_dataset(
    policy,
    value_network,
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
    gamma_env=0.99,
    gamma_latent=0.95,
    gae_lambda=1.0,
    save_path="no.pt"
    ):    
    num_collections = dataset_size // num_examples
    global_episode_idx = 0
    all_trajectory_data = []
    
    # Ensure value network is in eval mode
    value_network.eval()
    value_network.to(device)

    for i in tqdm(range(num_collections), desc="Collecting Data"): 
        
        # 1) Collect trajectories
        trajectory_data, returns, _ = collect_trajectories_flat(
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

        # 2) Attach MC Targets
        # This replaces add_discounted_rewards_to_go_torch
        trajectory_data = attach_gae_to_trajectories(
            trajectory_data=trajectory_data,
            final_rewards=returns,
            current_episode_idx_start=global_episode_idx,
            value_network=value_network,
            device=device,
            gamma_env=gamma_env,
            gamma_latent=gamma_latent,
            gae_lambda=gae_lambda 
        )
        
        # 3) Map 'return' (from GAE) to 'target' (expected by dataset)
        for item in trajectory_data:
            item['target'] = item['return']
        
        global_episode_idx += num_examples
        all_trajectory_data.extend(trajectory_data)

    keys = all_trajectory_data[0].keys()
    
    collated_data = {}
    for k in keys:
        example_item = all_trajectory_data[0][k]
        
        if torch.is_tensor(example_item): 
            collated_data[k] = torch.stack([item[k] for item in all_trajectory_data])
        elif isinstance(example_item, np.ndarray): 
            collated_data[k] = torch.stack([torch.from_numpy(item[k]) for item in all_trajectory_data])
        else: 
            collated_data[k] = torch.tensor([item[k] for item in all_trajectory_data])

    torch.save(collated_data, save_path)
    return all_trajectory_data


def validate_value_network(
    network,
    dataloader,
    device
):
    """Run validation and return average loss."""
    network.eval()
    criterion = nn.MSELoss()
    
    val_losses = []
    
    with torch.no_grad():
        for batch in dataloader:
            latents = batch["latents"].to(device)
            timesteps = batch["t"].to(device)
            cond = batch["cond"].to(device)
            targets = batch["target"].to(device).unsqueeze(-1)

            pred_values = network(
                sample=latents, 
                timestep=timesteps, 
                global_cond=cond
            )

            loss = criterion(pred_values, targets)
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def train_value_network(
    network,
    optimizer,
    trajectory_data,
    num_epochs,
    batch_size,
    device,
    val_split=0.1,
    early_stopping_patience=None,
    save_best_path=None
):
    """
    Train value network with validation set support.
    
    Args:
        network: The value network to train
        optimizer: Optimizer for training
        trajectory_data: Dataset (dict or list format)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
        val_split: Fraction of data to use for validation (default: 0.1)
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs (None to disable)
        save_best_path: Path to save best model based on val loss (None to disable)
    
    Returns:
        dict with 'train_loss', 'val_loss', 'best_val_loss', 'best_epoch'
    """
    criterion = nn.MSELoss()
    
    # Create full dataset
    full_dataset = CriticDataset(trajectory_data)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        # pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        # pin_memory=True
    )
    
    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # --- Training ---
        network.train()
        batch_losses = []
        pbar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch in pbar:
            latents = batch["latents"].to(device)
            timesteps = batch["t"].to(device)
            cond = batch["cond"].to(device)
            targets = batch["target"].to(device).unsqueeze(-1) 

            # print(f"target_returns: mean={targets.mean():.2f}, std={targets.std():.2f}, min={targets.min():.2f}, max={targets.max():.2f}")

            pred_values = network(
                sample=latents, 
                timestep=timesteps, 
                global_cond=cond
            )

            loss = criterion(pred_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            wandb.log({"critic_batch_loss": loss.item()})
            pbar.set_postfix({"loss": loss.item()})
        
        train_loss = np.mean(batch_losses)
        epoch_train_losses.append(train_loss)
        
        val_loss = validate_value_network(network, val_dataloader, device)
        epoch_val_losses.append(val_loss)
        
        wandb.log({
            "epoch": epoch + 1,
            "critic_train_loss": train_loss,
            "critic_val_loss": val_loss,
        })
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            if save_best_path:
                torch.save(network.state_dict(), save_best_path)
                print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
        
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break

    return {
        'train_loss': np.mean(epoch_train_losses),
        'val_loss': np.mean(epoch_val_losses),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_train_loss': epoch_train_losses[-1],
        'final_val_loss': epoch_val_losses[-1],
    }


def main():
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    num_diffusion_iters = 100
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    bc_model_path = "bc_policy.pth"
    
    gamma_env = 0.99
    gamma_latent = 0.95
    
    gae_lambda_pretrain = 1.0 

    parallel_envs = 256 # Reduced for safe memory usage, increase if capable
    dataset_size = 512

    critic_condition_on_latent = True
    
    val_split = 0.1  # 10% validation
    early_stopping_patience = 5  # Stop if no improvement for 1 epochs (set to None to disable)
    save_best_model = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    ).to(device)
    bc_model.load_state_dict(torch.load(bc_model_path))
    bc_model.eval()
    
    value_network = CNNValueFunction(
        input_dim=2,
        global_cond_dim=5*obs_horizon,
        down_dims=[128, 256, 512, 1024]
    ).to(device)

    if not critic_condition_on_latent:
        print("Critic (offline) is STATE-ONLY (independent of diffusion latents).")
        value_network = StateIndependentValueWrapper(value_network).to(device)
    else:
        print("Critic (offline) is LATENT-CONDITIONED.")

    if os.path.exists("critic_dataset_gae.pt"):
        print("Loading existing dataset...")
        collated_data = torch.load("critic_dataset_gae.pt")
    else:
        print("Creating new dataset with MC targets...")
        vec_env = SyncVectorEnv([lambda: PushTEnv() for _ in range(parallel_envs)])
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        collated_data = create_dataset(
            policy=bc_model,
            value_network=value_network, 
            vec_env=vec_env,
            noise_scheduler=noise_scheduler,
            dataset_size=dataset_size,
            obs_horizon=obs_horizon,
            num_examples=parallel_envs,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
            num_diffusion_iters=num_diffusion_iters,
            stats=stats,
            device=device,
            gamma_env=gamma_env,
            gamma_latent=gamma_latent,
            gae_lambda=gae_lambda_pretrain,
            save_path="critic_dataset_gae.pt"
        )
    
    optimizer = torch.optim.AdamW(value_network.parameters(), lr=1e-5, weight_decay=1e-4)
    num_epochs = 30

    wandb.init(project="pusht-critic")

    results = train_value_network(
        network=value_network,
        optimizer=optimizer,
        trajectory_data=collated_data,
        num_epochs=num_epochs,
        batch_size=256,
        device=device,
        val_split=val_split,
        early_stopping_patience=early_stopping_patience,
        save_best_path="critic_network_best.pth" if save_best_model else None
    )
    
    print(f"\nTraining complete!")
    print(f"  Best validation loss: {results['best_val_loss']:.6f} (epoch {results['best_epoch']})")
    print(f"  Final train loss: {results['final_train_loss']:.6f}")
    print(f"  Final val loss: {results['final_val_loss']:.6f}")

    # Save final model (may not be the best)
    torch.save(
        value_network.state_dict(),
        "critic_network.pth"
    )
    
    # Log final results to wandb
    wandb.log({
        "best_val_loss": results['best_val_loss'],
        "best_epoch": results['best_epoch'],
    })

if __name__ == "__main__":
    main()