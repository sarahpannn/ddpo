import torch
import numpy as np
import wandb
import time
import math
import os
import random
from gymnasium.vector import SyncVectorEnv
from torch.optim.lr_scheduler import LambdaLR
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from pusht_env import PushTAdapter, PushTEnv
from dataset import PushTStateDataset
from network import ConditionalUnet1D
# from ddpo_utils import collect_trajectories_flat, update_model_efficiently
from ddpo_utils_new import collect_trajectories_flat, update_model_efficiently
from evaluate import evaluate_push_t

def main():
    # 1. Define Hyperparameters in a Dictionary
    config_dict = {
        "pred_horizon": 16,
        "obs_horizon": 2,
        "action_horizon": 8,
        "obs_dim": 5,
        "action_dim": 2,
        "num_diffusion_iters": 15,
        "num_pg_iters": 1000,
        "batch_size": 128,        # episodes per PG iteration
        "max_env_steps": 100,
        # "big_batch_size": 8192 * 2 * 2,
        "num_train_chunks": 1,
        "epochs": 1,
        "lr": 2e-7,
        "weight_decay": 1e-6,
        "clip_eps": 0.15,         # Moved from update_model_efficiently call
        "warmup_ratio": 0.05,     # Extracted from the calculation
        "dataset_path": "pusht_cchi_v7_replay.zarr.zip",
        "seed": 42,
        "load_pretrained": True,
        "gradient_clip": 0.5,
    }

    # 2. Initialize WandB with the config
    
    wandb.init(
        project="diffusion_policy_ddpo",
        config=config_dict
    )
    
    # 3. Use wandb.config (Allows sweeps to override values automatically)
    config = wandb.config

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    wandb.define_metric("global_step")
    wandb.define_metric("stability/*", step_metric="global_step")
    wandb.define_metric("main/*", step_metric="traj_step")
    wandb.define_metric("time/*", step_metric="traj_step")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset (needed for stats)
    dataset = PushTStateDataset(
        dataset_path=config.dataset_path,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon
    )
    stats = dataset.stats

    # Network
    noise_pred_net = ConditionalUnet1D(
        input_dim=config.action_dim,
        global_cond_dim=config.obs_dim * config.obs_horizon
    ).to(device)
    wandb.watch(noise_pred_net, log="all", log_freq=10)

    # Load pre-trained weights if available
    if config.load_pretrained and os.path.isfile('bc_policy.pth'):
        print("Loading pre-trained weights from bc_policy.pth")
        state_dict = torch.load('bc_policy.pth', map_location=device)
        noise_pred_net.load_state_dict(state_dict)
    else:
        print("Warning: bc_policy.pth not found or disabled. Starting from scratch.")

    # Scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Environment
    # Note: Use config.batch_size to determine number of parallel envs
    env = SyncVectorEnv([lambda: PushTAdapter() for _ in range(config.batch_size)])

    noise_pred_net.train()

    optimizer = torch.optim.AdamW(
        noise_pred_net.parameters(),
        lr=config.lr, 
        weight_decay=config.weight_decay
    )

    # Calculate steps using config values
    total_steps = (
        # config.num_pg_iters * config.max_env_steps * config.num_diffusion_iters * config.epochs * config.batch_size // config.big_batch_size
        # config.num_train_chunks * config.epochs * config.num_pg_iters
        config.epochs * config.num_pg_iters
    )

    warmup_steps = int(config.warmup_ratio * total_steps)

    lr_scheduler = LambdaLR(optimizer, lambda step: step / warmup_steps if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

    global_step = 0

    for it in range(config.num_pg_iters):
        if it % 20 == 0:
            avg_max_score, success_rate = evaluate_push_t(
                noise_pred_net, 
                noise_scheduler,
                stats, 
                device, 
                num_diffusion_iters=config.num_diffusion_iters,
                obs_horizon=config.obs_horizon,
                action_horizon=config.action_horizon,
                max_steps=config.max_env_steps,
            )

            wandb.log({
                "eval_max_score": avg_max_score,
                "eval_success_rate": success_rate
            })
        
        # ===== Phase 1: collect trajectories (no grad) =====
        time_gen_0 = time.time()
        trajectory_data, returns, meta = collect_trajectories_flat(
            env=env,
            model=noise_pred_net,
            noise_scheduler=noise_scheduler,
            stats=stats,
            batch_size=config.batch_size,
            obs_horizon=config.obs_horizon,
            pred_horizon=config.pred_horizon,
            action_horizon=config.action_horizon,
            num_diffusion_iters=config.num_diffusion_iters,
            device=device,
            max_env_steps=config.max_env_steps,
        )
        time_gen_1 = time.time()
        batch_collection_time = time_gen_1 - time_gen_0
        print(f'collected trajectories in {batch_collection_time:.2f} sec')

        if len(meta) > 0:
            rewards_by_seed = {}
            for ep in meta:
                s = ep["init_seed"]
                rewards_by_seed.setdefault(s, []).append(ep["final_reward"])

            log_dict = {}
            for s, vals in rewards_by_seed.items():
                vals = np.array(vals, dtype=np.float32)
                log_dict[f"train/return_seed_{s}"] = float(vals.mean())
                log_dict[f"train/max_return_seed_{s}"] = float(vals.max())
                # Optional: success if reward > some threshold
                # log_dict[f"train/success_seed_{s}"] = float((vals > THRESH).mean())

            wandb.log(log_dict, step=global_step)


        # ===== Phase 2: big batched model update =====
        time_opt_0 = time.time()
        avg_loss, kl_loss, global_step = update_model_efficiently(
            model=noise_pred_net,
            noise_scheduler=noise_scheduler,
            optimizer=optimizer,
            trajectory_data=trajectory_data,
            returns=returns,
            device=device,
            lr_scheduler=lr_scheduler,
            # batch_size=config.big_batch_size,
            num_batches_per_epoch=config.num_train_chunks,
            epochs=config.epochs,
            clip_eps=config.clip_eps, # Using config value
            global_step=global_step,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            gradient_clip=config.gradient_clip,
        )
        time_opt_1 = time.time()
        optimization_time = time_opt_1 - time_opt_0
        print(f'optimization in {optimization_time:.2f} sec')

        print(f"[DDPO] iter {it:04d} loss={avg_loss:.3f} "
              f"return_mean={returns.mean().item():.2f} "
              f"return_std={returns.std().item():.2f}")

        wandb.log({
            "main/ddpo_loss": avg_loss,
            "main/avg_return": returns.mean().item(),
            "main/return_std": returns.std().item(),
            "main/kl_loss": kl_loss,
            "main/loss_no_kl": avg_loss - kl_loss,
            "time/batch_collection_time_sec": batch_collection_time,
            "time/optimization_time_sec": optimization_time,
            "main/lr": optimizer.param_groups[0]['lr'],
            "traj_step": it
        })

if __name__ == "__main__":
    main()