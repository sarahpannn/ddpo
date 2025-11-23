# train_ppo.py

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

from pusht_env import PushTAdapter
from dataset import PushTStateDataset
from network import ConditionalUnet1D, CNNValueFunction
from ddpo_utils_new import (
    collect_trajectories_flat,
    gaussian_log_prob,
    compute_posterior_mean_var,
)
from critic_utils import attach_gae_to_trajectories
from evaluate import evaluate_push_t


def ensure_scheduler_buffers_on_device(noise_scheduler, device):
    """
    Small helper to mirror what you're already doing in ddpo_utils_new.
    """
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    noise_scheduler.alphas = noise_scheduler.alphas.to(device)
    noise_scheduler.betas = noise_scheduler.betas.to(device)
    # 'one' is a custom buffer you added in ddpo_utils_new
    if not hasattr(noise_scheduler, "one"):
        noise_scheduler.one = torch.tensor(1.0, device=device)
    else:
        noise_scheduler.one = noise_scheduler.one.to(device)


def main():
    # 1. Define Hyperparameters in a Dictionary (very similar to train_ddpo)
    config_dict = {
        "pred_horizon": 16,
        "obs_horizon": 2,
        "action_horizon": 8,
        "obs_dim": 5,
        "action_dim": 2,
        "num_diffusion_iters": 15,
        "num_pg_iters": 1000,
        "batch_size": 128, 
        "max_env_steps": 100,
        "num_train_chunks": 2,
        "epochs": 1,  
        "actor_lr": 3e-7,
        "critic_lr": 1e-5,
        "weight_decay": 1e-6,
        "critic_weight_decay": 1e-4,
        "clip_eps": 0.15,
        "warmup_ratio": 0.05,
        "dataset_path": "pusht_cchi_v7_replay.zarr.zip",
        "seed": 42,
        "load_pretrained_actor": True,
        "load_pretrained_critic": True,
        "critic_path": "critic_network.pth",
        "actor_bc_path": "bc_policy.pth",
        "gradient_clip": 0.5,
        "gamma_env": 0.95,
        "gamma_latent": 0.95,
        "gae_lambda": 0.90,
        "value_coef": 0.5,
        "project_name": "pusht-ddpo-ppo",
        "initialization_seeds": None,
        "use_kl": True,
        "kl_coef": 0.01,
    }

    # 2. Initialize W&B
    wandb.init(
        project=config_dict["project_name"],
        config=config_dict,
    )
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset (only used for stats)
    dataset = PushTStateDataset(
        dataset_path=config.dataset_path,
        pred_horizon=config.pred_horizon,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
    )
    stats = dataset.stats

    # Actor (diffusion policy)
    noise_pred_net = ConditionalUnet1D(
        input_dim=config.action_dim,
        global_cond_dim=config.obs_dim * config.obs_horizon,
    ).to(device)
    # wandb.watch(noise_pred_net, log="all", log_freq=10)

    # Optionally load BC actor
    if config.load_pretrained_actor and os.path.isfile(config.actor_bc_path):
        print(f"Loading pre-trained actor from {config.actor_bc_path}")
        state_dict = torch.load(config.actor_bc_path, map_location=device)
        noise_pred_net.load_state_dict(state_dict)
    else:
        print("Warning: BC actor weights not found or disabled. Starting actor from scratch.")

    # Critic
    value_network = CNNValueFunction(
        input_dim=config.action_dim,
        global_cond_dim=config.obs_dim * config.obs_horizon,
    ).to(device)
    wandb.watch(value_network, log="all", log_freq=10)

    if config.load_pretrained_critic and os.path.isfile(config.critic_path):
        print(f"Loading pre-trained critic from {config.critic_path}")
        critic_state = torch.load(config.critic_path, map_location=device)
        value_network.load_state_dict(critic_state)
    else:
        print("Critic starting from scratch (or offline training path not found).")

    # Scheduler for diffusion timesteps
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # Environment
    env = SyncVectorEnv([lambda: PushTAdapter() for _ in range(config.batch_size)])

    # Optimizers
    actor_optimizer = torch.optim.AdamW(
        noise_pred_net.parameters(),
        lr=config.actor_lr,
        weight_decay=config.weight_decay,
    )
    critic_optimizer = torch.optim.AdamW(
        value_network.parameters(),
        lr=config.critic_lr,
        weight_decay=config.critic_weight_decay,
    )

    # LR scheduler for actor (same warmup / cosine idea as train_ddpo)
    total_steps = config.epochs * config.num_pg_iters
    warmup_steps = int(config.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    actor_lr_scheduler = LambdaLR(actor_optimizer, lr_lambda)

    global_step = 0
    global_episode_idx = 0

    for it in range(config.num_pg_iters):
        if it % 20 == 0 and it > 0:
            avg_max_score, success_rate =evaluate_push_t(
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
                "eval_success_rate": success_rate,
                "traj_step": it,
            })

        print(f"\n=== PPO Iteration {it + 1}/{config.num_pg_iters} ===")

        # ===== Phase 1: Collect trajectories =====
        time_gen_0 = time.time()
        trajectory_data, returns = collect_trajectories_flat(
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
            episode_idx=global_episode_idx,
            initialization_seeds=config.initialization_seeds,
        )
        time_gen_1 = time.time()
        batch_collection_time = time_gen_1 - time_gen_0
        print(f"Collected trajectories in {batch_collection_time:.2f} sec")

        # Attach critic values + GAE-based advantages/returns
        trajectory_data = attach_gae_to_trajectories(
            trajectory_data=trajectory_data,
            final_rewards=returns,
            current_episode_idx_start=global_episode_idx,
            value_network=value_network,
            device=device,
            num_diffusion_iters=config.num_diffusion_iters,
            gamma_env=config.gamma_env,
            gamma_latent=config.gamma_latent,
            gae_lambda=config.gae_lambda,
        )

        global_episode_idx += config.batch_size  # one episode per env

        # Flatten trajectory_data into tensors
        latents = torch.cat(
            [torch.as_tensor(item["latents"]) for item in trajectory_data],
            dim=0,
        ).to(device)  # (N, pred_horizon, action_dim)

        next_latents = torch.cat(
            [torch.as_tensor(item["next_latents"]) for item in trajectory_data],
            dim=0,
        ).to(device)  # (N, pred_horizon, action_dim)

        timesteps = torch.tensor(
            [int(item["t"]) for item in trajectory_data],
            dtype=torch.long,
            device=device,
        )  # (N,)

        conds = torch.cat(
            [torch.as_tensor(item["cond"]) for item in trajectory_data],
            dim=0,
        ).to(device)  # (N, obs_horizon * obs_dim)

        advantages = torch.tensor(
            [float(item["advantage"]) for item in trajectory_data],
            dtype=torch.float32,
            device=device,
        )  # (N,)

        target_returns = torch.tensor(
            [float(item["return"]) for item in trajectory_data],
            dtype=torch.float32,
            device=device,
        )  # (N,)

        # Normalize advantages (standard PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = latents.shape[0]
        print(f"Total trajectory samples: {N}")

        # ===== Phase 2: Compute old log-probs =====
        ensure_scheduler_buffers_on_device(noise_scheduler, device)

        with torch.no_grad():
            noise_pred_old = noise_pred_net(
                sample=latents,
                timestep=timesteps,
                global_cond=conds,
            )
            mu_t_old, var_t_old = compute_posterior_mean_var(
                noise_scheduler=noise_scheduler,
                x_t=latents,
                noise_pred=noise_pred_old,
                timesteps=timesteps,
            )

            # We ignore horizon slicing and just use full pred_horizon,
            # consistent with your current update_model_efficiently (use_slice = False).
            old_log_probs = gaussian_log_prob(
                x=next_latents,
                mean=mu_t_old,
                var=var_t_old,
            )
            old_log_probs = old_log_probs.detach()
            assert torch.all(torch.isfinite(old_log_probs)), "Old log probs have NaNs/inf."

        # ===== Phase 3: PPO-style joint actor & critic update =====
        time_opt_0 = time.time()

        noise_pred_net.train()
        value_network.train()

        total_loss_val = 0.0
        total_policy_loss_val = 0.0
        total_value_loss_val = 0.0
        total_approx_kl = 0.0
        num_batches = 0

        # Mini-batch size based on N and num_train_chunks
        batch_size = max(1, N // config.num_train_chunks)

        for epoch in range(config.epochs):
            indices = torch.randperm(N, device=device)
            for start in range(0, N, batch_size):
                idx = indices[start:start + batch_size]

                b_latents = latents[idx]
                b_next_latents = next_latents[idx]
                b_t = timesteps[idx]
                b_cond = conds[idx]
                b_old_log_probs = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = target_returns[idx]

                # --- Actor forward ---
                noise_pred = noise_pred_net(
                    sample=b_latents,
                    timestep=b_t,
                    global_cond=b_cond,
                )
                mu_t, var_t = compute_posterior_mean_var(
                    noise_scheduler=noise_scheduler,
                    x_t=b_latents,
                    noise_pred=noise_pred,
                    timesteps=b_t,
                )

                new_log_probs = gaussian_log_prob(
                    x=b_next_latents,
                    mean=mu_t,
                    var=var_t,
                )

                # --- PPO clipped objective ---
                raw_log_ratio = new_log_probs - b_old_log_probs  # (B,)
                raw_log_ratio = torch.nan_to_num(raw_log_ratio, nan=0.0)
                log_ratio = torch.clamp(raw_log_ratio, -10.0, 10.0)
                ratio = torch.exp(log_ratio)

                approx_kl = 0.5 * torch.mean((new_log_probs - b_old_log_probs) ** 2)
                total_approx_kl += approx_kl.detach().item()

                wandb.log({
                    "stability/log_ratio_mean": raw_log_ratio.mean().item(),
                    "stability/log_ratio_max": raw_log_ratio.max().item(),
                    "stability/log_ratio_min": raw_log_ratio.min().item(),
                    "stability/log_ratio_std": raw_log_ratio.std().item(),
                    "stability/ratio_mean": ratio.mean().item(),
                    "stability/ratio_max": ratio.max().item(),
                    "stability/ratio_min": ratio.min().item(),
                    "stability/ratio_std": ratio.std().item(),
                    "stability/approx_kl": approx_kl.item(),
                    "global_step": global_step,
                })

                assert torch.all(torch.isfinite(log_ratio)), "log_ratio has NaNs/inf."
                assert torch.all(torch.isfinite(ratio)), "ratio has NaNs/inf."

                b_adv = torch.clamp(b_adv, -5.0, 5.0)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Critic loss ---
                values_pred = value_network(
                    sample=b_latents,
                    timestep=b_t,
                    global_cond=b_cond,
                ).squeeze(-1)

                value_loss = 0.5 * (values_pred - b_ret).pow(2).mean()

                loss = policy_loss + config.value_coef * value_loss

                if config.use_kl: loss = loss + config.kl_coef * approx_kl

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()

                # Gradient norm logging + clipping
                total_norm = 0.0
                for p in list(noise_pred_net.parameters()) + list(value_network.parameters()):
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                if config.gradient_clip is not None and config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        noise_pred_net.parameters(),
                        config.gradient_clip,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        value_network.parameters(),
                        config.gradient_clip,
                    )

                wandb.log({
                    "stability/grad_norm": total_norm,
                    "global_step": global_step,
                })

                actor_optimizer.step()
                critic_optimizer.step()

                total_loss_val += loss.item()
                total_policy_loss_val += policy_loss.item()
                total_value_loss_val += value_loss.item()
                num_batches += 1
                global_step += 1

        avg_loss = total_loss_val / max(1, num_batches)
        avg_policy_loss = total_policy_loss_val / max(1, num_batches)
        avg_value_loss = total_value_loss_val / max(1, num_batches)
        avg_approx_kl = total_approx_kl / max(1, num_batches)

        actor_lr_scheduler.step()

        time_opt_1 = time.time()
        optimization_time = time_opt_1 - time_opt_0

        print(
            f"[Iter {it}] avg_loss={avg_loss:.6f}, "
            f"policy_loss={avg_policy_loss:.6f}, "
            f"value_loss={avg_value_loss:.6f}, "
            f"avg_return={returns.mean().item():.2f}, "
            f"return_std={returns.std().item():.2f}"
        )

        wandb.log({
            "main/ppo_loss": avg_loss,
            "main/policy_loss": avg_policy_loss,
            "main/value_loss": avg_value_loss,
            "main/avg_return": returns.mean().item(),
            "main/return_std": returns.std().item(),
            "main/kl_loss": avg_approx_kl,
            "time/batch_collection_time_sec": batch_collection_time,
            "time/optimization_time_sec": optimization_time,
            "main/actor_lr": actor_optimizer.param_groups[0]["lr"],
            "main/critic_lr": critic_optimizer.param_groups[0]["lr"],
            "traj_step": it,
        })


if __name__ == "__main__":
    main()
