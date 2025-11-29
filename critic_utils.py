# critic_utils.py

import torch
from typing import List, Dict, Any
from collections import defaultdict

# critic_utils.py

import torch
from collections import defaultdict


@torch.no_grad()
def attach_gae_to_trajectories(
    trajectory_data,
    final_rewards,
    current_episode_idx_start,
    value_network,
    device,
    gamma_env: float = 0.99,
    gamma_latent: float = 0.95,
    gae_lambda: float = 1.0,
    per_seed_norm: bool = True,
):

    device = torch.device(device)
    value_network = value_network.to(device)
    value_network.eval()

    # ------------------------------------------------------------------
    # 1) Map episode index -> final reward (fallback)
    # ------------------------------------------------------------------
    reward_map = {
        current_episode_idx_start + i: float(r)
        for i, r in enumerate(final_rewards)
    }

    # ------------------------------------------------------------------
    # 2) Batch value prediction
    # ------------------------------------------------------------------
    latents = torch.stack(
        [torch.as_tensor(it["latents"]) for it in trajectory_data]
    ).to(device)
    timesteps = torch.tensor(
        [int(it["t"]) for it in trajectory_data], dtype=torch.long, device=device
    )
    cond = torch.stack(
        [torch.as_tensor(it["cond"]) for it in trajectory_data]
    ).to(device)

    # Handle possible singleton dims
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.squeeze(1)
    if cond.ndim == 3 and cond.shape[1] == 1:
        cond = cond.squeeze(1)

    with torch.no_grad():
        all_values = value_network(
            sample=latents,
            timestep=timesteps,
            global_cond=cond,
        ).squeeze(-1)  # (N,)

    for i, it in enumerate(trajectory_data):
        it["value_pred"] = float(all_values[i].item())

    # ------------------------------------------------------------------
    # 3) Group by episode and run GAE in correct temporal order
    # ------------------------------------------------------------------
    episodes_group = defaultdict(list)
    for item in trajectory_data:
        episodes_group[item["episode_idx"]].append(item)

    for ep_idx, items in episodes_group.items():
        final_r = reward_map.get(ep_idx, 0.0)

        # True temporal order:
        #   env_step ascending, diffusion timestep descending (T-1,...,0)
        items.sort(key=lambda x: (x["env_step"], -x["t"]))

        next_value = 0.0
        next_advantage = 0.0

        # Backwards over (env_step, t)
        for i in reversed(range(len(items))):
            item = items[i]
            curr_val = float(item["value_pred"])
            curr_t = int(item["t"])

            is_last = (i == len(items) - 1)

            # ----- Reward -----
            # If we stored per-step reward on the item, use it.
            # Otherwise fall back to "all reward at final step".
            if "reward" in item:
                r_t = float(item["reward"])
            else:
                r_t = final_r if is_last else 0.0

            # Two time-scales:
            #  - inside diffusion chain (t > 0): gamma_latent
            #  - when we move to next env_step (t == 0): gamma_env
            gamma = gamma_env if curr_t == 0 else gamma_latent

            # No bootstrapping from terminal state
            mask = 0.0 if is_last else 1.0

            # TD residual and GAE
            delta = r_t + gamma * next_value * mask - curr_val
            advantage = delta + gamma * gae_lambda * next_advantage * mask

            # Store raw (unnormalized) advantage and value target
            item["advantage_raw"] = float(advantage)
            item["advantage"] = float(advantage)  # will overwrite after norm
            item["return"] = float(advantage + curr_val)
            item["value"] = curr_val

            next_value = curr_val
            next_advantage = advantage

    if per_seed_norm:
        seed_to_indices = defaultdict(list)
        seed_to_adv = defaultdict(list)

        for idx, item in enumerate(trajectory_data):
            seed = item.get("init_seed", None)
            seed_key = int(seed) if seed is not None else "global"
            seed_to_indices[seed_key].append(idx)
            seed_to_adv[seed_key].append(float(item["advantage_raw"]))

        for seed_key, indices in seed_to_indices.items():
            adv_tensor = torch.tensor(
                seed_to_adv[seed_key], device=device, dtype=torch.float32
            )
            mean = adv_tensor.mean()
            std = adv_tensor.std(unbiased=False)
            if std <= 0:
                std = torch.tensor(1.0, device=device)

            norm_adv = (adv_tensor - mean) / (std + 1e-8)

            for v, idx in zip(norm_adv.tolist(), indices):
                trajectory_data[idx]["advantage"] = float(v)

    return trajectory_data



@torch.no_grad()
def attach_mc_to_trajectories(
    trajectory_data: List[Dict[str, Any]],
    final_rewards: torch.Tensor,
    current_episode_idx_start: int,
    value_network,
    device: torch.device,
    gamma_env: float = 0.99,
    gamma_latent: float = 0.95,
    **kwargs,  # Accept but ignore extra kwargs like gae_lambda
):
    """
    Attach Monte Carlo returns (no bootstrapping).
    This is equivalent to GAE with lambda=1.0.
    
    Kept for backward compatibility.
    """
    return attach_gae_to_trajectories(
        trajectory_data=trajectory_data,
        final_rewards=final_rewards,
        current_episode_idx_start=current_episode_idx_start,
        value_network=value_network,
        device=device,
        gamma_env=gamma_env,
        gamma_latent=gamma_latent,
        gae_lambda=1.0,  # MC = GAE with lambda=1.0
    )


def add_discounted_rewards_to_go_torch(
    trajectory_data: List[Dict[str, Any]],
    final_rewards: torch.Tensor,
    current_episode_idx_start: int,
    num_diffusion_iters: int,
    gamma_env: float,
    gamma_latent: float,
    device: torch.device,
):
    """
    Compute discounted reward-to-go for each trajectory item.
    This is a pure reward-based target (no value function).
    
    Used for offline critic pretraining.
    """
    device = torch.device(device)
    
    # Map episode index to final reward
    reward_map = {
        current_episode_idx_start + i: float(r)
        for i, r in enumerate(final_rewards)
    }
    
    # Group by episode
    episodes_group = defaultdict(list)
    for item in trajectory_data:
        episodes_group[item['episode_idx']].append(item)
    
    for ep_idx, items in episodes_group.items():
        final_r = reward_map.get(ep_idx, 0.0)
        
        # Sort by temporal order: env_step ascending, diffusion t descending
        items.sort(key=lambda x: (x['env_step'], -x['t']))
        
        # Backward pass to compute discounted return
        running_return = 0.0
        
        for i in reversed(range(len(items))):
            item = items[i]
            curr_t = item['t']
            
            is_last = (i == len(items) - 1)
            r_t = final_r if is_last else 0.0
            
            gamma = gamma_env if curr_t == 0 else gamma_latent
            
            if is_last:
                running_return = r_t
            else:
                running_return = r_t + gamma * running_return
            
            item['discounted_reward_to_go'] = running_return
    
    return trajectory_data