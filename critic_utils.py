# critic_utils.py

import torch
from typing import List, Dict, Any

# Make sure train_critic.py is on your PYTHONPATH / in same folder
from train_critic import add_discounted_rewards_to_go_torch


@torch.no_grad()
def attach_gae_to_trajectories(
    trajectory_data: List[Dict[str, Any]],
    final_rewards: torch.Tensor,
    current_episode_idx_start: int,
    value_network,
    device: torch.device,
    num_diffusion_iters: int,
    gamma_env: float = 0.99,
    gamma_latent: float = 0.95,
    gae_lambda: float = 0.95,  # kept for API compatibility; unused
):
    """
    Attach critic values + MC-style two-timescale targets + advantages.

    This uses the *same* target definition as in train_critic.py:

        add_discounted_rewards_to_go_torch(
            trajectory_data, final_rewards, ..., gamma_env, gamma_latent
        )

    After calling that, every item in trajectory_data has:
        item["discounted_reward_to_go"]

    We then:
      - Evaluate critic to get V(s_t)
      - Set:
          item["return"]    = discounted_reward_to_go
          item["value"]     = V(s_t)
          item["advantage"] = item["return"] - item["value"]

    Args:
        trajectory_data: list of dicts from collect_trajectories_flat
        final_rewards:   tensor (num_envs,) with episodic returns
        current_episode_idx_start: starting episode index for this batch
        value_network:   CNNValueFunction
        device:          torch device
        num_diffusion_iters: K diffusion steps per env step
        gamma_env:       env-step discount (same as offline critic training)
        gamma_latent:    diffusion-step discount (same as offline)
        gae_lambda:      unused (PPO-style GAE disabled here)

    Returns:
        trajectory_data with added fields:
          - "value"
          - "return"
          - "advantage"
    """
    device = torch.device(device)

    # 1) Reuse your existing logic to compute discounted_reward_to_go
    trajectory_data = add_discounted_rewards_to_go_torch(
        trajectory_data=trajectory_data,
        final_rewards=final_rewards,
        current_episode_idx_start=current_episode_idx_start,
        num_diffusion_iters=num_diffusion_iters,
        gamma_env=gamma_env,
        gamma_latent=gamma_latent,
        device=device,
    )

    # 2) Stack inputs for critic forward pass
    #    (We don't need to group by episode here; we've already got targets per item.)
    latents = torch.cat(
        [torch.as_tensor(it["latents"]) for it in trajectory_data],
        dim=0,
    ).to(device)  # (N, pred_horizon, action_dim)

    timesteps = torch.tensor(
        [int(it["t"]) for it in trajectory_data],
        dtype=torch.long,
        device=device,
    )  # (N,)

    cond = torch.cat(
        [torch.as_tensor(it["cond"]) for it in trajectory_data],
        dim=0,
    ).to(device)  # (N, obs_horizon * obs_dim)

    # 3) Critic predictions
    value_network.eval()
    values = value_network(
        sample=latents,
        timestep=timesteps,
        global_cond=cond,
    ).squeeze(-1)  # (N,)

    values_cpu = values.detach().cpu().tolist()

    # 4) Attach value, return (target), and advantage per item
    assert len(values_cpu) == len(trajectory_data)
    for v, it in zip(values_cpu, trajectory_data):
        R_t = float(it["discounted_reward_to_go"])
        V_t = float(v)
        A_t = R_t - V_t

        it["value"] = V_t
        it["return"] = R_t
        it["advantage"] = A_t

    return trajectory_data
