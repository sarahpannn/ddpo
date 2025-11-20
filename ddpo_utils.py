import torch
import numpy as np
from dataset import normalize_data, unnormalize_data

def gaussian_log_prob(x, mean, var):
    var = torch.clamp(var, min=1e-4)
    log_prob_per_dim = -0.5 * (
        ((x - mean) ** 2) / var
        + torch.log(2 * torch.pi * var)
    )
    # true log prob
    return log_prob_per_dim.flatten(start_dim=1).sum(dim=1)

def rollout_ddpo_collect_flat(
    env,
    model,
    noise_scheduler,
    stats,
    episode_idx: int,   # now interpreted as "episode_start_idx" for vec env
    obs_horizon,
    pred_horizon,
    action_horizon,
    num_diffusion_iters,
    device,
    max_env_steps=100,
):
    """
    Parallel DDPO rollout over a VectorEnv.

    env: VectorEnv with env.num_envs environments.
    Returns:
        trajectory_data: list of dicts, each with keys:
            - 'latents':      x_t          (tensor, shape [1, pred_horizon, action_dim], CPU)
            - 't':            timestep int
            - 'cond':         obs_cond     (tensor, shape [1, obs_horizon*obs_dim], CPU)
            - 'next_latents': x_{t-1}      (tensor, shape [1, pred_horizon, action_dim], CPU)
            - 'episode_idx':  int in [episode_idx, episode_idx + num_envs)
        final_rewards: np.ndarray of shape (num_envs,) with total reward per env
    """
    vec_env = env
    num_envs = vec_env.num_envs
    action_dim = vec_env.single_action_space.shape[0]
    # obs_dim = vec_env.single_observation_space.shape[0]  # only needed if you want to sanity-check

    trajectory_data = []

    # Track rewards per env
    current_rewards = np.zeros(num_envs, dtype=np.float32)
    final_rewards = np.zeros(num_envs, dtype=np.float32)
    active_envs = np.ones(num_envs, dtype=bool)

    # ===== Reset all envs (no grad) =====
    with torch.no_grad():
        obs, infos = vec_env.reset()  # obs: (num_envs, obs_dim)
    # Initialize history: (num_envs, obs_horizon, obs_dim)
    obs_history = np.tile(obs[:, None, :], (1, obs_horizon, 1))

    step_idx = 0

    # Move scheduler buffers to device once
    with torch.no_grad():
        noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
        noise_scheduler.alphas = noise_scheduler.alphas.to(device)
        noise_scheduler.betas = noise_scheduler.betas.to(device)
        noise_scheduler.one = torch.tensor(1.0, device=device)

    while np.any(active_envs) and step_idx < max_env_steps:
        with torch.no_grad():
            # ===== Build conditioning for all envs =====
            # obs_history: (num_envs, obs_horizon, obs_dim)
            nobs = normalize_data(obs_history, stats=stats["obs"])
            nobs_t = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            obs_cond = nobs_t.flatten(start_dim=1)  # (num_envs, obs_horizon * obs_dim)

            if torch.isnan(nobs_t).any():
                print("NaNs in normalized obs")

            B = num_envs

            # Initial noisy actions x_T: (num_envs, pred_horizon, action_dim)
            naction = torch.randn(
                (B, pred_horizon, action_dim),
                device=device,
            )

            # Init diffusion timesteps
            noise_scheduler.set_timesteps(num_diffusion_iters, device=device)

            # ===== Diffusion denoising loop, record all transitions =====
            for k in noise_scheduler.timesteps:
                naction = naction.detach()

                t = torch.full((B,), k, device=device, dtype=torch.long)
                t_int = int(k)

                # Forward pass (no grad)
                noise_pred = model(
                    sample=naction,
                    timestep=t_int,
                    global_cond=obs_cond,
                )

                # Sample x_{t-1}
                step_out = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t_int,
                    sample=naction,
                )
                naction_next = step_out.prev_sample

                # Store this diffusion transition on CPU, one entry per *active* env
                naction_cpu = naction.detach().cpu()
                cond_cpu = obs_cond.detach().cpu()
                next_latents_cpu = naction_next.detach().cpu()

                for i in range(num_envs):
                    if active_envs[i]:
                        trajectory_data.append({
                            "latents": naction_cpu[i:i+1],          # x_t   shape [1, pred_horizon, action_dim]
                            "t": t_int,
                            "cond": cond_cpu[i:i+1],                # obs_cond shape [1, obs_horizon*obs_dim]
                            "next_latents": next_latents_cpu[i:i+1],# x_{t-1}
                            "episode_idx": episode_idx + i,         # global episode index
                        })

                naction = naction_next

            # ===== Decode final clean actions and step envs =====
            naction_np = naction.detach().cpu().numpy()  # (num_envs, pred_horizon, action_dim)
            action_pred = unnormalize_data(naction_np, stats=stats["action"])

            # Take first action_horizon actions
            start = obs_horizon - 1
            end = start + action_horizon
            # (num_envs, action_horizon, action_dim)
            action_seq = action_pred[:, start:end, :]

            # Execute action_horizon env steps
            for ah in range(action_seq.shape[1]):
                step_actions = action_seq[:, ah, :]  # (num_envs, action_dim)
                next_obs, rewards, terminateds, truncateds, infos = vec_env.step(step_actions)
                dones = np.logical_or(terminateds, truncateds)

                for i in range(num_envs):
                    if active_envs[i]:
                        current_rewards[i] += rewards[i]

                        # Shift history and append new obs
                        obs_history[i] = np.roll(obs_history[i], -1, axis=0)
                        obs_history[i, -1] = next_obs[i]

                        if dones[i] or step_idx + 1 >= max_env_steps:
                            active_envs[i] = False
                            final_rewards[i] = current_rewards[i]

                step_idx += 1
                if not np.any(active_envs):
                    break

    return trajectory_data, final_rewards

def update_model_efficiently(
    model,
    noise_scheduler,
    optimizer,
    trajectory_data,
    returns,
    device,
    lr_scheduler=None,
    batch_size=1024,
    clip_eps=0.2,
    epochs=5,
):
    """
    Vectorized PPO-style update over all diffusion steps from all episodes.

    trajectory_data: list of dicts with keys:
        'latents'      : (1, pred_horizon, action_dim)
        't'            : int timestep
        'cond'         : (1, obs_horizon*obs_dim)
        'next_latents' : (1, pred_horizon, action_dim)
        'episode_idx'  : int
    returns: (num_episodes,) tensor on device
    """
    # ---- 0. Flatten data into big tensors on device ----
    latents = torch.cat([d["latents"] for d in trajectory_data], dim=0).to(device)        # (N, pred_horizon, action_dim)
    timesteps = torch.tensor([d["t"] for d in trajectory_data], dtype=torch.long, device=device)  # (N,)
    conds = torch.cat([d["cond"] for d in trajectory_data], dim=0).to(device)            # (N, obs_horizon*obs_dim)
    next_latents = torch.cat([d["next_latents"] for d in trajectory_data], dim=0).to(device)  # (N, pred_horizon, action_dim)
    episode_indices = torch.tensor([d["episode_idx"] for d in trajectory_data],
                                   dtype=torch.long, device=device)                      # (N,)

    N = latents.shape[0]

    # ---- 1. Compute per-episode advantages, then broadcast per diffusion step ----
    adv_per_episode = (returns - returns.mean()) / (returns.std() + 1e-8)  # (E,)
    advantages = adv_per_episode[episode_indices]                           # (N,)
    advantages = advantages.detach()                                        # no grad through baseline

    # ---- 2. Ensure scheduler buffers on device ----
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    noise_scheduler.alphas = noise_scheduler.alphas.to(device)
    noise_scheduler.betas = noise_scheduler.betas.to(device)
    noise_scheduler.one = noise_scheduler.one.to(device)

    torch_clip_eps = torch.tensor(clip_eps, device=device)

    # ---- 3. Pre-compute old log-probs under current policy (pi_old) ----
    with torch.no_grad():
        # Forward pass once on ALL data
        noise_pred_old = model(
            sample=latents,
            timestep=timesteps,
            global_cond=conds,
        )  # (N, pred_horizon, action_dim)

        # Scheduler math vectorized over timesteps
        alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]  # (N,)
        alpha_prod_t_prev = torch.where(
            timesteps > 0,
            noise_scheduler.alphas_cumprod[timesteps - 1],
            noise_scheduler.one.expand_as(alpha_prod_t),
        )
        beta_prod_t = noise_scheduler.one - alpha_prod_t

        current_alpha_t = noise_scheduler.alphas[timesteps]
        current_beta_t = noise_scheduler.betas[timesteps]

        var_t = current_beta_t * (noise_scheduler.one - alpha_prod_t_prev) / (
            noise_scheduler.one - alpha_prod_t
        )  # (N,)

        # reshape to broadcast over (N, pred_horizon, action_dim)
        alpha_prod_t_b = alpha_prod_t.view(-1, 1, 1)
        beta_prod_t_b = beta_prod_t.view(-1, 1, 1)
        alpha_prod_t_prev_b = alpha_prod_t_prev.view(-1, 1, 1)
        current_alpha_t_b = current_alpha_t.view(-1, 1, 1)
        current_beta_t_b = current_beta_t.view(-1, 1, 1)
        var_t_b = var_t.view(-1, 1, 1)
        var_t_b = torch.clamp(var_t_b, min=1e-4)

        # Posterior mean mu_t (old policy)
        pred_original_sample_old = (
            latents - torch.sqrt(beta_prod_t_b) * noise_pred_old
        ) / torch.sqrt(alpha_prod_t_b)

        posterior_mean_coef1 = torch.sqrt(alpha_prod_t_prev_b) * current_beta_t_b / beta_prod_t_b
        posterior_mean_coef2 = torch.sqrt(current_alpha_t_b) * (
            noise_scheduler.one - alpha_prod_t_prev_b
        ) / beta_prod_t_b

        mu_t_old = posterior_mean_coef1 * pred_original_sample_old + posterior_mean_coef2 * latents

        # Old log-probs: log p_old(x_{t-1} | x_t, c)
        old_log_probs = gaussian_log_prob(
            x=next_latents,
            mean=mu_t_old,
            var=var_t_b,
        )  # (N,)
        old_log_probs = old_log_probs.detach()
        assert torch.all(torch.isfinite(old_log_probs)), old_log_probs


    

    # ---- 4. PPO-style update: loop over mini-batches, recompute new log-probs ----
    indices = torch.randperm(N, device=device)

    model.train()
    total_loss_val = 0.0
    num_batches = 0

    for _ in range(epochs):
      for start_idx in range(0, N, batch_size):
          idx = indices[start_idx:start_idx + batch_size]

          b_latents = latents[idx]              # (B, pred_horizon, action_dim)
          b_t = timesteps[idx]                  # (B,)
          b_cond = conds[idx]                   # (B, obs_horizon*obs_dim)
          b_next_latents = next_latents[idx]    # (B, pred_horizon, action_dim)
          b_adv = advantages[idx]               # (B,)
          b_old_log_probs = old_log_probs[idx]  # (B,)

          # --- new policy forward ---
          noise_pred = model(
              sample=b_latents,
              timestep=b_t,
              global_cond=b_cond,
          )

          # Scheduler math for this batch
          alpha_prod_t = noise_scheduler.alphas_cumprod[b_t]  # (B,)
          alpha_prod_t_prev = torch.where(
              b_t > 0,
              noise_scheduler.alphas_cumprod[b_t - 1],
              noise_scheduler.one.expand_as(alpha_prod_t),
          )
          beta_prod_t = noise_scheduler.one - alpha_prod_t

          current_alpha_t = noise_scheduler.alphas[b_t]
          current_beta_t = noise_scheduler.betas[b_t]

          var_t = current_beta_t * (noise_scheduler.one - alpha_prod_t_prev) / (
              noise_scheduler.one - alpha_prod_t
          )  # (B,)

          # reshape to broadcast for (B, pred_horizon, action_dim)
          alpha_prod_t_b = alpha_prod_t.view(-1, 1, 1)
          beta_prod_t_b = beta_prod_t.view(-1, 1, 1)
          alpha_prod_t_prev_b = alpha_prod_t_prev.view(-1, 1, 1)
          current_alpha_t_b = current_alpha_t.view(-1, 1, 1)
          current_beta_t_b = current_beta_t.view(-1, 1, 1)
          var_t_b = var_t.view(-1, 1, 1)

          pred_original_sample = (
              b_latents - torch.sqrt(beta_prod_t_b) * noise_pred
          ) / torch.sqrt(alpha_prod_t_b)

          posterior_mean_coef1 = torch.sqrt(alpha_prod_t_prev_b) * current_beta_t_b / beta_prod_t_b
          posterior_mean_coef2 = torch.sqrt(current_alpha_t_b) * (
              noise_scheduler.one - alpha_prod_t_prev_b
          ) / beta_prod_t_b

          mu_t = posterior_mean_coef1 * pred_original_sample + posterior_mean_coef2 * b_latents

          # New log-probs
          new_log_probs = gaussian_log_prob(
              x=b_next_latents,
              mean=mu_t,
              var=var_t_b,
          )  # (B,)

          assert b_old_log_probs.shape == new_log_probs.shape

          # --- PPO clipped objective ---
          log_ratio = new_log_probs - b_old_log_probs          # (B,)
          log_ratio = torch.nan_to_num(log_ratio, nan=0.0)
          log_ratio = torch.clamp(log_ratio, -0.15, 0.15)
          ratio = torch.exp(log_ratio)                         # (B,)

        #   print(f"  Ratio Min: {ratio.min().item():.4f}, Max: {ratio.max().item():.4f}")
        #   print(f"  Approx KL: {approx_kl.item():.4f}")

          assert torch.all(torch.isfinite(log_ratio)), "log_ratio is unstable"
          assert torch.all(torch.isfinite(ratio)), "ratio is unstable"

          b_adv = torch.clamp(b_adv, -10.0, 10.0)

          surr1 = ratio * b_adv
          surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
          loss = -torch.min(surr1, surr2).mean()

        #   print(f"Min b_adv: {b_adv.min().item():.4f}, Max b_adv: {b_adv.max().item():.4f}, Mean b_adv: {b_adv.mean().item():.4f}, Std b_adv: {b_adv.std().item():.4f}")
        #   print(f"Min ratio: {ratio.min().item():.4f}, Max ratio: {ratio.max().item():.4f}")
        #   print(f"Min log_ratio: {log_ratio.min().item():.4f}, Max log_ratio: {log_ratio.max().item():.4f}")
        #   print(f"Min old_log_probs: {b_old_log_probs.min().item():.4f}, Max old_log_probs: {b_old_log_probs.max().item():.4f}, shape: {b_old_log_probs.shape}")
        #   print(f"Min new_log_probs: {new_log_probs.min().item():.4f}, Max new_log_probs: {new_log_probs.max().item():.4f}, shape: {new_log_probs.shape}")

          optimizer.zero_grad()
          loss.backward()

          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
          optimizer.step()

          if lr_scheduler is not None:
              lr_scheduler.step()

          total_loss_val += float(loss.item())
          num_batches += 1

    avg_loss = total_loss_val / max(1, num_batches)
    return avg_loss

def collect_trajectories_flat(
    env,
    model,
    noise_scheduler,
    stats,
    batch_size,
    obs_horizon,
    pred_horizon,
    action_horizon,
    num_diffusion_iters,
    device,
    max_env_steps=100,
):
    """
    Parallel version using a VectorEnv.

    env: VectorEnv with env.num_envs == batch_size.

    Returns:
      - trajectory_data: list of per-diffusion-step dicts (same format as before)
      - returns: tensor of shape (batch_size,) on device
    """
    assert hasattr(env, "num_envs"), "Expected a Gymnasium VectorEnv"
    assert env.num_envs == batch_size, (
        f"VectorEnv num_envs ({env.num_envs}) must equal batch_size ({batch_size})"
    )

    # Collect one episode per env in parallel
    with torch.no_grad():
        trajectory_data, final_rewards = rollout_ddpo_collect_flat(
            env=env,
            model=model,
            noise_scheduler=noise_scheduler,
            stats=stats,
            episode_idx=0,  # starting global episode index
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            num_diffusion_iters=num_diffusion_iters,
            device=device,
            max_env_steps=max_env_steps,
        )

    # final_rewards: (num_envs,) np array
    returns = torch.tensor(final_rewards, device=device, dtype=torch.float32)
    return trajectory_data, returns
