import torch
import wandb
import numpy as np
from dataset import normalize_data, unnormalize_data


# ---------------------------------------------------------------------------
#  Gaussian log-prob helper
# ---------------------------------------------------------------------------

def gaussian_log_prob(x: torch.Tensor,
                      mean: torch.Tensor,
                      var: torch.Tensor) -> torch.Tensor:
    """
    Diagonal multivariate Gaussian log-prob.

    Args:
        x, mean, var: tensors broadcastable to the same shape (B, ...).
                      var is the diagonal variance.

    Returns:
        log_prob: shape (B,), the log p(x | mean, var) for each batch element.
    """
    # Keep variance strictly positive but don't distort it too much
    var = torch.clamp(var, min=1e-5)

    log_prob_per_dim = -0.5 * (
        ((x - mean) ** 2) / var + torch.log(2 * torch.pi * var)
    )
    # True log-prob = SUM over all non-batch dimensions
    return log_prob_per_dim.flatten(start_dim=1).sum(dim=1)


def _expand_like(x_1d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Expand a (B,) vector to match the shape (B, 1, 1, ...) of ref for broadcasting.
    """
    return x_1d.view(-1, *([1] * (ref.ndim - 1))).to(ref.device, ref.dtype)


def compute_posterior_mean_var(
    noise_scheduler,
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
):
    """
    Compute the DDPM posterior p(x_{t-1} | x_t, c) parameters (mean, var)
    in a way consistent between old and new policies.

    Args:
        noise_scheduler: a diffusers-style scheduler with
            .alphas_cumprod, .alphas, .betas, .one buffers
        x_t:       (B, ...) current latents
        noise_pred:(B, ...) predicted noise epsilon_theta(x_t, t, c)
        timesteps: (B,) integer timesteps (same as those used in rollout)

    Returns:
        mu_t:   (B, ...) posterior mean
        var_t:  (B, ...) posterior variance (diagonal)
    """
    device = x_t.device

    alphas_cumprod = noise_scheduler.alphas_cumprod
    alphas = noise_scheduler.alphas
    betas = noise_scheduler.betas
    one = noise_scheduler.one

    # (B,)
    alpha_prod_t = alphas_cumprod[timesteps]
    alpha_prod_t_prev = torch.where(
        timesteps > 0,
        alphas_cumprod[timesteps - 1],
        one.expand_as(alpha_prod_t),
    )
    beta_prod_t = one - alpha_prod_t

    alpha_t = alphas[timesteps]
    beta_t = betas[timesteps]

    # Posterior variance (fixed-small)
    var_t_scalar = beta_t * (one - alpha_prod_t_prev) / (one - alpha_prod_t)
    var_t_scalar = torch.clamp(var_t_scalar, min=1e-5)

    # Expand scalars to match x_t shape
    alpha_prod_t_b = _expand_like(alpha_prod_t, x_t)
    alpha_prod_t_prev_b = _expand_like(alpha_prod_t_prev, x_t)
    beta_prod_t_b = _expand_like(beta_prod_t, x_t)
    alpha_t_b = _expand_like(alpha_t, x_t)
    beta_t_b = _expand_like(beta_t, x_t)
    var_t = _expand_like(var_t_scalar, x_t)

    # Reconstruct x_0 from x_t and predicted noise
    x0_pred = (x_t - torch.sqrt(beta_prod_t_b) * noise_pred) / torch.sqrt(alpha_prod_t_b)

    # Posterior mean coefficients (standard DDPM form)
    posterior_mean_coef1 = (torch.sqrt(alpha_prod_t_prev_b) * beta_t_b) / (one - alpha_prod_t_b)
    posterior_mean_coef2 = (
        torch.sqrt(alpha_t_b) * (one - alpha_prod_t_prev_b) / (one - alpha_prod_t_b)
    )

    mu_t = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t
    return mu_t, var_t


def analytical_kl(mean1, var1, mean2, var2):
    """KL(N(mean1, var1) || N(mean2, var2))"""
    # var is diagonal, so we sum over dimensions
    numerator = (mean1 - mean2)**2 + var1 - var2
    denominator = 2 * var2
    # Log variance ratio (var1 / var2)
    log_ratio = torch.log(var2) - torch.log(var1) 
    
    kl = 0.5 * (log_ratio + numerator / denominator - 1)
    return kl.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
#  Rollout collection (vectorized over envs)
# ---------------------------------------------------------------------------

def rollout_ddpo_collect_flat(
    env,
    model,
    noise_scheduler,
    stats,
    episode_idx: int,   # interpreted as "episode_start_idx" for vec env
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
        if not np.any(active_envs):
            break

        with torch.no_grad():
            # ===== Build conditioning for all envs =====
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
                            "latents": naction_cpu[i:i+1],
                            "t": t_int,
                            "cond": cond_cpu[i:i+1],
                            "next_latents": next_latents_cpu[i:i+1],
                            "episode_idx": episode_idx + i,
                        })

                naction = naction_next

            # ===== Decode final clean actions and step envs =====
            naction_np = naction.detach().cpu().numpy()  # (num_envs, pred_horizon, action_dim)
            action_pred = unnormalize_data(naction_np, stats=stats["action"])

            # Take first action_horizon actions
            start = obs_horizon - 1
            end = start + action_horizon
            action_seq = action_pred[:, start:end, :]  # (num_envs, action_horizon, action_dim)

            # Execute action_horizon env steps
            for ah in range(action_seq.shape[1]):
                step_actions = action_seq[:, ah, :]

                # Zero out actions for inactive envs to avoid weird extra stepping
                masked_actions = step_actions.copy()
                masked_actions[~active_envs] = 0.0

                next_obs, rewards, terminateds, truncateds, infos = vec_env.step(masked_actions)
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
                if not np.any(active_envs) or step_idx >= max_env_steps:
                    break

    return trajectory_data, final_rewards


# ---------------------------------------------------------------------------
#  PPO-style update over collected diffusion transitions
# ---------------------------------------------------------------------------

def update_model_efficiently(
    model,
    noise_scheduler,
    optimizer,
    trajectory_data,
    returns,
    device,
    lr_scheduler=None,
    # batch_size=1024,
    num_batches_per_epoch=10,
    clip_eps=0.2,
    epochs=5,
    global_step=0,
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
    latents = torch.cat([d["latents"] for d in trajectory_data], dim=0).to(device)
    timesteps = torch.tensor([d["t"] for d in trajectory_data],
                             dtype=torch.long, device=device)
    conds = torch.cat([d["cond"] for d in trajectory_data], dim=0).to(device)
    next_latents = torch.cat([d["next_latents"] for d in trajectory_data],
                             dim=0).to(device)
    episode_indices = torch.tensor(
        [d["episode_idx"] for d in trajectory_data],
        dtype=torch.long, device=device
    )

    N = latents.shape[0]

    # ---- 1. Compute per-episode advantages (single normalization) ----
    adv_per_episode = (returns - returns.mean()) / (returns.std() + 1e-8)  # (E,)
    advantages = adv_per_episode[episode_indices]                           # (N,)
    advantages = advantages.detach()

    # ---- 2. Ensure scheduler buffers on device ----
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    noise_scheduler.alphas = noise_scheduler.alphas.to(device)
    noise_scheduler.betas = noise_scheduler.betas.to(device)
    noise_scheduler.one = noise_scheduler.one.to(device)

    # ---- 3. Pre-compute old log-probs under current policy (pi_old) ----
    with torch.no_grad():
        noise_pred_old = model(
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

        old_log_probs = gaussian_log_prob(
            x=next_latents,
            mean=mu_t_old,
            var=var_t_old,
        )
        old_log_probs = old_log_probs.detach()
        assert torch.all(torch.isfinite(old_log_probs)), old_log_probs

    # ---- 4. PPO-style update: loop over mini-batches, recompute new log-probs ----
    indices = torch.randperm(N, device=device)

    model.train()
    total_loss_val = 0.0
    total_kl = 0.0
    num_batches = 0

    batch_size = N // num_batches_per_epoch

    for _ in range(epochs):
        for start_idx in range(0, N, batch_size):
            idx = indices[start_idx:start_idx + batch_size]

            b_latents = latents[idx]
            b_t = timesteps[idx]
            b_cond = conds[idx]
            b_next_latents = next_latents[idx]
            b_adv = advantages[idx]
            b_old_log_probs = old_log_probs[idx]

            # --- new policy forward ---
            noise_pred = model(
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

            assert b_old_log_probs.shape == new_log_probs.shape

            # --- PPO clipped objective ---
            log_ratio = new_log_probs - b_old_log_probs           # (B,)
            log_ratio = torch.nan_to_num(log_ratio, nan=0.0)
            if (log_ratio.abs() > 5).any():
                print("Warning: unstable log_ratio in PPO update, using clamped version")
                stable_log_ratio = torch.clamp(log_ratio, -5, 5)
                ratio = torch.exp(stable_log_ratio)
            else: ratio = torch.exp(log_ratio)                          # (B,)

            wandb.log({
                "stability/log_ratio_mean": log_ratio.mean().item(),
                "stability/log_ratio_max": log_ratio.max().item(),
                "stability/log_ratio_min": log_ratio.min().item(),
                "stability/log_ratio_std": log_ratio.std().item(),
                "stability/ratio_mean": ratio.mean().item(),
                "stability/ratio_max": ratio.max().item(),
                "stability/ratio_min": ratio.min().item(),
                "stability/ratio_std": ratio.std().item(),
                "global_step": global_step,
            })

            assert torch.all(torch.isfinite(log_ratio)), f"log_ratio is unstable.\nMin: {log_ratio.min().item()}, Max: {log_ratio.max().item()}"
            assert torch.all(torch.isfinite(ratio)), f"ratio is unstable.\nMin: {ratio.min().item()}, Max: {ratio.max().item()}"

            b_adv = torch.clamp(b_adv, -10.0, 10.0)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            loss = -torch.min(surr1, surr2).mean()

            loss_kl = analytical_kl(
                mean1=mu_t_old[idx],
                var1=var_t_old[idx],
                mean2=mu_t,
                var2=var_t,
            ).mean() * 10

            total_kl += loss_kl.detach()

            loss = loss + loss_kl

            optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            total_loss_val += float(loss.item())
            num_batches += 1

            wandb.log({
                "gradients/total_norm": total_norm, e
                "global_step": global_step, 
            })
            global_step += 1

    avg_loss = total_loss_val / max(1, num_batches)
    return avg_loss, total_kl.mean().item(), global_step


# ---------------------------------------------------------------------------
#  Top-level trajectory collection helper
# ---------------------------------------------------------------------------

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

    returns = torch.tensor(final_rewards, device=device, dtype=torch.float32)
    return trajectory_data, returns
