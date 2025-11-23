import math
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
                  We follow ddpo-pytorch and return the MEAN over non-batch dims,
                  not the sum, so magnitudes stay O(1–10).
    """
    # Keep variance strictly positive but don't distort it too much
    var = torch.clamp(var, min=1e-5)

    log_prob_per_dim = -0.5 * (
        ((x - mean) ** 2) / var + torch.log(2 * math.pi * var)
    )
    # Mean over all non-batch dimensions -> shape (B,)
    if log_prob_per_dim.ndim > 1:
        log_prob = log_prob_per_dim.flatten(start_dim=1).mean(dim=1)
    else:
        log_prob = log_prob_per_dim
    return log_prob


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


def analytical_kl(mean1: torch.Tensor,
                  var1: torch.Tensor,
                  mean2: torch.Tensor,
                  var2: torch.Tensor,
                  eps: float = 1e-8) -> torch.Tensor:
    """
    KL( N(mean1, var1) || N(mean2, var2) ) for diagonal Gaussians.
    Not used as a penalty in the loss here, but kept for potential diagnostics.
    """
    var1 = torch.clamp(var1, min=eps)
    var2 = torch.clamp(var2, min=eps)

    log_ratio = torch.log(var2) - torch.log(var1)        # log(var2 / var1)
    sq_mean = (mean1 - mean2) ** 2

    kl_per_dim = 0.5 * (log_ratio + (var1 + sq_mean) / var2 - 1.0)

    if kl_per_dim.ndim > 1:
        return kl_per_dim.flatten(start_dim=1).mean(dim=1)  # (B,)
    else:
        return kl_per_dim


# ---------------------------------------------------------------------------
#  Rollout collection (vectorized over envs)
# ---------------------------------------------------------------------------

def rollout_ddpo_collect_flat(
    env,
    model,
    noise_scheduler,
    stats,
    episode_idx: int,
    obs_horizon,
    pred_horizon,
    action_horizon,
    num_diffusion_iters,
    device,
    max_env_steps=100,
    seed_options=[42, 100000]
):
    vec_env = env
    num_envs = vec_env.num_envs
    action_dim = vec_env.single_action_space.shape[0]

    trajectory_data = []

    # Track rewards over full episodes (for logging)
    current_rewards = np.zeros(num_envs, dtype=np.float32)
    final_rewards = np.zeros(num_envs, dtype=np.float32)
    active_envs = np.ones(num_envs, dtype=bool)

    # obs, infos = vec_env.reset()

    obs, infos = [], []

    # Reset envs
    with torch.no_grad():
        # shared_seed = 1000 + episode_idx 
        for e in vec_env.envs: 
            random_seed = np.random.choice(seed_options)
            ob, info = e.reset(seed=int(random_seed))
            obs.append(ob)
            infos.append(info)
        
        # print("Seed", random_seed, "first obs[0]:", obs[0])

        obs = np.array(obs)
        infos = np.array(infos)

        # assert obs are uniform
        # assert np.all(obs == obs[0]), "Env resets returned non-uniform observations"
        # assert np.all(infos == infos[0]), "Env resets returned non-uniform infos"

    # [num_envs, obs_horizon, obs_dim]
    obs_history = np.tile(obs[:, None, :], (1, obs_horizon, 1))
    step_idx = 0

    # Move scheduler buffers once
    with torch.no_grad():
        noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
        noise_scheduler.alphas = noise_scheduler.alphas.to(device)
        noise_scheduler.betas = noise_scheduler.betas.to(device)
        noise_scheduler.one = torch.tensor(1.0, device=device)

    while np.any(active_envs) and step_idx < max_env_steps:
        # Snapshot of which envs are alive at the *start* of this 8-step chunk
        block_active_envs = active_envs.copy()
        if not np.any(block_active_envs):
            break

        with torch.no_grad():
            # --- Prepare Conditioning ---
            # Use obs_history for all envs, but we'll only keep entries for block_active_envs
            nobs = normalize_data(obs_history, stats=stats["obs"])
            nobs_t = torch.from_numpy(nobs).to(device, dtype=torch.float32)
            obs_cond = nobs_t.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

            B = num_envs

            # --- Pre-allocate Trajectory Storage on GPU ---
            traj_latents = torch.zeros(
                (num_diffusion_iters, B, pred_horizon, action_dim), device=device
            )
            traj_next_latents = torch.zeros(
                (num_diffusion_iters, B, pred_horizon, action_dim), device=device
            )
            traj_timesteps = torch.zeros(
                (num_diffusion_iters,), device=device, dtype=torch.long
            )

            # Initial noise
            naction = torch.randn((B, pred_horizon, action_dim), device=device)
            noise_scheduler.set_timesteps(num_diffusion_iters, device=device)

            # --- Diffusion Loop (pure GPU) ---
            for idx, k in enumerate(noise_scheduler.timesteps):
                naction = naction.detach()
                t = torch.full((B,), k, device=device, dtype=torch.long)

                # Record current state and time
                traj_latents[idx] = naction
                traj_timesteps[idx] = k

                # Forward pass
                noise_pred = model(
                    sample=naction,
                    timestep=int(k),
                    global_cond=obs_cond,
                )

                # Scheduler step
                step_out = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=int(k),
                    sample=naction,
                )
                naction_next = step_out.prev_sample

                # Record next state
                traj_next_latents[idx] = naction_next

                # Update for next loop
                naction = naction_next

            # --- Bulk Transfer & Formatting (CPU Side) ---
            all_latents_cpu = traj_latents.cpu()            # (T, B, pred_horizon, action_dim)
            all_next_latents_cpu = traj_next_latents.cpu()  # (T, B, pred_horizon, action_dim)
            all_timesteps_cpu = traj_timesteps.cpu()        # (T,)
            cond_cpu = obs_cond.cpu()                       # (B, obs_horizon * obs_dim)

            # --- Environment Execution for this block (8 actions max) ---
            naction_np = naction.detach().cpu().numpy()
            action_pred = unnormalize_data(naction_np, stats=stats["action"])

            # Slice the 8 actions we actually execute
            start = obs_horizon - 1
            end = start + action_horizon  # pred_horizon == action_horizon in your setup
            action_seq = action_pred[:, start:end, :]       # (B, action_horizon, action_dim)

            # Per-block reward and length (per-env)
            segment_returns = np.zeros(num_envs, dtype=np.float32)
            segment_lengths = np.zeros(num_envs, dtype=np.int32)

            for ah in range(action_seq.shape[1]):
                step_actions = action_seq[:, ah, :]

                # Zero out actions for envs that are already inactive
                masked_actions = step_actions.copy()
                masked_actions[~active_envs] = 0.0

                next_obs, rewards, terminateds, truncateds, infos = vec_env.step(
                    masked_actions
                )
                dones = np.logical_or(terminateds, truncateds)

                for i in range(num_envs):
                    if active_envs[i]:
                        r = rewards[i]
                        # Update per-block and per-episode rewards
                        segment_returns[i] += r
                        segment_lengths[i] += 1
                        # current_rewards[i] = max(current_rewards[i], r)
                        current_rewards[i] += r

                        # Roll observation history
                        obs_history[i] = np.roll(obs_history[i], -1, axis=0)
                        obs_history[i, -1] = next_obs[i]

                        # Episode done?
                        if dones[i] or step_idx + 1 >= max_env_steps:
                            active_envs[i] = False
                            final_rewards[i] = current_rewards[i]

                step_idx += 1
                if not np.any(active_envs) or step_idx >= max_env_steps:
                    break

            # --- Attach per-chunk returns to all diffusion states from this block ---
            # Only envs that were alive at the *start* of the block get data.
            for t_idx in range(num_diffusion_iters):
                t_val = int(all_timesteps_cpu[t_idx])
                for i in range(num_envs):
                    if block_active_envs[i]:
                        trajectory_data.append(
                            {
                                "latents": all_latents_cpu[t_idx, i : i + 1],
                                "t": t_val,
                                "cond": cond_cpu[i : i + 1],
                                "next_latents": all_next_latents_cpu[t_idx, i : i + 1],
                                "episode_idx": episode_idx + i,
                            }
                        )

    # final_rewards is still per full episode (for logging / eval)
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
    num_batches_per_epoch=10,
    clip_eps=0.05,   # slightly more conservative than vanilla PPO
    epochs=5,
    global_step=0,
    action_horizon=None,   # NEW: how many actions were actually executed
    obs_horizon=None,      # NEW: to align with rollout slicing (start = obs_horizon - 1)
    gradient_clip=0.5
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

    If action_horizon is not None, we only compute log-probs over the
    executed slice of the horizon:
        start_idx = (obs_horizon - 1) if obs_horizon is given else 0
        end_idx   = start_idx + action_horizon
    """
    # ---- 0. Flatten data into big tensors on device ----
    latents = torch.cat([d["latents"] for d in trajectory_data], dim=0).to(device)
    timesteps = torch.tensor(
        [d["t"] for d in trajectory_data],
        dtype=torch.long, device=device
    )
    conds = torch.cat([d["cond"] for d in trajectory_data], dim=0).to(device)
    next_latents = torch.cat(
        [d["next_latents"] for d in trajectory_data],
        dim=0
    ).to(device)
    episode_indices = torch.tensor(
        [d["episode_idx"] for d in trajectory_data],
        dtype=torch.long, device=device
    )
    N = latents.shape[0]

    # ---- 0.5 Figure out which slice of the prediction horizon to use ----
    pred_horizon = latents.shape[1]
    use_slice = (action_horizon is not None) and (latents.ndim == 3)
    use_slice = False
    if use_slice:
        # latents: (N, pred_horizon, action_dim)
        assert action_horizon <= pred_horizon, (
            f"action_horizon ({action_horizon}) > pred_horizon ({pred_horizon})"
        )

        # If you want literal "take first 8, discard next 8", set obs_horizon = 1
        if obs_horizon is None:
            start_idx = 0
        else:
            start_idx = obs_horizon - 2

        end_idx = start_idx + action_horizon + 1
        assert end_idx <= pred_horizon, (
            f"Slice [{start_idx}:{end_idx}] exceeds pred_horizon={pred_horizon}"
        )
    else:
        start_idx = None
        end_idx = None

    # ---- 1. Compute per-episode advantages (single normalization) ----
    adv_per_episode = (returns - returns.mean()) / (returns.std() + 1e-8)  # (E,)
    advantages = adv_per_episode[episode_indices]                           # (N,)
    advantages = advantages.detach()
    advantages = torch.clamp(advantages, -5.0, 5.0)

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

        if use_slice:
            # Only executed actions along horizon
            mu_old_exec = mu_t_old[:, :action_horizon + 1, :]
            next_old_exec = next_latents[:, :action_horizon + 1, :]
            var_old_exec = var_t_old
        else:
            mu_old_exec = mu_t_old
            next_old_exec = next_latents
            var_old_exec = var_t_old

        old_log_probs = gaussian_log_prob(
            x=next_old_exec,
            mean=mu_old_exec,
            var=var_old_exec,
        )
        old_log_probs = old_log_probs.detach()
        assert torch.all(torch.isfinite(old_log_probs)), old_log_probs

    # ---- 4. PPO-style update ----
    model.train()
    total_loss_val = 0.0
    total_approx_kl = 0.0
    num_batches = 0

    batch_size = max(1, N // num_batches_per_epoch)

    for _ in range(epochs):
        indices = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = indices[start:start + batch_size]

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

            if use_slice:
                mu_exec = mu_t[:, :action_horizon + 1, :]
                next_exec = b_next_latents[:, :action_horizon + 1, :]
                var_exec = var_t
            else:
                mu_exec = mu_t
                next_exec = b_next_latents
                var_exec = var_t

            new_log_probs = gaussian_log_prob(
                x=next_exec,
                mean=mu_exec,
                var=var_exec,
            )

            assert b_old_log_probs.shape == new_log_probs.shape

            # --- PPO clipped objective ---
            raw_log_ratio = new_log_probs - b_old_log_probs           # (B,)
            raw_log_ratio = torch.nan_to_num(raw_log_ratio, nan=0.0)

            # Clamp in log-space to avoid exp overflow
            log_ratio = torch.clamp(raw_log_ratio, -10.0, 10.0)
            ratio = torch.exp(log_ratio)                              # (B,)

            # Approximate KL for logging (as in ddpo-pytorch)
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

            assert torch.all(torch.isfinite(log_ratio)), (
                f"log_ratio is unstable.\nMin: {log_ratio.min().item()}, "
                f"Max: {log_ratio.max().item()}"
            )
            assert torch.all(torch.isfinite(ratio)), (
                f"ratio is unstable.\nMin: {ratio.min().item()}, "
                f"Max: {ratio.max().item()}"
            )

            b_adv = torch.clamp(b_adv, -5.0, 5.0)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            loss = -torch.min(surr1, surr2).mean()

            optimizer.zero_grad()
            loss.backward()

            # Gradient norm logging + clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            total_loss_val += float(loss.item())
            num_batches += 1

            wandb.log({
                "gradients/total_norm": total_norm,
                "ppo/loss": loss.item(),
                "global_step": global_step,
            })
            global_step += 1

    avg_loss = total_loss_val / max(1, num_batches)
    avg_approx_kl = total_approx_kl / max(1, num_batches)

    if lr_scheduler is not None:
        lr_scheduler.step()

    return avg_loss, avg_approx_kl, global_step



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
    episode_idx=0,
    initialization_seeds=[42, 100000],
):
    """
    Parallel version using a VectorEnv.

    env: VectorEnv with env.num_envs == batch_size.

    Returns:
      - trajectory_data: list of per-diffusion-step dicts
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
            episode_idx=episode_idx,  # starting global episode index
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            num_diffusion_iters=num_diffusion_iters,
            device=device,
            max_env_steps=max_env_steps,
            seed_options=initialization_seeds,
        )

    returns = torch.tensor(final_rewards, device=device, dtype=torch.float32)
    return trajectory_data, returns
