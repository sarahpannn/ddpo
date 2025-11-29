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
    seed_options=None,
    sparse_rewards=False,
):
    vec_env = env
    num_envs = vec_env.num_envs
    action_dim = vec_env.single_action_space.shape[0]

    trajectory_data = []

    # Track rewards over full episodes (for logging)
    current_rewards = np.zeros(num_envs, dtype=np.float32)
    final_rewards = np.zeros(num_envs, dtype=np.float32)
    active_envs = np.ones(num_envs, dtype=bool)

    init_seeds_for_env = np.zeros(num_envs, dtype=np.int64)

    obs, infos = [], []
    episode_meta = [] 

    # Reset envs
    with torch.no_grad():
        if seed_options is not None:
            for i, e in enumerate(vec_env.envs): 
                random_seed = np.random.choice(seed_options)
                init_seeds_for_env[i] = random_seed
                ob, info = e.reset(seed=int(random_seed))
                obs.append(ob)
                infos.append(info)

            obs = np.array(obs)
            infos = np.array(infos)

        else: 
            for i, e in enumerate(vec_env.envs):
                seed = 42 + i
                ob, info = e.reset(seed=int(seed))
                init_seeds_for_env[i] = seed
                obs.append(ob)
                infos.append(info)

            obs = np.array(obs)
            infos = np.array(infos)
    
    # [num_envs, obs_horizon, obs_dim]
    obs_history = np.tile(obs[:, None, :], (1, obs_horizon, 1))
    step_idx = 0
    
    # ============== NEW: Track env_step per environment ==============
    env_step_counter = np.zeros(num_envs, dtype=np.int32)
    # =================================================================

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
        
        # ============== NEW: Snapshot env_step at start of block ==============
        block_env_steps = env_step_counter.copy()
        # ======================================================================

        with torch.no_grad():
            # --- Prepare Conditioning ---
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
            end = start + action_horizon
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
                        current_rewards[i] += r

                        # Roll observation history
                        obs_history[i] = np.roll(obs_history[i], -1, axis=0)
                        obs_history[i, -1] = next_obs[i]

                        # Episode done?
                        if dones[i] or step_idx + 1 >= max_env_steps:
                            active_envs[i] = False
                            final_rewards[i] = current_rewards[i]

                            episode_meta.append(
                                {
                                    "episode_idx": int(episode_idx + i),
                                    "init_seed": int(init_seeds_for_env[i]),
                                    "final_reward": float(current_rewards[i]),
                                }
                            )

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
                                "env_step": int(block_env_steps[i]),
                                "init_seed": int(init_seeds_for_env[i]),
                                "reward": float(segment_returns[i] if t_val == 0 else 0.0),
                            }
                        )
            
            # ============== NEW: Increment env_step for active envs ==============
            for i in range(num_envs):
                if block_active_envs[i]:
                    env_step_counter[i] += 1
            # =====================================================================

    # final_rewards is still per full episode (for logging / eval)
    return trajectory_data, final_rewards, episode_meta


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
    initialization_seeds=None,
    sparse_rewards=False,
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
        trajectory_data, final_rewards, meta = rollout_ddpo_collect_flat(
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
            sparse_rewards=sparse_rewards,
        )

    returns = torch.tensor(final_rewards, device=device, dtype=torch.float32)
    return trajectory_data, returns, meta