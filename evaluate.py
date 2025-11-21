import torch
import numpy as np
from tqdm.auto import tqdm
from skvideo.io import vwrite
from pusht_env import PushTAdapter
from gymnasium.vector import SyncVectorEnv
from dataset import normalize_data, unnormalize_data

def _make_grid_image(images, rows=None, cols=None):
    """
    images: list of (H, W, C) np arrays
    rows, cols: grid shape. If None, choose roughly square.
    """
    n = len(images)
    if n == 0:
        return None

    if rows is None or cols is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

    h, w, c = images[0].shape
    grid = np.zeros((rows * h, cols * w, c), dtype=images[0].dtype)

    for idx, img in enumerate(images):
        r = idx // cols
        c_ = idx % cols
        if r >= rows or c_ >= cols:
            break
        grid[r*h:(r+1)*h, c_*w:(c_+1)*w, :] = img

    return grid

def evaluate_push_t(
    model, 
    noise_scheduler, 
    stats, 
    device, 
    max_steps=200, 
    seed=100000, 
    filepath='vis.mp4',
    obs_horizon=2, 
    pred_horizon=16, 
    action_horizon=8, 
    action_dim=2, 
    num_diffusion_iters=100, 
    num_evals=15
):
    # 1. Environment Setup
    env = SyncVectorEnv([lambda: PushTAdapter() for _ in range(num_evals)])
    model.eval()
    
    # 2. Reset and Init
    obs, info = env.reset(seed=seed)
    
    obs_history = np.tile(obs[:, None, :], (1, obs_horizon, 1))
    
    max_rewards = np.zeros(num_evals)
    current_rewards = np.zeros(num_evals)
    
    # ---- GRID VIDEO: initial frame ----
    # render every env and tile into one big frame
    per_env_frames = [e.render() for e in env.envs]
    grid_frame = _make_grid_image(per_env_frames)
    imgs = [grid_frame]

    step_idx = 0
    pbar = tqdm(total=max_steps, desc="Eval PushT (Vectorized)", leave=False)

    # 3. Inference Loop
    while step_idx < max_steps:
        B = num_evals
        
        nobs = normalize_data(obs_history, stats=stats['obs'])
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        with torch.no_grad():
            obs_cond = nobs.flatten(start_dim=1)

            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device
            )
            naction = noisy_action

            noise_scheduler.set_timesteps(num_diffusion_iters, device=device)

            for k in noise_scheduler.timesteps:
                noise_pred = model(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        naction = naction.detach().to('cpu').numpy()
        action_pred = unnormalize_data(naction, stats=stats['action'])

        start = obs_horizon - 1
        end = start + action_horizon
        action_chunk = action_pred[:, start:end, :] 
        
        for i in range(action_chunk.shape[1]):
            if step_idx >= max_steps:
                break

            action_t = action_chunk[:, i, :]
            obs, reward, done, truncated, info = env.step(action_t)
            
            obs_history = np.roll(obs_history, shift=-1, axis=1)
            obs_history[:, -1, :] = obs
            
            current_rewards = reward
            max_rewards = np.maximum(max_rewards, current_rewards)
            
            # ---- GRID VIDEO: render all envs each step ----
            per_env_frames = [e.render() for e in env.envs]
            grid_frame = _make_grid_image(per_env_frames)
            imgs.append(grid_frame)
            
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(max_score=f"{np.mean(max_rewards):.2f}")

    pbar.close()
    
    avg_max_reward = np.mean(max_rewards)
    print(f'Evaluation Complete. Avg Max Score: {avg_max_reward:.4f}')

    success_rate = np.mean(max_rewards >= 0.95)
    print(f'Success Rate (>= 0.95): {success_rate:.4f}')
    
    vwrite(filepath, imgs)
    env.close()
    
    return avg_max_reward, success_rate
