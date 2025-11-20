import torch
import numpy as np
import collections
from tqdm.auto import tqdm
from skvideo.io import vwrite
from pusht_env import PushTEnv
from dataset import normalize_data, unnormalize_data

def evaluate_push_t(
    model, 
    noise_scheduler, 
    stats, 
    device, 
    max_steps=200, 
    seed=100000, 
    filepath='vis.mp4',
    obs_horizon=2,     # Default based on your snippet
    pred_horizon=16,   # You might need to adjust these defaults 
    action_horizon=8,  # based on your specific config
    action_dim=2,       # based on PushT
    num_diffusion_iters=100 # Default value, should be passed or set
):
    """
    Evaluates the diffusion policy on the PushT environment and saves a video.
    """
    
    # 1. Environment Setup
    # Create a fresh environment for evaluation to avoid affecting training RNG
    env = PushTEnv()
    env.seed(seed)
    
    # Ensure model is in eval mode
    model.eval()
    
    # 2. Reset and Init
    obs, info = env.reset()
    
    # Keep a queue of last steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon
    )
    
    # Storage for visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0
    
    # 3. Inference Loop
    # We use a tqdm bar to visualize progress during the eval
    with tqdm(total=max_steps, desc="Eval PushT", leave=False) as pbar:
        while not done:
            B = 1
            # Stack the last obs_horizon observations
            obs_seq = np.stack(obs_deque)
            
            # Normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # Infer action
            with torch.no_grad():
                # Reshape observation to (B, obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # Initialize action from Gaussian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device
                )
                naction = noisy_action

                # Init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters, device=device)

                # Denoising loop
                for k in noise_scheduler.timesteps:
                    # Predict noise
                    noise_pred = model(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # Inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # Unnormalize action
            naction = naction.detach().to('cpu').numpy()
            naction = naction[0] # (pred_horizon, action_dim)
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # Action Execution (Receding Horizon Control)
            # Only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end, :] # (action_horizon, action_dim)

            # Execute action_horizon number of steps without replanning
            for i in range(len(action)):
                # Stepping env
                obs, reward, done, _, info = env.step(action[i])
                
                # Save observations
                obs_deque.append(obs)
                
                # Save reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # Update progress
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                
                # Termination conditions
                if step_idx > max_steps:
                    done = True
                if done:
                    break
    
    # 4. Finalize and Save
    max_reward = max(rewards) if rewards else 0
    print(f'Evaluation Complete. Max Score: {max_reward}')
    
    # Save video
    vwrite(filepath, imgs)
    
    # Clean up
    env.close()
    
    return max_reward
