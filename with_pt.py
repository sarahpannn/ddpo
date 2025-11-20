#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/gist/sarahpannn/80281fa604f6455419376b7d086c645d/copy-of-diffusion_policy_state_pusht_demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:

#@markdown ### **Imports**
# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
import huggingface_hub
import huggingface_hub.utils
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import wandb
import time

# env import
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

torch.manual_seed(42)


# In[3]:


#@title
# @markdown ### **Environment**
# @markdown Defines a PyMunk-based Push-T environment `PushTEnv`.
# @markdown
# @markdown **Goal**: push the gray T-block into the green area.
# @markdown
# @markdown Adapted from [Implicit Behavior Cloning](https://implicitbc.github.io/)

TYPE_AGENT = 1
TYPE_BLOCK = 2
TYPE_WALL  = 3

positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function wont actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordiante system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

        circle_edge = pos + Vec2d(radius, 0).rotated(angle)
        p2 = to_pygame(circle_edge, self.surface)
        line_r = 2 if radius > 20 else 1
        # pygame.draw.lines(self.surface, outline_color.as_int(), False, [p, p2], line_r)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

# env
class PushTEnv(gym.Env): # Changed from gym.Env to gymnasium.Env
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False,
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        super().__init__() # Added call to super().__init__()
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatiblity
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

    def _install_collision_handlers(self):
        """Register collision callbacks in a way that works across pymunk builds."""
        self.n_contact_points = 0

        def _post_solve(arbiter, space, data):
            # Only count contacts that involve the BLOCK (so you don't count every wall-wall etc.)
            a, b = arbiter.shapes
            if TYPE_BLOCK in (a.collision_type, b.collision_type):
                cps = arbiter.contact_point_set
                self.n_contact_points += len(getattr(cps, "points", []))
            # no return needed for post_solve

        # Try the most specific API first, then fall back gracefully
        if hasattr(self.space, "add_collision_handler"):
            # fire for block-vs-agent and block-vs-wall
            h1 = self.space.add_collision_handler(TYPE_BLOCK, TYPE_AGENT)
            h1.post_solve = _post_solve
            h2 = self.space.add_collision_handler(TYPE_BLOCK, TYPE_WALL)
            h2.post_solve = _post_solve

        elif hasattr(self.space, "add_wildcard_collision_handler"):
            # fire for any collision involving the block
            h = self.space.add_wildcard_collision_handler(TYPE_BLOCK)
            h.post_solve = _post_solve

        elif hasattr(self.space, "add_default_collision_handler"):
            # last resort: fire for all collisions, filter inside callback
            h = self.space.add_default_collision_handler()
            h.post_solve = _post_solve

        else:
            raise RuntimeError("No collision handler API found on pymunk.Space")

    def reset(self, seed=None, options=None): # Modified reset method signature for Gymnasium
        super().reset(seed=seed) # Added call to super().reset()
        self._seed = seed # Keep original seed logic if needed, or rely on super().reset()
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatiblity
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _count_contacts_now(self) -> int:
        """Count agent–block contact points using current arbiters (no handlers needed)."""
        count = 0
        # Preferred path: iterate arbiters produced by the last space.step()
        if hasattr(self.space, "arbiters"):
            for arb in self.space.arbiters:
                a, b = arb.shapes
                bodies = (a.body, b.body)
                if (self.agent in bodies) and (self.block in bodies):
                    cps = arb.contact_point_set
                    count += len(getattr(cps, "points", []))
            return count

        # Fallback path: query collisions against the block’s shapes (older/stripped builds)
        try:
            for s in self.block.shapes:
                infos = self.space.shape_query(s)  # list of ShapeQueryInfo
                for qi in infos:
                    if getattr(qi, "shape", None) in getattr(self.agent, "shapes", []):
                        cps = getattr(qi, "contact_point_set", None)
                        if cps is not None:
                            count += len(getattr(cps, "points", []))
            return count
        except Exception:
            # If even this isn’t available, return 0 rather than crashing.
            return 0

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz

        if action is not None:
            self.latest_action = action
            for _ in range(n_steps):
                # PD control
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Physics step
                self.space.step(dt)

                # Count contacts that occurred during this sub-step
                self.n_contact_points += self._count_contacts_now()

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold
        terminated = done
        truncated = done

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here dosn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is aleady ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def seed(self, seed=None):
        # Gymnasium handles seeding in reset(), so this method might not be strictly necessary
        # depending on how it's used in the rest of the code.
        # For now, keeping it to maintain compatibility with potential external calls.
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatiblity with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2],
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # --- new, version-agnostic collision install ---
        #self._install_collision_handlers()

        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')
        shape.collision_type = TYPE_WALL          # <--- add this
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        shape.collision_type = TYPE_AGENT
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape1.collision_type = TYPE_BLOCK
        shape2.collision_type = TYPE_BLOCK
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body


# In[4]:


from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS
#@markdown ### **Env Demo**
#@markdown Standard Gym Env (0.21.0 API)

# 0. create env object
env = PushTEnv()

# 1. seed env for initial state.
# Seed 0-200 are used for the demonstration dataset.
seed = 1000
# env.seed(1000) # Seed is now passed to reset

# 2. must reset before use
obs, info = env.reset(seed=seed)

# 3. 2D positional action space [0,512]
action = env.action_space.sample()

# 4. Standard gym step method
obs, reward, terminated, truncated, info = env.step(action)

# prints and explains each dimension of the observation and action vectors
with np.printoptions(precision=4, suppress=True, threshold=5):
    print("Obs: ", repr(obs))
    print("Obs:        [agent_x,  agent_y,  block_x,  block_y,    block_angle]")
    print("Action: ", repr(action))
    print("Action:   [target_agent_x, target_agent_y]")


# In[5]:


#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTStateDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data (obs, action) from a zarr storage
#@markdown - Normalizes each dimension of obs and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `obs`: shape (obs_horizon, obs_dim)
#@markdown  - key `action`: shape (pred_horizon, action_dim)

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
        from zarr.storage import ZipStore

        # read from zarr dataset (zip in-place)
        store = ZipStore(dataset_path, mode="r")  # dataset_path is the .zip file
        dataset_root = zarr.open_group(store=store, mode="r")
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample


# In[6]:


#@markdown ### **Dataset Demo**

# download demonstration data from Google Drive
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

# create dataset from file
dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)


# In[16]:


# @markdown ### **Network**
# @markdown
# @markdown Defines a 1D UNet architecture `ConditionalUnet1D`
# @markdown as the noise prediction network
# @markdown
# @markdown Components
# @markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
# @markdown - `Downsample1d` Strided convolution to reduce temporal resolution
# @markdown - `Upsample1d` Transposed convolution to increase temporal resolution
# @markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
# @markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
# @markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
# @markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x


# In[62]:


# @title
# observation and action dimensions corrsponding to
# the output of PushTEnv
obs_dim = 5
action_dim = 2

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))

# the noise prediction network
# takes noisy action, diffusion iteration and observation as input
# predicts the noise added to action
noise = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    global_cond=obs.flatten(start_dim=1))

# illustration of removing noise
# the actual noise removal is performed by NoiseScheduler
# and is dependent on the diffusion noise schedule
denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations

# noise_scheduler = DDPMScheduler(
#     num_train_timesteps=num_diffusion_iters,
#     # the choise of beta schedule has big impact on performance
#     # we found squared cosine works the best
#     beta_schedule='squaredcos_cap_v2',
#     # clip output to [-1,1] to improve stability
#     clip_sample=True,
#     # our network predicts noise (instead of denoised action)
#     prediction_type='epsilon'
# )

num_diffusion_iters = 30
noise_scheduler = DDIMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

#@markdown ### **Training**
#@markdown
#@markdown Takes about an hour. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights

if not os.path.isfile('bc_policy.pth'):
    wandb.init(project="pusht-bc")

    num_epochs = 50

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # data normalized in dataset
                    # device transfer
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    # observation as FiLM conditioning
                    # (B, obs_horizon, obs_dim)
                    obs_cond = nobs[:,:obs_horizon,:]
                    # (B, obs_horizon * obs_dim)
                    obs_cond = obs_cond.flatten(start_dim=1)

                    # sample noise to add to actions
                    noise = torch.randn(naction.shape, device=device)

                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)

                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler every batch
                    # this is different from standard pytorch behavior
                    lr_scheduler.step()

                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],})

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net.parameters())

                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

    wandb.finish()

    torch.save(
        noise_pred_net.state_dict(),
        'bc_policy.pth'
    )

else: noise_pred_net.load_state_dict(torch.load('bc_policy.pth'))

# Weights of the EMA model
# is used for inference
# ema_noise_pred_net = noise_pred_net
# ema.copy_to(ema_noise_pred_net.parameters())

# In[9]:


#@markdown ### **PushTAdapter for Vectorization**
#@markdown Use envs = SyncVectorEnv([lambda: PushTAdapter() for _ in range(num_envs)])

# diffusion policy import
class PushTAdapter(gym.Env):
    """Adapts old PushTEnv to new Gymnasium API."""
    def __init__(self):
        self.env = PushTEnv() # Initialize your custom class
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space

        old_obs = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=old_obs.low,
            high=old_obs.high,
            shape=old_obs.shape,
            dtype=old_obs.dtype
        )

        # Fix Action Space
        old_act = self.env.action_space
        self.action_space = gym.spaces.Box(
            low=old_act.low,
            high=old_act.high,
            shape=old_act.shape,
            dtype=old_act.dtype
        )

        # Gymnasium requires metadata for render modes
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def reset(self, seed=None, options=None):
        # handle seeding for both old and new gym
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)

        raw_ret = self.env.reset()

        if isinstance(raw_ret, tuple):
            obs = raw_ret[0]
        else:
            obs = raw_ret

        # Force it to be a numpy array to satisfy the Box space
        obs = np.array(obs, dtype=self.observation_space.dtype)

        return obs, {}

    def step(self, action):
        # CALL STEP
        raw_ret = self.env.step(action)

        # FIX: Handle 4-value (old) or 5-value (new) returns
        if len(raw_ret) == 4:
            obs, reward, done, info = raw_ret
            truncated = False
        elif len(raw_ret) == 5:
            obs, reward, terminated, truncated, info = raw_ret
            done = terminated or truncated
        else:
            raise ValueError(f"Env returned {len(raw_ret)} values, expected 4 or 5.")

        # Force obs to numpy
        obs = np.array(obs, dtype=self.observation_space.dtype)

        # Return 5 values for Gymnasium
        return obs, reward, bool(done), bool(truncated), info

    def render(self):
        return self.env.render(mode='rgb_array')


# In[51]:


def gaussian_log_prob(x, mean, var):
    var = torch.clamp(var, min=1e-4)
    log_prob_per_dim = -0.5 * (
        ((x - mean) ** 2) / var
        + torch.log(2 * torch.pi * var)
    )
    return log_prob_per_dim.flatten(start_dim=1).mean(dim=1)


# In[ ]:


# @title
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

          b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
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
    action_dim=2       # based on PushT
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

num_pg_iters = 1000
batch_size =  256        # episodes per PG iteration
max_env_steps = 100
big_batch_size = 512   # per-update sample batch in update_model_efficiently
epochs = 1

num_envs = batch_size

env = SyncVectorEnv([lambda: PushTAdapter() for _ in range(num_envs)])
noise_pred_net.train()

wandb.init(project="diffusion_policy_ddpo")

optimizer = torch.optim.AdamW(
    noise_pred_net.parameters(),
    lr=1e-5, weight_decay=1e-6
)

total_steps = num_pg_iters * max_env_steps * num_diffusion_iters * epochs * batch_size // big_batch_size

warmup_steps = int(0.01 * total_steps)

# lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
lr_scheduler = LambdaLR(optimizer, lambda step: step / warmup_steps if step < warmup_steps else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

for it in range(num_pg_iters):
    if it % 10 == 0:
        max_score = evaluate_push_t(noise_pred_net, noise_scheduler,
                                    stats, device)

        wandb.log({"eval_max_score": max_score}, step=it)
    
    # ===== Phase 1: collect trajectories (no grad) =====
    time_gen_0 = time.time()
    trajectory_data, returns = collect_trajectories_flat(
        env=env,
        model=noise_pred_net,
        noise_scheduler=noise_scheduler,
        stats=stats,
        batch_size=batch_size,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
        action_horizon=action_horizon,
        num_diffusion_iters=num_diffusion_iters,
        device=device,
        max_env_steps=max_env_steps,
    )
    time_gen_1 = time.time()
    batch_collection_time = time_gen_1 - time_gen_0
    print(f'collected trajectories in {batch_collection_time:.2f} sec')

    # ===== Phase 2: big batched model update =====
    time_opt_0 = time.time()
    avg_loss = update_model_efficiently(
        model=noise_pred_net,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        trajectory_data=trajectory_data,
        returns=returns,
        device=device,
        batch_size=big_batch_size,
        epochs=epochs,
        clip_eps=0.15,
    )
    time_opt_1 = time.time()
    optimization_time = time_opt_1 - time_opt_0
    print(f'optimization in {optimization_time:.2f} sec')

    print(f"[DDPO] iter {it:04d} loss={avg_loss:.3f} "
          f"return_mean={returns.mean().item():.2f} "
          f"return_std={returns.std().item():.2f}")

    wandb.log({
        "ddpo_loss": avg_loss,
        "avg_return": returns.mean().item(),
        "return_std": returns.std().item(),
        "time/batch_collection_time_sec": batch_collection_time,
        "time/optimization_time_sec": optimization_time,
        "lr": optimizer.param_groups[0]['lr'],
    }, step=it)


# In[ ]:







