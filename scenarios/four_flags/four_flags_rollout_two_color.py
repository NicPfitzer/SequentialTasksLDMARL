#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from tensordict import TensorDict
from collections import deque
from typing import Dict
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController
from wandb import agent

from scenarios.four_flags.load_config import load_scenario_config
from scenarios.four_flags.language import LanguageUnit, load_decoder, load_sequence_model
from scenarios.four_flags.language import FIND_GOAL, FIND_SWITCH, FIND_RED, FIND_GREEN, FIND_BLUE, FIND_PURPLE, STATES, NUM_AUTOMATA

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment  

from scenarios.four_flags.four_flags_scenario import FourFlagsScenario as BaseFourFlagsScenario


task_color_map = {
    0: (FIND_RED, FIND_GREEN),
    1: (FIND_RED, FIND_BLUE),
    2: (FIND_RED, FIND_PURPLE),

    3: (FIND_GREEN, FIND_RED),
    4: (FIND_GREEN, FIND_BLUE),
    5: (FIND_GREEN, FIND_PURPLE),

    6: (FIND_BLUE, FIND_RED),
    7: (FIND_BLUE, FIND_GREEN),
    8: (FIND_BLUE, FIND_PURPLE),

    9: (FIND_PURPLE, FIND_RED),
    10: (FIND_PURPLE, FIND_GREEN),
    11: (FIND_PURPLE, FIND_BLUE),
}

class FourFlagsScenario(BaseFourFlagsScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)
        assert 1 <= self.n_passages <= 20

        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        
        self._prepare_obs_layout()
        self._prepare_state_map(device)
        self._prepare_task_state(world)
        
        self._add_passages(world)
        self._add_switch(world)
        self._add_flags(world)
        self._init_language_unit(world)
        self._add_agents_and_goals(world)
        
        self._init_trails(world)

        return world
    
    def _init_trails(self, world: World):
        # Tunables
        self.trail_max_len = getattr(self, "trail_max_len", 500)   # number of stored points
        self.trail_stride = getattr(self, "trail_stride", 1)       # sample every N rendered frames
        self.trail_width = getattr(self, "trail_width", 10)         # line width
        self._trail_tick = 0

        # trails[agent_idx][env_idx] -> {"pos": deque[[2]], "state": deque[int]}
        self.trails = [
            [
                {"pos": deque(maxlen=self.trail_max_len),
                 "state": deque(maxlen=self.trail_max_len)}
                for _ in range(world.batch_dim)
            ]
            for _ in range(self.n_agents)
        ]

    def _clear_trails(self, env_index: int | None = None):
        if not hasattr(self, "trails"):
            return
        if env_index is None:
            for i in range(self.n_agents):
                for b in range(self.world.batch_dim):
                    self.trails[i][b]["pos"].clear()
                    self.trails[i][b]["state"].clear()
        else:
            for i in range(self.n_agents):
                self.trails[i][env_index]["pos"].clear()
                self.trails[i][env_index]["state"].clear()

    def _record_trail(self, env_index: int):
        # Called from extra_render so it records only when rendering
        self._trail_tick += 1
        if self._trail_tick % self.trail_stride != 0:
            return
        for i, a in enumerate(self.world.agents):
            # store current position and the discrete state_ driving color
            pos = a.state.pos[env_index].detach().clone()
            st = int(a.state_[env_index].item())
            self.trails[i][env_index]["pos"].append(pos)
            self.trails[i][env_index]["state"].append(st)
    
    def _add_switch(self, world):
        self.switch_radius = 1.1 * (self.agent_spacing + self.agent_radius) / (3)**0.5 # slightly larger than inscribed circle of tetrahedron
        self.switch = Landmark(
            name=f"switch",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.switch_radius),
            color=Color.YELLOW,
            collision_filter=lambda e: not isinstance(e.shape, Box),
        )
        world.add_landmark(self.switch)
        self.switch.target_flag = torch.zeros((world.batch_dim, 2), device=world.device, dtype=torch.int)
        self.switch._target_set = torch.zeros(world.batch_dim, dtype=torch.bool, device=world.device)
        
        num_classes = max(task_color_map.keys()) + 1
        lookup = torch.tensor(
            [task_color_map[i] for i in range(num_classes)],
            device=world.device,
            dtype=torch.long,   # or match self.switch.target_flag.dtype
        )
        self.switch._color_lookup = lookup
            
    def _init_language_unit(self, world: World):
        #load_decoder(model_path=self.decoder_model_path, embedding_size=self.embedding_size, device=world.device)
        load_sequence_model(model_path=self.sequence_model_path, embedding_size=self.embedding_size, event_size=self.event_dim, state_size=self.state_dim, device=world.device)
        
        # Load Action Policy
        cfg, seed = generate_cfg(config_path=self.policy_config_path, config_name=self.policy_config_name, restore_path=self.policy_restore_path, device=world.device.type)
        self.policy = get_policy_from_cfg(cfg, seed)

        self.language_unit = LanguageUnit(
            batch_size=world.batch_dim,
            embedding_size=self.embedding_size,
            use_embedding_ratio=self.use_embedding_ratio,
            device=world.device,
        )
        self.language_unit.load_sequence_data(json_path=self.data_json_path, device=world.device)
        self.visualize_semidims = False

    def _add_agents_and_goals(self, world: World):
        
        self.goals = []
        self.team_hit_switch = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)
        self.team_found_flags = torch.zeros((world.batch_dim, 4), dtype=torch.bool, device=world.device)

        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                v_range=self.agent_v_range,
                f_range=self.agent_f_range,
                u_range=self.agent_u_range,
                discrete_action_nvec= [2 for _ in range(self.event_dim)]
                #discrete_action_nvec=[self.event_dim]
            )
            if self.use_velocity_controller:
                agent.controller = VelocityController(
                    agent, world, ctrl_params=(2.0, 6.0, 0.002), pid_form="standard"
                )
            
            agent.h = torch.zeros(
                (world.batch_dim, self.embedding_size),
                dtype=torch.float32,
                device=world.device,
            )
            
            agent.y = torch.zeros( 
                (world.batch_dim, self.embedding_size),
                dtype=torch.float32,
                device=world.device,
            )
            
            agent.e = torch.zeros(
                (world.batch_dim, self.event_dim),
                dtype=torch.float32,
                device=world.device,
            )
            
            agent.state_ = torch.zeros(
                (world.batch_dim,),
                dtype=torch.float32,
                device=world.device,
            )

            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius * 2.),
                color=Color.LIGHT_GREEN,
            )
            agent.goal = goal
            self.goals.append(goal)
            world.add_landmark(goal)

            self._set_initial_goal_coords(agent, world)
            agent.hit_switch = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)
            agent.found_flags = torch.zeros((world.batch_dim, 4), dtype=torch.bool, device=world.device)
            agent.on_goal = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)

            world.add_agent(agent)
    
    def _prepare_task_state(self, world: World):
        # task state is the automaton state
        self.y = torch.zeros(
            (world.batch_dim, self.embedding_size), dtype=torch.float32, device=world.device
        )
        self.h = torch.zeros(
            (world.batch_dim, self.embedding_size), dtype=torch.float32, device=world.device
        )
    
    def next_task_state(self, world: World, env_index: int = None):
        
        # Build event tensor: [E, F+1]
        e_all = torch.cat(
            [self.team_found_flags, self.team_hit_switch.unsqueeze(1)], dim=-1
        ).float()

        E = e_all.shape[0]
        device = world.device

        # Select the working set of env indices
        if env_index is None:
            idx = torch.arange(E, device=device)
        else:
            idx = torch.as_tensor(env_index, device=device)

        if idx.numel() == 0:
            return

        # We update ALL selected envs (no previous-state condition)
        upd_idx = idx

        # Run the RNN update only on selected envs
        y = self.y[upd_idx]
        h = self.h[upd_idx]
        e = e_all[upd_idx]

        next_h, state_one_hot = self.language_unit.compute_forward_rnn(event=e, y=y, h=h)

        self.h[upd_idx] = next_h
        state_one_hot = state_one_hot.unsqueeze(0) if state_one_hot.dim() == 1 else state_one_hot
        
        # compute class index for the selected rows (local to upd_idx order)
        bits = (torch.sigmoid(state_one_hot[:, :NUM_AUTOMATA]) > 0.5)
        weights = (2 ** torch.arange(NUM_AUTOMATA - 1, -1, -1, device=world.device)).to(torch.int)
        class_idx_local = (bits * weights).sum(dim=1).to(torch.int)  # shape = [len(upd_idx)]

        # only write rows we haven't set yet
        not_set_local = ~self.switch._target_set[upd_idx]
        if torch.any(not_set_local):
            rows_to_set = upd_idx[not_set_local]                  # absolute row ids in [0..B)
            class_idx_to_set = class_idx_local[not_set_local]     # aligned with rows_to_set
            
            targets = self.switch._color_lookup[class_idx_to_set]
            targets = targets.to(self.switch.target_flag.dtype)
            self.switch.target_flag[rows_to_set] = targets
            self.switch._target_set[rows_to_set] = True
            
        states = torch.argmax(torch.sigmoid(state_one_hot[:, NUM_AUTOMATA:]), dim=-1)
        self.language_unit.states[upd_idx] = states

    @torch.no_grad()
    def set_goal_coords(self, agent: Agent, world: World, env_index: int = None):
        # targets: [B,6,2]
        self.next_task_state(world, env_index)
        targets = torch.stack(
            [
                agent.goal.state.pos,         # 0
                self.switch.state.pos,        # 1
                self.flags[0].state.pos,      # 2
                self.flags[1].state.pos,      # 3
                self.flags[2].state.pos,      # 4
                self.flags[3].state.pos,      # 5
            ],
            dim=1,
        )
        if env_index is None:
            idx = self._state_map[self.language_unit.states]          # [B]
            b = torch.arange(world.batch_dim, device=world.device)
            agent.goal_coords = targets[b, idx]                       # [B,2]
        else:
            idx = int(self._state_map[int(self.language_unit.states[env_index])])
            agent.goal_coords[env_index] = targets[env_index, idx]
    
    def reset_agents(self, env_index: int = None):
        
        if env_index is None:
            self.language_unit.states.zero_()
            self.y = self.language_unit.task_embeddings.clone()
            self.h.zero_()
        else:
            self.language_unit.states[env_index].zero_()
            self.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
            self.h[env_index].zero_()
        
        for agent in self.world.agents:

            self.set_goal_coords(agent, self.world, env_index)
            # Reset agent variables
            if env_index is None:
                agent.global_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal_coords, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                agent.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal_coords[env_index]
                    )
                    * self.shaping_factor
                )
            agent.found_flags.zero_() if env_index is None else agent.found_flags[env_index].zero_()
            agent.hit_switch.zero_() if env_index is None else agent.hit_switch[env_index].zero_()
            agent.on_goal.zero_() if env_index is None else agent.on_goal[env_index].zero_()
            
            if env_index is None:
                agent.y = self.language_unit.task_embeddings.clone()
                agent.h.zero_()
                agent.e.zero_()
                agent.state_.zero_()
            else:
                agent.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
                agent.h[env_index].zero_()
                agent.e[env_index].zero_()
                agent.state_[env_index].zero_()
            # Set the agent positions
            self.reset_agent_pos(agent,env_index)
            self._clear_trails(env_index)
            self._step_counter = 0
    def switch_hit_logic(self, agent: Agent):
        
        overlapping_switch = self.world.is_overlapping(agent, self.switch)
        batch_idx = torch.arange(self.world.batch_dim).unsqueeze(1)  # (batch_dim, 1)
        flags = agent.found_flags[batch_idx, self.switch.target_flag]  # (batch_dim, 2)

        flag_on_switch = overlapping_switch.bool() & (flags == 1).any(dim=1)
        agent.hit_switch |= flag_on_switch
        self.team_hit_switch |= agent.hit_switch
        self.rew[overlapping_switch & (self.language_unit.states == FIND_SWITCH)] += 0.05

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        obs_dict = {}
        obs_components = []
        
        if not self.break_all_wall:
            passage_obs = []
            for pose in self.landmark_poses[: self.n_passages]:
                passage_obs.append(pose - agent.state.pos)
            passage_obs = torch.stack(passage_obs, dim=1).view(self.world.batch_dim, -1)
            obs_components.append(passage_obs)
 
        obs_dict["pos"] = agent.state.pos
        obs_dict["vel"] = agent.state.vel
        obs_dict["event"] = torch.cat([agent.found_flags, agent.hit_switch.unsqueeze(1)], dim=-1).float()
        obs_components.append(agent.goal.state.pos - agent.state.pos)
        obs_components.append(self.switch.state.pos - agent.state.pos)
        
        for flag in self.flags:
            obs_components.append(flag.state.pos - agent.state.pos)
        
        obs_components.append(torch.cat([agent.found_flags, agent.hit_switch.unsqueeze(1)], dim=-1).float())

        obs_dict["sentence_embedding"] = agent.h.clone()
        obs_dict["obs"] = torch.cat(obs_components, dim=-1)
        obs_dict["task_state"] = self.language_unit.states.float()
        obs_dict["agent_state"] = agent.state_.float()

        return obs_dict
    
    def process_action(self, agent):
        
        is_first = agent == self.world.agents[0]
        if is_first:
            # Replace agent action with the multitask policy action
            obs_all_agents = self._collect_observations()
            actions = self.policy(obs_all_agents)
            for i, a in enumerate(self.world.agents):
                # Clip e to 0, 1 integers
                raw_event = a.action.u[:, :self.event_dim]
                a.e = torch.where(
                    raw_event > 0.,
                    torch.ones_like(raw_event, dtype=torch.float32),
                    torch.zeros_like(raw_event, dtype=torch.float32),
                )
                
                # raw_event = a.action.u[:,0]
                # idx = ((raw_event + 1.1) / (2.2) * self.event_dim).long()
                # idx = idx.clamp(0, self.event_dim - 1)
                # a.e = torch.nn.functional.one_hot(idx, num_classes=self.event_dim).float()
                
                a.action.u = actions.get(("agents", "action"))[:,i,:]

        # Clamp square to circle
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)

        # Zero small input
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.005] = 0

        if self.use_velocity_controller and not self.use_kinematic_model:
            agent.controller.process_force()
            
    def _collect_observations(self):
        data_dict = {} 

        for agent in self.world.agents:
            
            obs = self.observation(agent)
            for key, value in obs.items():
                data_dict.setdefault(key, []).append(value.unsqueeze(1).float())

        obs_dict = {
            ("agents", "observation", key): torch.cat(tensor_list, dim=1)
            for key, tensor_list in data_dict.items()
        }

        return TensorDict(obs_dict)
    
    def pre_step(self):
        # keep a step counter on the world or on this class
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0

        self._step_counter += 1

        # only update every 5th step
        if self._step_counter % 5 != 0 and self._step_counter > 1:
            return

        for agent in self.world.agents:
            # agent.h: [E, H], agent.y: [E, Y] (batched per env)
            h = agent.h.clone()
            y = agent.y.clone()

            # event vector
            e = torch.cat([self.team_found_flags,
                        self.team_hit_switch.unsqueeze(1)], dim=-1).float()

            next_h, next_state = self.language_unit.compute_forward_rnn(
                event=e,
                y=y,
                h=h,
            )

            # Write back only for the changed envs
            agent.h = next_h
            agent.state_ = torch.argmax(
                torch.sigmoid(next_state[:, NUM_AUTOMATA:]), dim=-1
            )
            self.set_goal_coords(agent, self.world)


    def done(self):

        return torch.all(
            torch.stack(
                [
                    torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=1)
                    <= a.shape.radius / 2
                    for a in self.world.agents
                ],
                dim=1,
            ),
            dim=1,
        )


    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        
        self._record_trail(env_index)

        colors = {
            FIND_RED: Color.RED,
            FIND_GREEN: Color.GREEN,
            FIND_BLUE: Color.BLUE,
            FIND_PURPLE: Color.PURPLE,
            FIND_SWITCH: Color.YELLOW,
            FIND_GOAL: Color.LIGHT_GREEN,
        }
        
        geoms = []
        # 0: right wall, 1: top wall, 2: left wall, 3: bottom wall
        for i in range(4):
            is_vertical = (i % 2 == 0)        # right/left walls are vertical
            # Length along the wall’s axis:
            length = (
                2 * self.world.y_semidim + 2 * self.agent_radius if is_vertical
                else 2 * self.world.x_semidim
            )

            geom = Line(length=length).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            # Center position of each wall
            if i == 0:   # right wall
                cx =  self.world.x_semidim
                cy = 0.0
            elif i == 1: # top wall
                cx = 0.0
                cy =  self.world.y_semidim + self.agent_radius
            elif i == 2: # left wall
                cx = -self.world.x_semidim 
                cy = 0.0
            else:        # bottom wall
                cx = 0.0
                cy = -self.world.y_semidim - self.agent_radius

            xform.set_translation(cx, cy)
            xform.set_rotation(float(torch.pi / 2) if is_vertical else 0.0)

            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and color.ndim > 1:
                # make env_index a plain int
                if isinstance(env_index, torch.Tensor):
                    env_index = int(env_index.item())
                color = color[env_index]
            geom.set_color(*color)

            geoms.append(geom)

        state = self.language_unit.states[env_index].item()
        state_color = colors.get(state, Color.GRAY).value
        
        # # Add ring on switch to indicate target flag
        # target_states = self.switch.target_flag[env_index]
        # if self.switch._target_set[env_index] and 0 <= target_state < len(self.state_color_map):
        #     ring = rendering.make_circle(radius=self.switch.shape.radius + 0.05, filled=False)
        #     ring.set_linewidth(10)
        #     xform = rendering.Transform()
        #     xform.set_translation(*self.switch.state.pos[env_index])
        #     ring.add_attr(xform)
        #     ring.set_color(*self.state_color_map[target_state].value)
        #     geoms.append(ring)


        for i, agent1 in enumerate(self.world.agents):

            # Agent state
            agent_color = colors.get(agent1.state_[env_index].item(), Color.GRAY).value
            circle = rendering.make_circle(radius=self.agent_radius / 3, filled=True)
            xform = rendering.Transform()
            xform.set_translation(*agent1.state.pos[env_index])
            circle.add_attr(xform)
            circle.set_color(*agent_color)
            geoms.append(circle)
            
            tr = self.trails[i][env_index]
                
            for t in range(1, len(tr["pos"])):
                p0 = tr["pos"][t - 1]
                p1 = tr["pos"][t]
                st = tr["state"][t]
                color_rgb = self.state_color_map[st].value  # e.g., (r, g, b)

                seg = rendering.Line(p0, p1, width=self.trail_width)
                seg.set_color(*color_rgb)
                seg.set_linewidth(self.trail_width)
                geoms.append(seg)
            
            # # Add rings for visited flags
            # for j, state in self.flag_state_map.items():
            #     if agent1.found_flags[env_index, j]:
            #         ring = rendering.make_circle(
            #             radius=0.05 + j * 0.02,
            #             filled=False,
            #         )
            #         ring.set_linewidth(5)
            #         xform = rendering.Transform()
            #         xform.set_translation(*agent1.state.pos[env_index])
            #         ring.add_attr(xform)
            #         ring.set_color(*self.state_color_map[state].value)
            #         geoms.append(ring)
            
            # # Add a ring for switch hit
            # if agent1.hit_switch[env_index]:
            #     ring = rendering.make_circle(
            #         radius=0.05 + len(self.flags) * 0.02,
            #         filled=False,
            #     )
            #     ring.set_linewidth(3)
            #     xform = rendering.Transform()
            #     xform.set_translation(*agent1.state.pos[env_index])
            #     ring.add_attr(xform)
            #     ring.set_color(*Color.YELLOW.value)
            #     geoms.append(ring)
                
            # Communication lines
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                if self.world.get_distance(agent1, agent2)[env_index] <= self.comms_radius * (self.x_semidim + self.y_semidim)/2:
                    line = rendering.Line(
                        agent1.state.pos[env_index], agent2.state.pos[env_index], width=5
                    )
                    line.set_color(*Color.GRAY.value)
                    line.set_linewidth(2)
                    geoms.append(line)
        
        try:
            sentence = self.language_unit.summary[env_index]
            geom = rendering.TextLine(
                text=sentence,
                font_size=6
            )
            geom.label.color = (0, 0, 0, 255)
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
        except:
            print("No sentence found for this environment index, or syntax is wrong.")
            pass
            
        return geoms
    
import copy
def get_policy_from_cfg(cfg: DictConfig, seed: int):
    experiment = benchmarl_setup_experiment(cfg, seed=seed, main_experiment=False)
    policy_copy = copy.deepcopy(experiment.policy)  # full independent copy
    del experiment  # drop the rest
    return policy_copy

import os
def generate_cfg(overrides: list[str] = None, config_path: str = "../conf", config_name: str = "conf", restore_path: str = None, device: str = "cpu") -> DictConfig:
    overrides = overrides or []  # e.g., ["restore_path=some_path"]
    # current working directory
    print(f"Current working directory: {os.getcwd()}")
    # Add directory to restore_path if it is not None
    restore_path = os.path.join(os.getcwd(), restore_path) if restore_path else None
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=overrides)
    experiment_name = list(cfg.keys())[0]
    seed = cfg.seed
    cfg[experiment_name].experiment.restore_file = restore_path
    cfg[experiment_name].experiment.restore_map_location = device
    cfg = cfg[experiment_name]  # Get the config for the specific experiment
    return cfg, seed

if __name__ == "__main__":
    scenario = FourFlagsScenario()
    render_interactively(
       scenario, control_two_agents=True, n_passages=1, shared_reward=False, display_info=False
    )