#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from tensordict import TensorDict
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

from scenarios.four_flags.load_config import load_scenario_config
from scenarios.four_flags.language import LanguageUnit, load_decoder, load_sequence_model
from scenarios.four_flags.language import FIND_GOAL, FIND_SWITCH, FIND_RED, FIND_GREEN, FIND_BLUE, FIND_PURPLE, STATES, NUM_AUTOMATA

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment

class FourFlagsScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)
        assert 1 <= self.n_passages <= 20

        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        
        self._prepare_obs_layout()
        self._add_passages(world)
        self._add_switch(world)
        self._add_flags(world)
        self._init_language_unit(world)
        self._add_agents_and_goals(world)

        return world
    
    def _add_passages(self, world):
        self.landmarks = []
        num_passages = int((2 * world.y_semidim + 2 * self.agent_radius + 1e-4) // self.passage_width)
        if self.break_all_wall:
            self.n_passages = num_passages
        for i in range(num_passages):
            passage = Landmark(
                name=f"passage {i}",
                collide=True,
                movable=False,
                shape=Box(length=self.passage_length, width=self.passage_width), # Rotated 90 degrees
                color=Color.RED,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            self.landmarks.append(passage)
            world.add_landmark(passage)

    def _add_switch(self, world):
        self.switch = Landmark(
            name=f"switch",
            collide=False,
            movable=False,
            shape=Sphere(radius=self.agent_radius * 3),
            color=Color.YELLOW,
            collision_filter=lambda e: not isinstance(e.shape, Box),
        )
        world.add_landmark(self.switch)
        
        self.x_room_max = torch.ones((world.batch_dim, 1), device=world.device, dtype=torch.float32) * (self.x_semidim - self.chamber_width - self.switch_radius - self.passage_length / 2)
        self.x_room_min = torch.ones((world.batch_dim, 1), device=world.device, dtype=torch.float32) * (-self.x_semidim + self.switch_radius)
        x_room_center = (self.x_room_max - self.x_room_min) / 2
        y_room_center = torch.zeros((world.batch_dim, 1), device=world.device, dtype=torch.float32)
        self.room_center =  torch.cat((x_room_center, y_room_center), dim=1)
        self.switch.state.pos = self.room_center.clone()
        
    def _add_flags(self, world):
        self.flags = []
        self.colors = [Color.RED, Color.GREEN, Color.BLUE, Color.PURPLE]
        for i,c in enumerate(self.colors):
            flag = Landmark(
                name=f"flag {i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=self.agent_radius * 5),
                color=c,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            self.flags.append(flag)
            world.add_landmark(flag)
            
    def _init_language_unit(self, world: World):
        #load_decoder(model_path=self.decoder_model_path, embedding_size=self.embedding_size, device=world.device)
        load_sequence_model(model_path=self.sequence_model_path, embedding_size=self.embedding_size, event_size=self.event_dim, state_size=self.state_dim, device=world.device)

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
        self.prev_team_found_flags = self.team_found_flags.clone()
        self.prev_team_hit_switch = self.team_hit_switch.clone()

        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                v_range=self.agent_v_range,
                f_range=self.agent_f_range,
                u_range=self.agent_u_range,
            )
            if self.use_velocity_controller:
                agent.controller = VelocityController(
                    agent, world, ctrl_params=(2.0, 6.0, 0.002), pid_form="standard"
                )
            
            agent.y = torch.zeros( 
                (world.batch_dim, self.embedding_size),
                dtype=torch.float32,
                device=world.device,
            )

            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
                color=Color.LIGHT_GREEN,
            )
            
            agent.goal = goal
            self.goals.append(goal)
            world.add_landmark(goal)

            agent.hit_switch = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)
            agent.found_flags = torch.zeros((world.batch_dim, 4), dtype=torch.bool, device=world.device)
            agent.on_goal = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)

            world.add_agent(agent)
            
    def _prepare_obs_layout(self):
        # compute once
        P = 0 if self.break_all_wall else self.n_passages
        # 2 coords per relative vector
        self._obs_dim = 2*(P + 1 + 1 + 4)  # passages + goal + switch + 4 flags
        # slice index helpers
        i = 0
        self._sl_passages = slice(i, i + 2*P); i += 2*P
        self._sl_goal     = slice(i, i + 2);   i += 2
        self._sl_switch   = slice(i, i + 2);   i += 2
        self._sl_flags    = slice(i, i + 8)

    
    def reset_landmarks(self, env_index: int = None):
        
        self.x_room_max = (self.x_semidim - self.chamber_width - self.agent_radius - self.passage_length / 2)
        self.x_room_min = (-self.x_semidim + self.agent_radius)
        x_room_center = (self.x_room_max + self.x_room_min) / 2
        y_room_center = 0
        self.room_center =  torch.tensor([x_room_center, y_room_center], dtype=torch.float32, device=self.world.device)
        if env_index is None:
            self.switch.state.pos = self.room_center.repeat(self.world.batch_dim, 1)
        else:
            self.switch.state.pos[env_index] = self.room_center.clone()
        
        # Randomize the positions of the flags
        rand_indices = torch.randperm(len(self.flags)).tolist()
        for i, j in enumerate(rand_indices):
            flag = self.flags[j]
            xx = self.x_room_min + self.agent_radius * 3 + (i % 2) * (self.x_room_max - self.x_room_min - 2 * (self.agent_radius * 3))
            yy = -self.y_semidim + self.agent_radius + self.agent_radius * 3 + (i // 2) * (2 * self.y_semidim - 2 * (self.agent_radius + self.agent_radius * 3))
            if env_index is None:
                flag.set_pos(
                    torch.tensor(
                        [xx, yy],
                        dtype=torch.float32,
                        device=self.world.device,
                    ).repeat(self.world.batch_dim, 1),
                    batch_index=env_index,
                )
            else:
                flag.set_pos(
                    torch.tensor(
                        [xx, yy],
                        dtype=torch.float32,
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )

    def reset_goals(self, env_index: int = None):
        
        central_goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (3 * self.agent_radius + self.agent_spacing)
                    + self.passage_width / 2 + self.x_semidim - self.chamber_width,
                    self.x_semidim - (3 * self.agent_radius + self.agent_spacing),
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.y_semidim + (3 * self.agent_radius + self.agent_spacing),
                    self.y_semidim - (3 * self.agent_radius + self.agent_spacing),
                ),

            ],
            dim=1,
        )

        order = torch.randperm(self.n_agents).tolist()
        goals = [self.goals[i] for i in order]
        for i, goal in enumerate(goals):
            if i == self.n_agents - 1:
                goal.set_pos(
                    central_goal_pos,
                    batch_index=env_index,
                )
            else:
                goal.set_pos(
                    central_goal_pos
                    + torch.tensor(
                        [
                            [
                                (
                                    0.0
                                    if i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 0
                                        else -self.agent_spacing
                                    )
                                ),
                                (
                                    0.0
                                    if not i % 2
                                    else (
                                        self.agent_spacing
                                        if i == 1
                                        else -self.agent_spacing
                                    )
                                ),
                            ],
                        ],
                        device=self.world.device,
                    ),
                    batch_index=env_index,
                )
                
    def reset_agent_pos(self, agent: Agent, env_index: int = None):

        # Goal coordinates are set to zero
        agent.hit_switch[env_index].zero_() if env_index is not None else agent.hit_switch.zero_()

        pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.x_semidim + self.agent_radius,
                    self.x_semidim - self.chamber_width - self.passage_length / 2 - self.agent_radius,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1 + self.agent_radius,
                    1 - self.agent_radius,
                ),
            ],
            dim=1,
        )

        agent.set_pos(pos, batch_index=env_index)
        if env_index is None:
            agent.global_shaping = (
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            agent.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )
    
    def reset_agents(self, env_index: int = None):
        
        for agent in self.world.agents:
            
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
            
    def reset_world_at(self, env_index: int = None):
        
        self.reset_landmarks(env_index)
        
        # Reset Environment Variables
        if env_index is None:
                self.language_unit.reset_all()
        else:
            self.language_unit.reset_env(env_index)

        self.language_unit.sample_sequence_dataset(env_index)
        self.team_hit_switch.zero_() if env_index is None else self.team_hit_switch[env_index].zero_()
        self.team_found_flags.zero_() if env_index is None else self.team_found_flags[env_index].zero_()
        self.prev_team_found_flags = self.team_found_flags.clone()
        self.prev_team_hit_switch = self.team_hit_switch.clone()
        self.init.zero_() if env_index is None else self.init[env_index].zero_()
        self.new_state.zero_() if env_index is None else self.new_state[env_index].zero_()
        
        # # Reset the automaton state
        # if env_index is None:
        #     self.automaton_state = [self.automaton._initial for _ in range(self.world.batch_dim)]
        #     self.language_unit.states = torch.tensor([STATES[self.automaton._initial] for _ in range(self.world.batch_dim)], device=self.world.device)
        # else:
        #     self.automaton_state[env_index] = self.automaton._initial
        #     self.language_unit.states[env_index] = STATES[self.automaton._initial]
        if env_index is None:
            self.y = self.language_unit.task_embeddings.clone()
        else:
            self.y[env_index] = self.language_unit.task_embeddings[env_index].clone()

        # Set the goal positions
        self.reset_goals(env_index)
        # Reset Agents
        self.reset_agents(env_index)

        order = torch.randperm(len(self.landmarks)).tolist()
        passages = [self.landmarks[i] for i in order]
        for i, passage in enumerate(passages):
            # if not passage.collide:
            #     passage.is_rendering[:] = False
            passage.set_pos(
                torch.tensor(
                    [
                        self.x_semidim - self.chamber_width,
                        -self.y_semidim + self.passage_width / 2 - self.agent_radius + self.passage_width * i,
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                ),
                batch_index=env_index,
            )
        self.landmark_poses = [landmark.state.pos.clone() for landmark in self.landmarks]
            
    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if self.shared_reward:
            if is_first:
                self.rew = torch.zeros(
                    self.world.batch_dim,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                for a in self.world.agents:
                    dist_to_goal = torch.linalg.vector_norm(
                        a.state.pos - a.goal.state.pos, dim=1
                    )
                    agent_shaping = dist_to_goal * self.shaping_factor
                    self.rew += a.global_shaping - agent_shaping
                    a.global_shaping = agent_shaping
        else:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            dist_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=1
            )
            agent_shaping = dist_to_goal * self.shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping

        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    self.rew[self.world.is_overlapping(a, agent)] -= 2
            for landmark in self.landmarks:
                if landmark.collide:
                    self.rew[self.world.is_overlapping(agent, landmark)] -= 0
            for i, flag in enumerate(self.flags):
                overlapping_flag = self.world.is_overlapping(agent, flag)
                agent.found_flags[:,i] |= overlapping_flag.bool()
                self.team_found_flags[:,i] |= agent.found_flags[:,i]
                self.rew[overlapping_flag] += 0.05
                
            overlapping_switch = self.world.is_overlapping(agent, self.switch)
            agent.hit_switch |= overlapping_switch.bool()
            self.team_hit_switch |= agent.hit_switch
            self.rew[overlapping_switch] += 0.05

            overlapping_goal = self.world.is_overlapping(agent, agent.goal)
            self.rew[overlapping_goal] += 0.05
            # only the *first-time* hits (overlap & never hit before)
        
        # A shared reward for being on the team state
        #self.rew += (self.language_unit.states == agent.state_).float() * 0.5

        # Despawn passage if switch is hit
        goal_indices = torch.where(self.language_unit.states == FIND_GOAL)[0]
        if goal_indices.any():
            passages = self.landmarks[: self.n_passages]
            for passage in passages:
                passage.state.pos[goal_indices] = self._get_outside_pos(None)[goal_indices]
                
        # --- Edge penalty: outside the safe box (+/- (semidim - agent_radius)) ---
        sx = self.world.x_semidim - self.agent_radius
        sy = self.world.y_semidim - self.agent_radius

        # agent.state.pos: [B, 2] -> x, y
        pos = agent.state.pos  # on world.device
        x, y = pos[:, 0], pos[:, 1]

        # Touching or beyond the edges on either axis (allow optional tolerance)
        eps = getattr(self, "edge_eps", 0.0)
        at_or_beyond_x = (x >= sx - eps) | (x <= -sx + eps)
        at_or_beyond_y = (y >= sy - eps) | (y <= -sy + eps)
        hit_edge = at_or_beyond_x | at_or_beyond_y  # [B]

        edge_penalty = getattr(self, "edge_penalty", 1.0)
        self.rew[hit_edge] -= edge_penalty

        return self.rew * 0.01

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

        obs_dict["sentence_embedding"] = agent.y.clone()
        obs_dict["obs"] = torch.cat(obs_components, dim=-1)
        obs_dict["task_state"] = self.language_unit.states.float()
        obs_dict["agent_state"] = agent.state_.float()

        return obs_dict
        
    def _get_outside_pos(self, env_index):
        """Get a position far outside the environment to hide entities."""
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)   
    
    def process_action(self, agent):
        
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

        for agent in self.world.agents:
            # agent.h: [E, H], agent.y: [E, Y] (batched per env)
            h = agent.h.clone()
            y = agent.y.clone()
            #e = agent.e.clone()
            e = torch.cat([self.team_found_flags, self.team_hit_switch.unsqueeze(1)], dim=-1).float()
            next_h, next_state = self.language_unit.compute_forward_rnn(
                event=e,
                y=y,
                h=h,
            )

            # Write back only for the changed envs
            agent.h = next_h
            agent.state_ = torch.argmax(torch.sigmoid(next_state[:,NUM_AUTOMATA:]), dim=-1)
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

        
        for i, agent1 in enumerate(self.world.agents):
            
            # Add rings for visited flags
            for j, _ in enumerate(self.flags):
                if agent1.found_flags[env_index, j]:
                    ring = rendering.make_circle(
                        radius=0.05 + j * 0.02,
                        filled=False,
                    )
                    ring.set_linewidth(2)
                    xform = rendering.Transform()
                    xform.set_translation(*agent1.state.pos[env_index])
                    ring.add_attr(xform)
                    ring.set_color(*self.colors[j].value)
                    geoms.append(ring)
                    
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                if self.world.get_distance(agent1, agent2)[env_index] <= self.comms_radius * (self.x_semidim + self.y_semidim)/2:
                    line = rendering.Line(
                        agent1.state.pos[env_index], agent2.state.pos[env_index], width=5
                    )
                    line.set_color(*Color.GREEN.value)
                    geoms.append(line)
        
        try:
            sentence = self.language_unit.response[env_index]
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
def generate_cfg(overrides: list[str] = None, config_path: str = "../conf", config_name: str = "conf", restore_path: str = None) -> DictConfig:
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
    cfg = cfg[experiment_name]  # Get the config for the specific experiment
    return cfg, seed

if __name__ == "__main__":
    scenario = FourFlagsScenario()
    render_interactively(
       scenario, control_two_agents=True, n_passages=1, shared_reward=False
    )