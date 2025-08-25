#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

from scenarios.four_flags.load_config import load_scenario_config
from scenarios.four_flags.language import LanguageUnit, load_decoder, load_sequence_model
from scenarios.four_flags.language import FIND_GOAL, FIND_SWITCH, FIND_RED, FIND_GREEN, FIND_BLUE, FIND_PURPLE

class FourFlagsScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)
        assert 1 <= self.n_passages <= 20
        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        
        self._prepare_obs_layout()
        self._prepare_state_map(device)
        
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
        self.switch.target_flag = torch.zeros(world.batch_dim, device=world.device, dtype=torch.int)
        self.switch._target_set = torch.zeros(world.batch_dim, dtype=torch.bool, device=world.device)
        
    def _add_flags(self, world):
        self.flags = []
        self.colors = [Color.RED, Color.GREEN, Color.BLUE, Color.PURPLE]
        self.flag_radius = 1.1 * (self.agent_spacing + self.agent_radius) / (3)**0.5
        for i,c in enumerate(self.colors):
            flag = Landmark(
                name=f"flag {i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=self.flag_radius),
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

            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
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
    
    def _prepare_obs_layout(self):
        # compute once
        P = 0 if self.break_all_wall else self.n_passages
        # 2 coords per relative vector
        self._obs_dim = 2*(P + 1 + 1 + 4) + self.event_dim # passages + goal + switch + 4 flags + events
        # slice index helpers
        i = 0
        self._sl_passages = slice(i, i + 2*P); i += 2*P
        self._sl_goal     = slice(i, i + 2);   i += 2
        self._sl_switch   = slice(i, i + 2);   i += 2
        self._sl_flags    = slice(i, i + 8);   i += 8
        self._sl_events   = slice(i, i + self.event_dim); i += self.event_dim
        assert i == self._obs_dim
        
    def _prepare_state_map(self, device):
        # map env state code -> index in targets tensor
        # indices: 0:GOAL, 1:SWITCH, 2:RED, 3:GREEN, 4:BLUE, 5:PURPLE
        max_code = int(max(FIND_GOAL, FIND_SWITCH, FIND_RED, FIND_GREEN, FIND_BLUE, FIND_PURPLE))
        self._state_map = torch.full((max_code+1,), -1, dtype=torch.long, device=device)
        self._state_map[FIND_GOAL]   = 0
        self._state_map[FIND_SWITCH] = 1
        self._state_map[FIND_RED]    = 2
        self._state_map[FIND_GREEN]  = 3
        self._state_map[FIND_BLUE]   = 4
        self._state_map[FIND_PURPLE] = 5
    
    def _set_initial_goal_coords(self, agent: Agent, world: World):
        agent.goal_coords = torch.zeros((world.batch_dim, 2), dtype=torch.float32, device=world.device)
        self.set_goal_coords(agent, world)

    @torch.no_grad()
    def set_goal_coords(self, agent: Agent, world: World, env_index: int = None):
        # targets: [B,6,2]
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
            xx = self.x_room_min + self.flag_radius + (i % 2) * (self.x_room_max - self.x_room_min - 2 * self.flag_radius)
            yy = -self.y_semidim + self.flag_radius + (i // 2) * (2 * self.y_semidim - 2 * self.flag_radius)
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

        # pick a random central point
        central_goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    self.agent_radius + self.passage_width / 2 + self.x_semidim - self.chamber_width,
                    self.x_semidim - self.agent_radius,
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
            # line along y axis
            offset = torch.tensor(
                [[0.0, (i - (self.n_agents - 1) / 2) * self.agent_spacing]],
                device=self.world.device,
            )
            goal.set_pos(
                central_goal_pos + offset,
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
    
    def reset_agents(self, env_index: int = None):
        
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
                agent.h = self.language_unit.subtask_embeddings.clone()
                agent.e.zero_()
            else:
                agent.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
                agent.h[env_index] = self.language_unit.subtask_embeddings[env_index].clone()
                agent.e[env_index].zero_()
            
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
        self.switch._target_set.zero_() if env_index is None else self.switch._target_set[env_index].zero_()
        self.switch.target_flag.zero_() if env_index is None else self.switch.target_flag[env_index].zero_()
        
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
        
    
    def switch_hit_logic(self, agent: Agent):
        
        overlapping_switch = self.world.is_overlapping(agent, self.switch)
        agent.hit_switch |= overlapping_switch.bool()
        self.team_hit_switch |= agent.hit_switch
        self.rew[overlapping_switch] += 0.05

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
                        a.state.pos - a.goal_coords, dim=1
                    )
                    agent_shaping = dist_to_goal * self.shaping_factor
                    self.rew += a.global_shaping - agent_shaping
                    a.global_shaping = agent_shaping
                    
                    if a.collide:
                        for landmark in self.landmarks:
                            if landmark.collide:
                                self.rew[self.world.is_overlapping(a, landmark)] -= self.collision_penalty
                        for i, flag in enumerate(self.flags):
                            overlapping_flag = self.world.is_overlapping(a, flag)
                            a.found_flags[:,i] |= overlapping_flag.bool()
                            self.team_found_flags[:,i] |= a.found_flags[:,i]
                            self.rew[overlapping_flag] += 0.05

                        self.switch_hit_logic(a)

                        overlapping_goal = self.world.is_overlapping(a, a.goal)
                        self.rew[overlapping_goal] += 0.05
        else:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )
            dist_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal_coords, dim=1
            )
            agent_shaping = dist_to_goal * self.shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping

            if agent.collide:
                for landmark in self.landmarks:
                    if landmark.collide:
                        self.rew[self.world.is_overlapping(agent, landmark)] -= 0
                for i, flag in enumerate(self.flags):
                    overlapping_flag = self.world.is_overlapping(agent, flag)
                    agent.found_flags[:,i] |= overlapping_flag.bool()
                    self.team_found_flags[:,i] |= agent.found_flags[:,i]
                    self.rew[overlapping_flag] += 0.05

                self.switch_hit_logic(agent)

            overlapping_goal = self.world.is_overlapping(agent, agent.goal)
            self.rew[overlapping_goal] += 0.05
            # only the *first-time* hits (overlap & never hit before)

        # Despawn passage if goal is hit
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
        
        # --- Soft minimum-distance penalty between agents (no need to overlap) ---
        min_sep = getattr(self, "min_separation", self.agent_spacing * 0.8)  # target min distance
        sep_weight = getattr(self, "separation_penalty", self.collision_penalty)        # strength

        # Per-agent penalty vs others (this agent only)
        pen = torch.zeros(self.world.batch_dim, device=self.world.device)
        for other in self.world.agents:
            if other is agent:
                continue
            d = torch.linalg.vector_norm(agent.state.pos - other.state.pos, dim=1)  # [B]
            short = (min_sep - d).clamp_min(0.0)
            pen += sep_weight * short * short
        self.rew -= pen

        return self.rew * 0.01

    def observation(self, agent: Agent):
        # pre-size once per call; the collector reuses tensors under the hood
        obs_buf = torch.empty((self.world.batch_dim, self._obs_dim),
                            device=self.world.device, dtype=torch.float32)

        if not self.break_all_wall:
            # Stack current passage poses to [B, P, 2] without Python loop
            # (landmark.state.pos is already [B,2])
            P = self.n_passages
            pass_pos = torch.stack([lm.state.pos for lm in self.landmarks[:P]], dim=1)  # [B,P,2]
            rel = pass_pos - agent.state.pos.unsqueeze(1)                                # [B,P,2]
            obs_buf[:, self._sl_passages] = rel.reshape(self.world.batch_dim, -1)

        obs_buf[:, self._sl_goal]   = agent.goal.state.pos - agent.state.pos
        obs_buf[:, self._sl_switch] = self.switch.state.pos - agent.state.pos
        obs_buf[:, self._sl_events] = torch.cat([agent.found_flags, agent.hit_switch.unsqueeze(1)], dim=-1).float()

        # flags: build once, no loop
        flags_pos = torch.stack([f.state.pos for f in self.flags], dim=1)               # [B,4,2]
        obs_buf[:, self._sl_flags] = (flags_pos - agent.state.pos.unsqueeze(1)).reshape(self.world.batch_dim, 8)

        return {
            "pos": agent.state.pos,
            "vel": agent.state.vel,
            "sentence_embedding": agent.h,
            "obs": obs_buf,
        }
        
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

    @torch.no_grad()
    def done(self):
        B = self.world.batch_dim
        agents = self.world.agents

        # 1) All agents on their goals
        on_goal = torch.all(
            torch.stack([
                torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=1) <= a.shape.radius / 2
                for a in agents
            ], dim=1),
            dim=1,
        )  # [B]

        # 2) All agents have found each flag color
        per_agent_flags = torch.stack([a.found_flags for a in agents], dim=1)  # [B,N,4]
        all_found_each_color = per_agent_flags.all(dim=1)                      # [B,4] (R,G,B,P)

        # 3) All agents have touched the switch
        per_agent_switch = torch.stack([a.hit_switch for a in agents], dim=1)  # [B,N]
        all_agents_hit_switch = per_agent_switch.all(dim=1)                    # [B]

        # 4) Build state-conditional termination table
        table = torch.stack(
            [
                on_goal,                    # FIND_GOAL
                all_agents_hit_switch,      # FIND_SWITCH (now requires everyone)
                all_found_each_color[:, 0], # FIND_RED
                all_found_each_color[:, 1], # FIND_GREEN
                all_found_each_color[:, 2], # FIND_BLUE
                all_found_each_color[:, 3], # FIND_PURPLE
            ],
            dim=1,
        )  # [B,6]

        idx = self._state_map[self.language_unit.states]  # [B]
        b = torch.arange(B, device=self.world.device)
        return table[b, idx]


    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering
        
        colors ={
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
                    
            # Communication lines
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                if self.world.get_distance(agent1, agent2)[env_index] <= self.comms_radius * (self.x_semidim + self.y_semidim)/2:
                    line = rendering.Line(
                        agent1.state.pos[env_index], agent2.state.pos[env_index], width=5
                    )
                    line.set_color(*state_color)
                    line.set_linewidth(2)
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

if __name__ == "__main__":
    scenario = FourFlagsScenario()
    render_interactively(
       scenario, control_two_agents=True, n_passages=1, shared_reward=False, display_info=False
    )