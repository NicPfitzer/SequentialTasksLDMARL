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
from scenarios.four_flags.language import LanguageUnit, load_decoder, load_sequence_model, load_multitask_data
from scenarios.four_flags.language import FIND_GOAL, FIND_SWITCH, FIND_RED, FIND_GREEN, FIND_BLUE, FIND_PURPLE

class FourFlagsScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)
        assert 1 <= self.n_passages <= 20

        world = World(batch_dim, device, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        
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
            shape=Sphere(radius=self.switch_radius),
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
                shape=Sphere(radius=self.agent_radius),
                color=c,
                collision_filter=lambda e: not isinstance(e.shape, Box),
            )
            self.flags.append(flag)
            world.add_landmark(flag)
            
    def _init_language_unit(self, world: World):
        if self.use_rnn:
            #load_decoder(model_path=self.decoder_model_path, embedding_size=self.embedding_size, device=world.device)
            load_sequence_model(model_path=self.sequence_model_path, embedding_size=self.embedding_size, event_size=self.event_dim, state_size=self.state_dim, device=world.device)
            load_multitask_data(json_path=self.data_json_path, device=world.device)

        self.language_unit = LanguageUnit(
            batch_size=world.batch_dim,
            embedding_size=self.embedding_size,
            use_embedding_ratio=self.use_embedding_ratio,
            device=world.device,
        )
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

            world.add_agent(agent)
    
    def _set_initial_goal_coords(self, agent: Agent, world: World):
        agent.goal_coords = torch.zeros((world.batch_dim, 2), dtype=torch.float32, device=world.device)
        self.set_goal_coords(agent, world)

    def set_goal_coords(self, agent: Agent, world: World, env_index: int = None):
        
        states = self.language_unit.states  # shape: (B,)
        self.state_to_pos = {
            FIND_SWITCH: self.switch.state.pos,
            FIND_GOAL:   agent.goal.state.pos,
            FIND_RED:    self.flags[FIND_RED-2].state.pos,
            FIND_GREEN:  self.flags[FIND_GREEN-2].state.pos,
            FIND_BLUE:   self.flags[FIND_BLUE-2].state.pos,
            FIND_PURPLE: self.flags[FIND_PURPLE-2].state.pos,
        }
        
        if env_index is None:

            for s, pos in self.state_to_pos.items():
                mask = (states == s)
                if mask.any():
                    agent.goal_coords[mask] = pos[mask]
        else:
            for s, pos in self.state_to_pos.items():
                if states[env_index] == s:
                    agent.goal_coords[env_index] = pos[env_index]
    
    def reset_landmarks(self, env_index: int = None):
        
        self.x_room_max = torch.ones((self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) * (self.x_semidim - self.chamber_width - self.switch_radius - self.passage_length / 2)
        self.x_room_min = torch.ones((self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32) * (-self.x_semidim + self.switch_radius)
        x_room_center = (self.x_room_max + self.x_room_min) / 2
        y_room_center = torch.zeros((self.world.batch_dim, 1), device=self.world.device, dtype=torch.float32)
        self.room_center =  torch.cat((x_room_center, y_room_center), dim=1)
        self.switch.state.pos = self.room_center.clone()
        
        # Randomize the positions of the flags
        rand_indices = torch.randperm(len(self.flags)).tolist()
        for i, j in enumerate(rand_indices):
            flag = self.flags[j]
            flag.set_pos(
                torch.tensor(
                    [
                        self.x_room_min + self.switch_radius + (i % 2) * (self.x_room_max - self.x_room_min - 2 * (self.switch_radius)),
                        -self.y_semidim + self.agent_radius + self.switch_radius + (i // 2) * (2 * self.y_semidim - 2 * (self.agent_radius + self.switch_radius)),
                    ],
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
            
            if env_index is None:
                agent.y = self.language_unit.task_embeddings.clone()
                agent.h = self.language_unit.subtask_embeddings.clone()
                agent.e.zero_()
            else:
                agent.y[env_index] = self.language_unit.task_embeddings[env_index]
                agent.h[env_index] = self.language_unit.subtask_embeddings[env_index]
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

        self.language_unit.sample_multitask_dataset(env_index)
        self.team_hit_switch.zero_() if env_index is None else self.team_hit_switch[env_index].zero_()
        self.team_found_flags.zero_() if env_index is None else self.team_found_flags[env_index].zero_()
        
        # Initialize Goal Coordinates
        if env_index is None:
                    hit_switch_indices = torch.where(self.language_unit.states == FIND_GOAL)[0]
                    self.team_hit_switch[hit_switch_indices] = True

        else:
            self.team_hit_switch[env_index] = self.language_unit.states[env_index] == FIND_GOAL
        
        # Reset Agents
        self.reset_agents(env_index)

        # Set the goal positions
        self.reset_goals(env_index)

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
                        a.state.pos - a.goal_coords, dim=1
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
                agent.state.pos - agent.goal_coords, dim=1
            )
            agent_shaping = dist_to_goal * self.shaping_factor
            self.rew += agent.global_shaping - agent_shaping
            agent.global_shaping = agent_shaping

        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    self.rew[self.world.is_overlapping(a, agent)] -= 0
            for landmark in self.landmarks:
                if landmark.collide:
                    self.rew[self.world.is_overlapping(agent, landmark)] -= 0
            for i, flag in enumerate(self.flags):
                agent.found_flags[:,i] |= self.world.is_overlapping(agent, flag).bool()
                self.team_found_flags[:,i] |= agent.found_flags[:,i]
                
            agent.hit_switch |= self.world.is_overlapping(agent, self.switch).bool()
            self.team_hit_switch |= agent.hit_switch
            # only the *first-time* hits (overlap & never hit before)
        
        # Despawn passage if switch is hit
        if self.team_hit_switch.any():
            indices = torch.where(self.team_hit_switch & torch.all(self.team_found_flags, dim=1))[0]
            passages = self.landmarks[: self.n_passages]
            for passage in passages:
                passage.state.pos[indices] = self._get_outside_pos(None)[indices]

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
        obs_dict["event"] = agent.e
        obs_components.append(agent.goal.state.pos - agent.state.pos)
        obs_components.append(self.switch.state.pos - agent.state.pos)
        if not self.use_rnn:
            obs_dict["sentence_embedding"] = torch.empty((self.world.batch_dim, 0), device=self.world.device)
            obs_components.append(agent.e)  # Convert bool to float for observation
        else:
            obs_dict["sentence_embedding"] = agent.h
        obs_dict["obs"] = torch.cat(obs_components, dim=-1)

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

    def done(self):
        
        dones = torch.zeros(
            self.world.batch_dim,
            dtype=torch.bool,
            device=self.world.device,
        )
        
        on_goal = torch.all(
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
        
        states = self.language_unit.states
        state_to_dones = {
            FIND_GOAL: on_goal,
            FIND_SWITCH: self.team_hit_switch,
            FIND_RED: self.team_found_flags[:, 0],
            FIND_GREEN: self.team_found_flags[:, 1],
            FIND_BLUE: self.team_found_flags[:, 2],
            FIND_PURPLE: self.team_found_flags[:, 3],
        }
        for s, pos in self.state_to_pos.items():
            mask = (states == s)
            if mask.any():
                dones[mask] = state_to_dones[s][mask]
        return dones

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        geoms = []
        for i in range(4):
            geom = Line(length=2 + self.agent_radius * 2).get_geometry()
            xform = rendering.Transform()
            geom.add_attr(xform)

            xform.set_translation(
                (
                    0.0
                    if i % 2
                    else (
                        self.world.x_semidim + self.agent_radius
                        if i == 0
                        else -self.world.x_semidim - self.agent_radius
                    )
                ),
                (
                    0.0
                    if not i % 2
                    else (
                        self.world.x_semidim + self.agent_radius
                        if i == 1
                        else -self.world.x_semidim - self.agent_radius
                    )
                ),
            )
            xform.set_rotation(torch.pi / 2 if not i % 2 else 0.0)
            color = Color.BLACK.value
            if isinstance(color, torch.Tensor) and len(color.shape) > 1:
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


if __name__ == "__main__":
    scenario = FourFlagsScenario()
    render_interactively(
       scenario, control_two_agents=True, n_passages=1, shared_reward=False
    )