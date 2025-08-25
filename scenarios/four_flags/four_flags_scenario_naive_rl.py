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

from scenarios.four_flags.four_flags_scenario import FourFlagsScenario as BaseFourFlagsScenario

class FourFlagsScenario(BaseFourFlagsScenario):
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
             
    def reset_agents(self, env_index: int = None):
        
        for agent in self.world.agents:
            
            agent.found_flags.zero_() if env_index is None else agent.found_flags[env_index].zero_()
            agent.hit_switch.zero_() if env_index is None else agent.hit_switch[env_index].zero_()
            agent.on_goal.zero_() if env_index is None else agent.on_goal[env_index].zero_()
            
            if env_index is None:
                agent.y = self.language_unit.task_embeddings.clone()
            else:
                agent.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
            # Set the agent positions
            self.reset_agent_pos(agent,env_index)
            
            
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
                agent.state.pos - agent.goal.state.pos, dim=1
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

if __name__ == "__main__":
    scenario = FourFlagsScenario()
    render_interactively(
       scenario, control_two_agents=True, n_passages=1, shared_reward=False, display_info=False
    )