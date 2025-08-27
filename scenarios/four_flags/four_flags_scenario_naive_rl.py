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
        self._prepare_state_map(device)
        self._add_passages(world)
        self._add_switch(world)
        self._add_flags(world)
        self._init_language_unit(world)
        self._add_agents_and_goals(world)

        return world
             
    def reset_agents(self, env_index: int = None):
        
        self.language_unit.states.fill_(FIND_GOAL)
        super().reset_agents(env_index)
        
        # for agent in self.world.agents:
            
        #     agent.found_flags.zero_() if env_index is None else agent.found_flags[env_index].zero_()
        #     agent.hit_switch.zero_() if env_index is None else agent.hit_switch[env_index].zero_()
        #     agent.on_goal.zero_() if env_index is None else agent.on_goal[env_index].zero_()
            
        #     if env_index is None:
        #         agent.y = self.language_unit.task_embeddings.clone()
        #     else:
        #         agent.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
        #     # Set the agent positions
        #     self.reset_agent_pos(agent,env_index)
    
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
            "sentence_embedding": agent.y,
            "obs": obs_buf,
        }
            
            
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

if __name__ == "__main__":
    scenario = FourFlagsScenario()
    render_interactively(
       scenario, control_two_agents=True, n_passages=1, shared_reward=False, display_info=False
    )