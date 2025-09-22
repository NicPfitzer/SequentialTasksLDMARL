# Copyright (c) 2022-2025.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

import torch
from tensordict import TensorDict
from collections import deque
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, World, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment  

# Reuse your helpers
from scenarios.four_rooms.load_config import load_scenario_config
from scenarios.four_rooms.language import LanguageUnit, load_sequence_model, FIND_FIRST_SWITCH, FIND_SECOND_SWITCH, FIND_THIRD_SWITCH, FIND_GOAL, STATES, NUM_AUTOMATA

from scenarios.four_rooms.four_rooms_scenario import FourRoomsSwitchesScenario

class FourRoomsRolloutScenario(FourRoomsSwitchesScenario):
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)
        # Tunables (provide defaults if not in config)
        self.require_all_to_finish = getattr(self, "require_all_to_finish", False)
        self.gate_thickness = getattr(self, "gate_thickness", 0.15)
        self.switch_radius = getattr(self, "switch_radius", 0.25)
        self.goal_radius = getattr(self, "goal_radius", None) or self.agent_radius * 3
        self.shaping_factor = getattr(self, "shaping_factor", 0.1)
        self.collision_penalty = getattr(self, "collision_penalty", 0.1)
        self.initialized_rnn = getattr(self, "initialized_rnn", False)

        world = World(
            batch_dim, device,
            x_semidim=self.x_semidim, y_semidim=self.y_semidim
        )

        self._setup_rooms(world)
        self._prepare_state_map()
        self._prepare_obs_layout()
        self._prepare_task_state(world)
        self._init_language_unit(world)

        self._add_gates(world)
        self._add_switches(world)
        self._add_agents_and_goals(world)
        
        self._init_trails(world)
    

        return world
    
    def _prepare_state_map(self):
        
        self.state_map = {
            FIND_FIRST_SWITCH: "first",
            FIND_SECOND_SWITCH: "second",
            FIND_THIRD_SWITCH: "third",
            FIND_GOAL: "goal",
        }
    
    def _init_language_unit(self, world: World):
        # Expect your sequence model to emit a progress state p in {0,1,2,3}
        load_sequence_model(
            model_path=self.sequence_model_path,
            embedding_size=self.embedding_size,
            event_size=self.event_dim,
            state_size=getattr(self, "state_dim", 1),
            device=world.device,
        )
        
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
        
    def _add_agents_and_goals(self, world: World):
        self.team_switch_hits = torch.zeros((world.batch_dim, 3), dtype=torch.bool, device=world.device)

        self.goals = []
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                v_range=self.agent_v_range,
                f_range=self.agent_f_range,
                u_range=self.agent_u_range,
                discrete_action_nvec= [2 for _ in range(self.event_dim)]
            )
            if self.use_velocity_controller:
                agent.controller = VelocityController(
                    agent, world, ctrl_params=(2.0, 6.0, 0.002), pid_form="standard"
                )

            # Language and event buffers
            agent.h = torch.zeros((world.batch_dim, self.embedding_size), device=world.device)
            agent.y = torch.zeros_like(agent.h)
            agent.e = torch.zeros((world.batch_dim, self.event_dim), device=world.device)
            agent.state_ = torch.zeros((world.batch_dim,), device=world.device)
            agent.switch_hits = torch.zeros((world.batch_dim, 3), dtype=torch.bool, device=world.device)
            agent.on_goal = torch.zeros((world.batch_dim,), dtype=torch.bool, device=world.device)
            agent.global_shaping = torch.zeros((world.batch_dim,), dtype=torch.float32, device=world.device)

            # Per-agent goal in room 3
            goal = Landmark(
                name=f"goal_{i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=self.goal_radius),
                color=Color.LIGHT_GREEN,
            )
            agent.goal = goal
            self.goals.append(goal)
            world.add_landmark(goal)

            # Target coordinate buffer for shaping
            agent.goal_coords = torch.zeros((world.batch_dim, 2), dtype=torch.float32, device=world.device)

            world.add_agent(agent)
    
    def _prepare_task_state(self, world: World):
        # task state is the automaton state
        self.y = torch.zeros(
            (world.batch_dim, self.embedding_size), dtype=torch.float32, device=world.device
        )
        self.h = torch.zeros(
            (world.batch_dim, self.n_agents, self.embedding_size), dtype=torch.float32, device=world.device
        )
        
        self.best_state = torch.zeros((world.batch_dim,), dtype=torch.long, device=world.device)
    
    def _set_initial_gate_states(self, env_index: int = None):
        self._gates_open.zero_() if env_index is None else self._gates_open[env_index].zero_()
        
    
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


    def next_task_state(self, world: World, agent_idx: int, env_index: int = None):

        # Build event tensor: [E, F+1]
        e_all = self.team_switch_hits.float().clone()

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
        h = self.h[upd_idx, agent_idx, :]
        e = e_all[upd_idx]

        next_h, state_one_hot = self.language_unit.compute_forward_rnn(event=e, y=y, h=h)

        self.h[upd_idx, agent_idx, :] = next_h
        state_one_hot = state_one_hot.unsqueeze(0) if state_one_hot.dim() == 1 else state_one_hot
            
        states = torch.argmax(torch.sigmoid(state_one_hot[:, NUM_AUTOMATA:]), dim=-1)
        self.best_state[upd_idx] = torch.max(self.best_state[upd_idx], states)

    def set_goal_coords(self, agent: Agent, agent_idx: int, world: World, env_index: int = None):
        
        self.next_task_state(world, agent_idx, env_index)
        # Target is the NEXT required objective given progress p
        def target_xy(b: int):
            p = int(self.best_state[b].item())
            p = max(0, min(3, p))
            if p <= 2:
                return self.switches[p].state.pos[b]
            else:
                return agent.goal.state.pos[b]

        if env_index is None:
            B = world.batch_dim
            coords = torch.stack([target_xy(b) for b in range(B)], dim=0)
            agent.goal_coords = coords
        else:
            agent.goal_coords[env_index] = target_xy(env_index)
    
    # -------------------- Reset logic --------------------
    
    def _spawn_agents_for_env(self, b: int):
        
        # If we activate initialization, agents are placed in their state room
        if self.initialized_rnn:
            for i, a in enumerate(self.world.agents):
                p = int(a.state_[b].item())
                p = max(0, min(3, p))
                xmin = self.room_bounds[p][0]
                xmax = self.room_bounds[p][1]
                x = torch.zeros((1,1), device=self.world.device).uniform_(xmin + self.agent_radius, xmax - self.agent_radius)
                y = torch.zeros((1,1), device=self.world.device).uniform_(self.y_min, self.y_max)
                a.set_pos(torch.cat([x, y], dim=1), batch_index=b)
                # Set the immediate target coordinates for shaping
                self.set_goal_coords(a, i, self.world, env_index=b)
        else:
            # By default, spawn anywhere in room 0
            for i, a in enumerate(self.world.agents):
                x = torch.zeros((1,1), device=self.world.device).uniform_(self.room_bounds[0][0] + self.agent_radius, self.room_bounds[0][1] - self.agent_radius)
                y = torch.zeros((1,1), device=self.world.device).uniform_(self.y_min, self.y_max)
                a.set_pos(torch.cat([x, y], dim=1), batch_index=b)
                
                # Set the immediate target coordinates for shaping
                self.set_goal_coords(a, i, self.world, env_index=b)

    def reset_agents(self, env_index=None):
        
        self.best_state.zero_() if env_index is None else self.best_state[env_index].zero_()
        
        if env_index is None:
            self.y = self.language_unit.task_embeddings.clone()
        else:
            self.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
            
        max_min_room = torch.ones((self.world.batch_dim,), dtype=torch.int64, device=self.world.device)*3 if env_index is None else torch.tensor(3, dtype=torch.int64, device=self.world.device)
        for i, a in enumerate(self.world.agents):
            desired_state = 0
            if env_index is None:
                a.y = self.y.clone()
                a.e.zero_()
                a.state_.zero_()
                a.switch_hits.zero_()
                a.on_goal.zero_()
                
                if self.initialized_rnn:
                    if self.initial_room is not None:
                        desired_state = self.initial_room
                    else:
                        if self.even_distribution:
                            desired_state = i
                        else:
                            desired_state = None
                            
                    # We re-randomize the agent RNN state if initialized_rnn is True, so that agents don't start with the same h
                    self.language_unit.sample_sequence_dataset(env_index, rollout=False, forced_state=desired_state)
                    g = self.language_unit.language_subtask_embeddings.clone()
                    a.h = self.language_unit.convert_language_to_latent(g)
                    self.h[:, i, :] = a.h.unsqueeze(0)
                    a.state_ = self.language_unit.states.clone()
                    max_min_room = torch.min(max_min_room, a.state_)

                else:
                    
                    if self.initial_room is not None:
                        desired_state = self.initial_room
                    else:
                        if self.even_distribution:
                            desired_state = i
                        else:
                            desired_state = 0
                            
                    a.h.zero_()
                    self.h[:, i, :].zero_()
                    a.state_[:] = desired_state
            else:
                a.y[env_index] = self.y[env_index].clone()
                a.e[env_index].zero_()
                a.state_[env_index].zero_()
                a.switch_hits[env_index].zero_()
                a.on_goal[env_index] = False
                if self.initialized_rnn:
                    
                    if self.initial_room is not None:
                        desired_state = self.state_map[self.initial_room]
                    else:
                        if self.even_distribution:
                            desired_state = i
                            a.switch_hits[env_index, :desired_state] = True
                        else:
                            desired_state = None

                    self.language_unit.sample_sequence_dataset(env_index, rollout=False, forced_state=desired_state)
                    g = self.language_unit.language_subtask_embeddings[env_index].clone()
                    a.h[env_index] = self.language_unit.convert_language_to_latent(g)
                    self.h[env_index, i, :] = a.h[env_index].unsqueeze(0)
                    a.state_[env_index] = self.language_unit.states[env_index].clone()
                    max_min_room = torch.min(max_min_room, a.state_[env_index])
                else:
                    
                    if self.initial_room is not None:
                        desired_state = self.initial_room
                    else:
                        if self.even_distribution:
                            desired_state = i
                        else:
                            desired_state = 0
                            
                    a.h[env_index].zero_()
                    self.h[env_index, i, :].zero_()
                    a.state_[env_index] = desired_state

        # Spawn agents based on progress state and set shaping baselines
        if env_index is None:
            
            if  self.initialized_rnn and (self.initial_room is not None or (self.initial_room is None and not self.even_distribution)):
                for a in self.world.agents:
                    B = self.world.batch_dim
                    C = a.switch_hits.size(-1)  # number of switches/rooms
                    cols = torch.arange(C, device=self.world.device).unsqueeze(0).expand(B, C)   # [B, C]
                    thresh = torch.clamp(max_min_room, 0, C).view(B, 1)                           # [B, 1]
                    mask = cols < thresh                                                           # [B, C] bool
                    a.switch_hits |= mask

            for b in range(self.world.batch_dim):
                self._spawn_agents_for_env(b)
        else:

            if self.initialized_rnn and (self.initial_room is not None or (self.initial_room is None and not self.even_distribution)):
                for a in self.world.agents:
                    a.switch_hits[env_index, :int(max_min_room.item())] = True

            self._spawn_agents_for_env(env_index)

        # Set initial shaping
        for a in self.world.agents:
            if env_index is None:
                a.global_shaping = torch.linalg.vector_norm(a.state.pos - a.goal_coords, dim=1) * self.shaping_factor
            else:
                a.global_shaping[env_index] = torch.linalg.vector_norm(
                    a.state.pos[env_index] - a.goal_coords[env_index], dim=0
                ) * self.shaping_factor
        
        self._clear_trails(env_index)
                
    #--------------------- Observation --------------------
    
    def observation_core(self, agent: Agent):
        obs = torch.empty((self.world.batch_dim, self._obs_dim), device=self.world.device, dtype=torch.float32)

        # Switches rel pos [B, 3, 2]
        s_pos = torch.stack([sw.state.pos for sw in self.switches], dim=1)
        obs[:, self._sl_switches] = (s_pos - agent.state.pos.unsqueeze(1)).reshape(self.world.batch_dim, 6) * 2

        # Goal rel pos
        obs[:, self._sl_goal] = (agent.goal.state.pos - agent.state.pos) * 2

        # Events vector: [s1_hit, s2_hit, s3_hit]
        events = torch.zeros((self.world.batch_dim, self.event_dim), device=self.world.device)
        events[:, 0:3] = agent.switch_hits.float()
        obs[:, self._sl_events] = events

        return {
            "pos": agent.state.pos * 2,
            "vel": agent.state.vel * 2,
            "sentence_embedding": agent.h,
            "obs": obs,
        }
        
    def observation(self, agent: Agent):
        
        obs_dict = self.observation_core(agent)
        obs_dict["event"] = agent.e
        obs_dict["task_state"] = self.best_state.float().clone()
        obs_dict["agent_state"] = agent.state_.float()
        return obs_dict
    
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
    
    #--------------------- Action processing --------------------

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
                
                a.action.u = actions.get(("agents", "action"))[:,i,:] / 2
    
    def pre_step(self):

        for i, agent in enumerate(self.world.agents):
            # agent.h: [E, H], agent.y: [E, Y] (batched per env)
            h = agent.h.clone()
            y = agent.y.clone()
            #e = agent.e.clone()
            e = self.team_switch_hits.float().clone()
            next_h, next_state = self.language_unit.compute_forward_rnn(
                event=e,
                y=y,
                h=h,
            )

            # Write back only for the changed envs
            agent.h = next_h
            agent.state_ = torch.argmax(torch.sigmoid(next_state[:,NUM_AUTOMATA:]), dim=-1)
            self.set_goal_coords(agent, i, self.world)
    
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
            task_state = self.best_state[env_index].item()
            self._record_trail(env_index)

            for i, agent1 in enumerate(self.world.agents):
                
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
                
                # Agent State
                agent_color = self.state_color_map[agent1.state_[env_index].item()].value
                circle = rendering.make_circle(radius=self.agent_radius / 3, filled=True)
                xform = rendering.Transform()
                xform.set_translation(*agent1.state.pos[env_index])
                circle.add_attr(xform)
                circle.set_color(*agent_color)
                geoms.append(circle)
                
                # Add rings for visited switches
                for j, state in self.switch_state_map.items():
                    if agent1.switch_hits[env_index, j]:
                        ring = rendering.make_circle(
                            radius=0.05 + j * 0.02,
                            filled=False,
                        )
                        ring.set_linewidth(3)
                        xform = rendering.Transform()
                        xform.set_translation(*agent1.state.pos[env_index])
                        ring.add_attr(xform)
                        ring.set_color(*self.state_color_map[state].value)
                        geoms.append(ring)
                    
                # Communication lines
                for j, agent2 in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    if self.world.get_distance(agent1, agent2)[env_index] <= self.comms_radius * (self.x_semidim + self.y_semidim)/2:
                        line = rendering.Line(
                            agent1.state.pos[env_index], agent2.state.pos[env_index], width=5
                        )
                        line.set_color(*Color.GRAY.value)
                        line.set_linewidth(5)
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
    scenario = FourRoomsRolloutScenario()
    render_interactively(
        scenario,
        control_two_agents=True,
        n_agents=3,
        shared_reward=True,
        display_info=False,
    )
    