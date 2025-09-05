# Copyright (c) 2022-2025.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, World, Sphere, Line
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, TorchUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

# Reuse your helpers
from scenarios.four_rooms.load_config import load_scenario_config
from scenarios.four_rooms.language import LanguageUnit, load_sequence_model, FIND_FIRST_SWITCH, FIND_SECOND_SWITCH, FIND_THIRD_SWITCH, FIND_GOAL, NUM_AUTOMATA

class FourRoomsSwitchesScenario(BaseScenario):
    """
    4 rooms along +x, 3 vertical gates between rooms.
    In rooms 0,1,2 there is one switch each. Room 3 contains per-agent goals.
    LanguageUnit.states gives a progress p in {0,1,2,3}:
      p=0 -> target switch_1, spawn in room 0, all gates closed
      p=1 -> target switch_2, gate_1 open, spawn in rooms 0+1
      p=2 -> target switch_3, gates_1..2 open, spawn in rooms 0+1+2
      p=3 -> target goal, gates_1..3 open, spawn in rooms 0+1+2
    This scenario optimizes only for the current subtask; episode succeeds when
    the current target is reached (any agent, or all agents if configured).
    """

    # -------------------- Scenario setup --------------------

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        load_scenario_config(kwargs, self)
        # Tunables (provide defaults if not in config)
        self.require_all_to_finish = getattr(self, "require_all_to_finish", False)
        self.gate_thickness = getattr(self, "gate_thickness", 0.15)
        self.switch_radius = getattr(self, "switch_radius", 0.25)
        self.goal_radius = getattr(self, "goal_radius", None) or self.agent_radius * 3
        self.shaping_factor = getattr(self, "shaping_factor", 0.1)
        self.collision_penalty = getattr(self, "collision_penalty", 0.1)

        world = World(
            batch_dim, device,
            x_semidim=self.x_semidim, y_semidim=self.y_semidim
        )

        self._setup_rooms(world)
        self._prepare_obs_layout()
        self._prepare_task_state(world)
        self._init_language_unit(world)

        self._add_gates(world)
        self._add_switches(world)
        self._add_agents_and_goals(world)

        return world

    def _setup_rooms(self, world: World):
        self._gates_open = torch.zeros((world.batch_dim, 3), dtype=torch.bool, device=world.device)
        # Define 4 equal rooms along x in the safe box that respects agent radius
        x_min = -world.x_semidim + self.agent_radius
        x_max =  world.x_semidim - self.agent_radius
        total_w = x_max - x_min
        room_w = total_w / 4.0

        # Room boundaries [ (xmin,xmax) for room i ]
        self.room_bounds = [
            (x_min + i*room_w, x_min + (i+1)*room_w) for i in range(4)
        ]
        # Gate x-positions at the dividers between rooms i and i+1
        self.gate_x = [self.room_bounds[i][1] for i in range(3)]
        self.y_min = -world.y_semidim + self.agent_radius
        self.y_max =  world.y_semidim - self.agent_radius

    def _prepare_obs_layout(self):
        # obs = [switches(3)*2, goal*2, events(4)]
        self.event_dim = getattr(self, "event_dim", 3)
        self._obs_dim = 2*3 + 2 + self.event_dim
        i = 0
        self._sl_switches = slice(i, i + 6); i += 6
        self._sl_goal     = slice(i, i + 2); i += 2
        self._sl_events   = slice(i, i + self.event_dim); i += self.event_dim
        assert i == self._obs_dim

    def _init_language_unit(self, world: World):
        # Expect your sequence model to emit a progress state p in {0,1,2,3}
        load_sequence_model(
            model_path=self.sequence_model_path,
            embedding_size=self.embedding_size,
            event_size=self.event_dim,
            state_size=getattr(self, "state_dim", 1),
            device=world.device,
        )
        self.language_unit = LanguageUnit(
            batch_size=world.batch_dim,
            embedding_size=self.embedding_size,
            use_embedding_ratio=self.use_embedding_ratio,
            device=world.device,
        )
        self.language_unit.load_sequence_data(json_path=self.data_json_path, device=world.device)
    
    def _prepare_task_state(self, world: World):
        # task state is the automaton state
        self.y = torch.zeros(
            (world.batch_dim, self.embedding_size), dtype=torch.float32, device=world.device
        )
        self.h = torch.zeros(
            (world.batch_dim, self.n_agents, self.embedding_size), dtype=torch.float32, device=world.device
        )
        
        self.best_state = torch.zeros((world.batch_dim,), dtype=torch.long, device=world.device)
    
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

    # -------------------- Entities --------------------

    def _add_gates(self, world: World):
        # Vertical slabs that block passage until opened (moved off-screen)
        self.gates = []
        for i, gx in enumerate(self.gate_x, start=1):
            gate = Landmark(
                name=f"gate_{i}",
                collide=True,
                movable=False,
                # skinny in x, spans full y
                shape=Box(length=self.gate_thickness, width=2*world.y_semidim),
                color=Color.GRAY,
            )
            world.add_landmark(gate)
            self.gates.append(gate)

    def _add_switches(self, world: World):
        # One switch centered randomly in rooms 0,1,2
        self.state_color_map = {FIND_FIRST_SWITCH: Color.YELLOW, FIND_SECOND_SWITCH: Color.ORANGE,
                                FIND_THIRD_SWITCH: Color.PURPLE, FIND_GOAL: Color.LIGHT_GREEN}
        self.switch_state_map = {0: FIND_FIRST_SWITCH, 1: FIND_SECOND_SWITCH, 2: FIND_THIRD_SWITCH}
        self.switches = []
        for i in range(3):
            sw = Landmark(
                name=f"switch_{i+1}",
                collide=False,
                movable=False,
                shape=Sphere(radius=self.switch_radius),
                color=self.state_color_map[self.switch_state_map[i]],
            )
            world.add_landmark(sw)
            self.switches.append(sw)

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
    
    # -------------------- Reset logic --------------------

    def reset_agents(self, env_index=None):
        
        if env_index is None:
            self.y = self.language_unit.task_embeddings.clone()
            self.h.zero_()
            self.best_state.zero_()
        else:
            self.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
            self.h[env_index].zero_()
            self.best_state[env_index].zero_()
        
        for a in self.world.agents:
            if env_index is None:
                a.y = self.language_unit.task_embeddings.clone()
                a.h = self.language_unit.rnn_subtask_embeddings.clone()
                a.e.zero_()
                a.switch_hits.zero_()
                a.on_goal.zero_()
            else:
                a.y[env_index] = self.language_unit.task_embeddings[env_index].clone()
                a.h[env_index] = self.language_unit.rnn_subtask_embeddings[env_index].clone()
                a.e[env_index].zero_()
                a.switch_hits[env_index].zero_()
                a.on_goal[env_index] = False

        # Spawn agents based on progress state and set shaping baselines
        if env_index is None:
            for b in range(self.world.batch_dim):
                self._spawn_agents_for_env(b)
        else:
            self._spawn_agents_for_env(env_index)

        # Set initial shaping
        for a in self.world.agents:
            if env_index is None:
                a.global_shaping = torch.linalg.vector_norm(a.state.pos - a.goal_coords, dim=1) * self.shaping_factor
            else:
                a.global_shaping[env_index] = torch.linalg.vector_norm(
                    a.state.pos[env_index] - a.goal_coords[env_index], dim=0
                ) * self.shaping_factor

    def reset_world_at(self, env_index: int = None):
        # Reset language and sample a progress state p in {0,1,2,3}
        if env_index is None:
            self.language_unit.reset_all()
        else:
            self.language_unit.reset_env(env_index)
        self.language_unit.sample_sequence_dataset(env_index)
        self.language_unit.states.zero_()

        # Place switches and goals
        self._place_switches(env_index)
        self._place_goals(env_index)

        # Open gates according to progress p at reset
        self._set_initial_gate_states(env_index)

        # Reset team vars
        if env_index is None:
            self.team_switch_hits.zero_()
        else:
            self.team_switch_hits[env_index].zero_()
        
        self.reset_agents(env_index)

        # Position gates (closed gates at divider x, open gates off-screen)
        self._position_gates(env_index)

    def _place_switches(self, env_index: int = None):
        # Random inside each room [0..2] with a margin
        margin = max(self.switch_radius, self.agent_radius)
        rooms = [0, 1, 2]
        for i, sw in enumerate(self.switches):
            xmin, xmax = self.room_bounds[rooms[i]]
            xx = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1),
                             device=self.world.device, dtype=torch.float32).uniform_(xmin+margin, xmax-margin)
            yy = torch.zeros_like(xx).uniform_(self.y_min+margin, self.y_max-margin)
            sw.set_pos(torch.cat([xx, yy], dim=1), batch_index=env_index)

    def _place_goals(self, env_index: int = None):
        # Place a stacked column of goals in room 3 with vertical spacing
        xmin, xmax = self.room_bounds[3]
        margin = max(self.goal_radius, self.agent_radius)
        base_x = torch.zeros((1, 1) if env_index is not None else (self.world.batch_dim, 1),
                             device=self.world.device, dtype=torch.float32).uniform_(xmin+margin, xmax-margin)
        base_y = torch.zeros_like(base_x).uniform_(self.y_min + (3*self.agent_radius + self.agent_spacing),
                                                   self.y_max - (3*self.agent_radius + self.agent_spacing))
        for i, goal in enumerate(self.goals):
            offset = torch.tensor([[0.0, (i - (self.n_agents - 1)/2)*self.agent_spacing]],
                                  device=self.world.device)
            pos = torch.cat([base_x,base_y],dim=1) + offset
            goal.set_pos(pos, batch_index=env_index)

    def _set_initial_gate_states(self, env_index: int = None):
        self._gates_open.zero_() if env_index is None else self._gates_open[env_index].zero_()

    def _spawn_agents_for_env(self, b: int):

        max_min_room = 3
        for i, a in enumerate(self.world.agents):

            # Spawn in accessible space up to the next closed gate
            if self.initial_room is None:
                if self.even_distribution:
                    p = i
                else:
                    p = torch.randint(0, 4, (1,), device=self.world.device).item()
                    max_min_room = min(max_min_room, p)
            else:
                p = self.initial_room
                a.switch_hits[b, :p] = True
            
            rightmost_room = min(p, 2)
            xmin = self.room_bounds[rightmost_room][0]
            xmax = self.room_bounds[rightmost_room][1]
            
            x = torch.zeros((1,1), device=self.world.device).uniform_(xmin + self.agent_radius, xmax - self.agent_radius)
            y = torch.zeros((1,1), device=self.world.device).uniform_(self.y_min, self.y_max)
            a.set_pos(torch.cat([x, y], dim=1), batch_index=b)

            # Set the immediate target coordinates for shaping
            self.set_goal_coords(a, i, self.world, env_index=b)
        
        if self.initial_room is None and not self.even_distribution:
            for a in self.world.agents:
                a.switch_hits[b, :max_min_room] = True

    def _position_gates(self, env_index: int = None):
        # Closed gate i sits at x = gate_x[i]; open gates are moved far away
        if env_index is None:
            for i, gate in enumerate(self.gates):
                closed_mask = ~self._gates_open[:, i]
                open_mask = self._gates_open[:, i]
                pos = torch.zeros((self.world.batch_dim, 2), device=self.world.device)
                # Place closed
                if closed_mask.any():
                    pos[closed_mask, 0] = self.gate_x[i]
                    pos[closed_mask, 1] = 0.0
                    gate.set_pos(pos, batch_index=None)
                # Hide open
                if open_mask.any():
                    pos[open_mask] = self._get_outside_pos(None)[open_mask]
                    gate.set_pos(pos, batch_index=None)
        else:
            for i, gate in enumerate(self.gates):
                if self._gates_open[env_index, i]:
                    gate.set_pos(self._get_outside_pos(env_index), batch_index=env_index)
                else:
                    gate.set_pos(torch.tensor([self.gate_x[i], 0.0], device=self.world.device), batch_index=env_index)

    # -------------------- Target selection --------------------

    @torch.no_grad()
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

    # -------------------- Stepping --------------------

    def process_action(self, agent):
        agent.action.u = TorchUtils.clamp_with_norm(agent.action.u, agent.u_range)
        action_norm = torch.linalg.vector_norm(agent.action.u, dim=1)
        agent.action.u[action_norm < 0.005] = 0
        if self.use_velocity_controller and not self.use_kinematic_model:
            agent.controller.process_force()
            
    def pre_step(self):

        for i, agent in enumerate(self.world.agents):

            self.set_goal_coords(agent, i, self.world)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device)
        # Shaping toward the current target
        dist_to_target = torch.linalg.vector_norm(agent.state.pos - agent.goal_coords, dim=1)
        agent_shaping = dist_to_target * self.shaping_factor
        self.rew += (agent.global_shaping - agent_shaping)
        agent.global_shaping = agent_shaping

        # Collisions with closed gates penalize
        for i, gate in enumerate(self.gates):
            # If a gate is open it is off-screen so overlap is false
            self.rew[self.world.is_overlapping(agent, gate)] -= self.collision_penalty

        # Subtask success detection
        # Determine current target per env
        p = self.language_unit.states.long().clamp_min(0).clamp_max(3)  # [B]
        b = torch.arange(self.world.batch_dim, device=self.world.device)

        # Overlaps
        overlaps_switch = [self.world.is_overlapping(agent, sw).bool() for sw in self.switches]  # 3 tensors [B]
        overlap_goal = self.world.is_overlapping(agent, agent.goal).bool()

        # Update per-agent and team events
        for k in range(3):
            agent.switch_hits[:, k] |= overlaps_switch[k]
        self.team_switch_hits |= agent.switch_hits
        agent.on_goal |= overlap_goal

        # Give sparse bonus when hitting the active target only
        # p in {0,1,2} -> switch p; p==3 -> goal
        for k in range(3):
            mask = overlaps_switch[k] & (p == k)
            self.rew[mask] += 0.05
        self.rew[overlap_goal & (p == 3)] += 0.05

        # Open gates immediately when the corresponding switch is hit
        for k in range(3):
            just_open = self.team_switch_hits[:, k] & (~self._gates_open[:, k])
            if just_open.any():
                self._gates_open[just_open, k] = True
        # Move opened gates off-screen
        self._position_gates(env_index=None)

        # Edge penalty
        sx = self.world.x_semidim - self.agent_radius
        sy = self.world.y_semidim - self.agent_radius
        x, y = agent.state.pos[:, 0], agent.state.pos[:, 1]
        eps = getattr(self, "edge_eps", 0.0)
        hit_edge = (x >= sx - eps) | (x <= -sx + eps) | (y >= sy - eps) | (y <= -sy + eps)
        self.rew[hit_edge] -= getattr(self, "edge_penalty", 1.0)

        # Soft separation penalty
        min_sep = getattr(self, "min_separation", self.agent_spacing * 0.8)
        sep_weight = self.collision_penalty
        pen = torch.zeros(self.world.batch_dim, device=self.world.device)
        for other in self.world.agents:
            if other is agent:
                continue
            d = torch.linalg.vector_norm(agent.state.pos - other.state.pos, dim=1)
            short = (min_sep - d).clamp_min(0.0)
            pen += sep_weight * short * short
        self.rew -= pen

        return self.rew * 0.01

    def observation(self, agent: Agent):
        obs = torch.empty((self.world.batch_dim, self._obs_dim), device=self.world.device, dtype=torch.float32)

        # Switches rel pos [B, 3, 2]
        s_pos = torch.stack([sw.state.pos for sw in self.switches], dim=1)
        obs[:, self._sl_switches] = (s_pos - agent.state.pos.unsqueeze(1)).reshape(self.world.batch_dim, 6)

        # Goal rel pos
        obs[:, self._sl_goal] = agent.goal.state.pos - agent.state.pos

        # Events vector: [s1_hit, s2_hit, s3_hit]
        events = torch.zeros((self.world.batch_dim, self.event_dim), device=self.world.device)
        events[:, 0:3] = self.team_switch_hits.float()
        obs[:, self._sl_events] = events

        return {
            "pos": agent.state.pos,
            "vel": agent.state.vel,
            "sentence_embedding": torch.zeros((self.world.batch_dim,1),device=self.world.device, dtype=torch.float32),
            "obs": obs,
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
        

    # -------------------- Utils --------------------

    def _get_outside_pos(self, env_index):
        return torch.empty(
            (1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p),
            device=self.world.device
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)
        

    def extra_render(self, env_index: int = 0):
            from vmas.simulator import rendering
            geoms = []
            task_state = self.best_state[env_index].item()

            for i, agent1 in enumerate(self.world.agents):
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
                        line.set_color(*self.state_color_map[task_state].value)
                        line.set_linewidth(2)
                        geoms.append(line)
            
            try:
                sentence = "RL"
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
    scenario = FourRoomsSwitchesScenario()
    render_interactively(
        scenario,
        control_two_agents=True,
        n_agents=2,
        shared_reward=True,
        display_info=False,
    )
