import sys
import csv
import math
import shutil
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import speech_recognition as sr

from tensordict import TensorDict
from benchmarl.utils import DEVICE_TYPING
from vmas.simulator.utils import TorchUtils

from scenarios.exploration import agent
from sequence_models.four_flags.model_training.rnn_model import EventRNN
from scenarios.four_flags.load_config import load_scenario_config

# ROS2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8

USING_FREYJA = False;
CREATE_MAP_FRAME = True;

if USING_FREYJA:
    sys.path.insert(0, "/home/npfitzer/robomaster_ws/install/freyja_msgs/lib/python3.10/site-packages")
    from freyja_msgs.msg import ReferenceState
    from freyja_msgs.msg import CurrentState
    from freyja_msgs.msg import WaypointTarget
else:
    from geometry_msgs.msg import Twist, PoseStamped, Pose
    from nav_msgs.msg import Odometry
    from tf_transformations import euler_from_quaternion


# Local Modules
from deployment.helper_utils import convert_ne_to_xy, convert_xy_to_ne
from trainer.benchmarl_setup_experiment_ import benchmarl_setup_experiment

X = 0
Y = 1

class State:
    def __init__(self, pos, vel, rot, device):
        self.device = device
        self.pos = torch.tensor(pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.vel = torch.tensor(vel, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.rot = torch.tensor(rot, dtype=torch.float32, device=self.device).unsqueeze(0)

class Agent:
    def __init__(
    self,
    node,
    robot_id: int,
    count: int,
    deployment_config,
    device: DEVICE_TYPING):
        
        self.timer: Optional[rclpy.timer.Timer] = None
        
        self.goal = None
        self._scale = torch.tensor([node.x_semidim, node.y_semidim], device=device)
        
        self.node = node
        self.robot_id = robot_id
        self.count = count
    
        # Timer to update state
        self.obs_dt = deployment_config.obs_dt
        self.mytime = 0.0
        
        # State buffer
        self.state_buffer = []
        self.state_buffer_length = 5
        
        self.a_range = deployment_config.a_range
        self.v_range = deployment_config.v_range
        self.device = device
        self.state_received = False
        if CREATE_MAP_FRAME:
            self.home_pn = None;
            self.home_pe = None;
        else:
            self.home_pn = 0.0;
            self.home_pe = 0.0;

        # State variables updated by current_state_callback
        self.state = State(
            pos=[0.0, 0.0],
            vel=[0.0, 0.0],
            rot=[0.0],
            device=self.device
        )
        
        self.set_goal()
        
        self.h = torch.zeros((self.node.embedding_size,), dtype=torch.float32, device=self.device)
        self.e = torch.zeros((self.node.event_size,), dtype=torch.float32, device=self.device)
        self.flags_found = torch.zeros((4,), dtype=torch.bool, device=self.device)
        self.hit_switch = torch.zeros((1,), dtype=torch.bool, device=self.device)
        self.task_state = torch.zeros((1,), dtype=torch.float32, device=self.device)
        self._prepare_obs_layout()

        # Get topic prefix from config or use default
        topic_prefix = getattr(deployment_config, "topic_prefix", "/robomaster_")
        
        # Create publisher for the robot
        if USING_FREYJA:
            self.pub = self.node.create_publisher(
                ReferenceState,
                f"{topic_prefix}{self.robot_id}/reference_state",
                1
            )
        else:
            self.pub = self.node.create_publisher(
                Twist,
                "/willow1/cmd_vel",
                1
            )

        
        # Create subscription with more descriptive variable name
        if USING_FREYJA:
            self.state_subscription = self.node.create_subscription(
                CurrentState,
                f"{topic_prefix}{self.robot_id}/current_state",
                self.freyja_current_state_callback,
                1
            )
        else:
            self.state_subscription = self.node.create_subscription(
                Odometry,
                "/willow/odometry/gps",
                self.odom_current_state_callback,
                1
        )
        
        # Log the subscription
        self.node.get_logger().info(f"Robot {self.robot_id} subscribing to: {topic_prefix}{self.robot_id}/current_state")
    
        # Create reference state message
        if USING_FREYJA:
            self.reference_state = ReferenceState()
        else:
            self.reference_state = Twist()
    
    def set_goal(self):
        
        y_min = -self.node.y_semidim + self.node.agent_radius + self.node.agent_radius * 3
        y_max = self.node.y_semidim - self.node.agent_radius - self.node.agent_radius * 3
        x_min = self.node.x_semidim - self.node.chamber_width + self.node.agent_radius + self.node.agent_radius * 3 + self.node.passage_length / 2
        x_max = self.node.x_semidim - self.node.agent_radius - self.node.agent_radius * 3
        x_center = (x_min + x_max) / 2
        n_agent = self.node.n_agents
        ratio = self.count / (n_agent-1)
        y = y_min + ratio * (y_max - y_min)
        x = x_center
        self.goal = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)

    def _prepare_obs_layout(self):

        # 2 coords per relative vector
        self._obs_dim = 2*(1 + 1 + 4)  # passages + goal + switch + 4 flags
        # slice index helpers
        i = 0
        self._sl_goal     = slice(i, i + 2);   i += 2
        self._sl_switch   = slice(i, i + 2);   i += 2
        self._sl_flags    = slice(i, i + 8)

    def freyja_current_state_callback(self, msg: CurrentState):
        # Extract current state values from the state vector
        current_pos_n = msg.state_vector[0]
        current_pos_e = msg.state_vector[1]
        current_rot = msg.state_vector[5]
        current_vel_n = msg.state_vector[3]
        current_vel_e = msg.state_vector[4]
        
        self.state.pos[0,X], self.state.pos[0,Y] = convert_ne_to_xy(current_pos_n, current_pos_e)
        self.state.vel[0,X], self.state.vel[0,Y] = convert_ne_to_xy(current_vel_n, current_vel_e)
        self.state.rot[0] = current_rot
        
        self.state_received = True

    def odom_current_state_callback(self, msg: Odometry):
        # Extract current state values from the state vector
        euler = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]);

        current_pos_n = msg.pose.pose.position.x;
        current_pos_e = msg.pose.pose.position.y;
        current_vel_n = msg.twist.twist.linear.x;
        current_vel_e = msg.twist.twist.linear.y;

        if CREATE_MAP_FRAME and self.home_pe is None:
            self.home_pn = current_pos_n;
            self.home_pe = current_pos_e;

        current_pos_n = (current_pos_n - self.home_pn);
        current_pos_e = (current_pos_e - self.home_pe);
        print(f"Robot {self.robot_id} - Euler[2]: {euler[2]:.4f}, pos_n: {current_pos_n:.4f}, pos_e: {current_pos_e:.4f}")

        self.state.pos[0,X], self.state.pos[0,Y] = convert_ne_to_xy(current_pos_n, current_pos_e)
        self.state.vel[0,X], self.state.vel[0,Y] = convert_ne_to_xy(current_vel_n, current_vel_e)
        self.state.rot[0] = euler[2];

        self.state_received = True
    
    def landmark_reached(self, landmark_type: str, landmark_id: int):
        if landmark_type == "flag":
            self.flags_found[landmark_id] = True
        elif landmark_type == "switch":
            self.hit_switch = True
    
    def landmark_overlap(self):
        # Check for overlap with flags
        for i, flag_pos in enumerate(self.node.flags):
            distance = torch.norm(self.state.pos - flag_pos.unsqueeze(0))
            if distance < self.node.agent_radius * 3:
                self.landmark_reached("flag", i)
                self.node.get_logger().info(f"Robot {self.robot_id} reached flag {i} at position {flag_pos.cpu().numpy()}")
        
        # Check for overlap with switch
        distance_to_switch = torch.norm(self.state.pos - self.node.switch.unsqueeze(0))
        if distance_to_switch < self.node.agent_radius * 3:
            self.landmark_reached("switch", 0)
            self.node.get_logger().info(f"Robot {self.robot_id} reached the switch at position {self.node.switch.cpu().numpy()}")

    def next_step_rnn(self):
        
        self.e = torch.cat([self.flags_found.to(torch.float32), self.hit_switch.to(torch.float32)], dim=-1)
        self.h, state_decoder_out = self.node.compute_forward_rnn(
            event=self.e.unsqueeze(0),
            y=self.node.y.unsqueeze(0),
            h=self.h.unsqueeze(0)
        )
        self.task_state = torch.argmax(torch.sigmoid(state_decoder_out[:, self.node.num_automata:]), dim=-1)

    def collect_observation(self):
        
        self.landmark_overlap()
        self.next_step_rnn()
        
        if self.goal is not None and self.state_received:
            
            obs_buf = torch.empty((self._obs_dim,),
                            device=self.device, dtype=torch.float32)
            obs_buf[:, self._sl_goal]   = (self.goal - self.state.pos) / self._scale
            obs_buf[:, self._sl_switch] = (self.node.switch - self.state.pos) / self._scale

            flags_pos = torch.stack([f for f in self.node.flags], dim=1)
            obs_buf[:, self._sl_flags] = (flags_pos - self.state.pos) / self._scale

            obs = {
                "pos": self.state.pos / self._scale,
                "vel": self.state.vel / self._scale,
                "event": self.e,
                "sentence_embedding": self.h,
                "obs": obs_buf,
            }
            
            self.state_buffer.append(obs)
            if len(self.state_buffer) > self.state_buffer_length:
                self.state_buffer = self.state_buffer[-self.state_buffer_length:]
       
        
    def send_zero_velocity(self):
        # Send a zero velocity command
        if USING_FREYJA:
            self.reference_state.vn = 0.0
            self.reference_state.ve = 0.0
            self.reference_state.header.stamp = self.node.get_clock().now().to_msg()
        else:
            self.reference_state.linear.x = 0.0
            self.reference_state.angular.z = 0.0

        self.node.get_logger().info(f"Robot {self.robot_id} - Zero velocity command sent.")
        self.pub.publish(self.reference_state)
        self.node.log_file.flush()
    
class World:
    def __init__(self, agents: List[Agent], dt: float):
        self.agents = agents
        self.dt = dt

class VmasModelsROSInterface(Node):

    def __init__(self, config_multitask: DictConfig, config_team_gnn: DictConfig, log_dir: Path):
        super().__init__("vmas_ros_interface")
        self.device = config_multitask.device 
        arena_config = config_multitask["arena_config"]
        deployment_config = config_multitask["deployment"]
        
        task_config = config_multitask["task"].params
        load_scenario_config(task_config,self)
        
        # Override environment dimensions
        self.x_semidim = arena_config.x_semidim
        self.y_semidim = arena_config.y_semidim
        self.task_x_semidim = task_config.x_semidim
        self.task_y_semidim = task_config.y_semidim

        # Load Action Policy
        cfg, seed = generate_cfg(config_path=self.policy_config_path, config_name=self.policy_config_name, restore_path=self.policy_restore_path)
        self.policy = get_policy_from_cfg(cfg, seed)
        
        # experiment_team_gnn = benchmarl_setup_experiment(cfg=config_team_gnn)
        # self.team_gnn_policy = experiment_team_gnn.policy
        
        # Load sequence model
        self.load_sequence_model(
            model_path=self.sequence_model_path,
            embedding_size=self.embedding_size,
            event_size=self.event_dim,
            state_size=self.state_dim,
            device=self.device
        )
        
        self.set_landmarks()

        # Create Agents
        agents: List[Agent] = []
        id_list = deployment_config.id_list
        assert len(id_list) == self.n_agents
        for i in range(self.n_agents):
            agent = Agent(
                node=self,
                robot_id=id_list[i],
                count = i,
                deployment_config = deployment_config,
                device=self.device
            )
            agents.append(agent)
        
        # Create action loop
        self.action_dt = deployment_config.action_dt
        self.max_steps = deployment_config.max_steps
        self.step_count = 0
        self.world = World(agents,self.action_dt)
        
        self.get_logger().info("ROS2 starting ..")
    
    def set_landmarks(self):
        """Set the landmarks for the scenario."""
        self.flags = []
        
        self.x_room_max = (self.x_semidim - self.chamber_width - self.agent_radius - self.passage_length / 2)
        self.x_room_min = (-self.x_semidim + self.agent_radius)
        x_room_center = (self.x_room_max + self.x_room_min) / 2
        y_room_center = 0
        self.room_center =  torch.tensor([x_room_center, y_room_center], dtype=torch.float32, device=self.world.device)
        self.switch = self.room_center.clone()
        
        # Set flag positions
        indices = [0,1,2,3] # Red, Green, Blue, Yellow
        for i in indices:
            xx = self.x_room_min + self.agent_radius * 3 + (i % 2) * (self.x_room_max - self.x_room_min - 2 * (self.agent_radius * 3))
            yy = -self.y_semidim + self.agent_radius + self.agent_radius * 3 + (i // 2) * (2 * self.y_semidim - 2 * (self.agent_radius + self.agent_radius * 3))
            flag = torch.zeros((2,), dtype=torch.float32, device=self.world.device)
            flag[0] = xx
            flag[1] = yy
            self.flags.append(flag)
            print(f"Flag {i} position: {flag}")
        
        # Wall center
        self.wall_center = torch.tensor([self.x_semidim - self.chamber_width, 0.0], dtype=torch.float32, device=self.world.device)
    
    def compute_forward_rnn(self, event: torch.Tensor, y: torch.Tensor, h: torch.Tensor):
        """
        Compute the next state of the RNN for the given environments.
        """
        next_h, state_decoder_out = self.sequence_model._forward(e=event, y=y, h=h)

        return next_h, state_decoder_out

    def load_sequence_model(self, model_path, embedding_size, event_size, state_size, device):
        """ Load the sequence model from a given path."""
        self.sequence_model = EventRNN(event_dim=event_size, y_dim=embedding_size, latent_dim=embedding_size, input_dim=64, state_dim=state_size, decoder=None)
        self.sequence_model.load_state_dict(torch.load(model_path, map_location=device))
        self.sequence_model.eval()

    def timer_callback(self):
        if not self._all_states_received():
            self.get_logger().info("Waiting for all agents to receive state.")
            return

        if self._reached_max_steps():
            self._handle_termination()
            return

        obs_list = self._collect_observations()
        if not obs_list:
            self.get_logger().warn("No valid observations collected. Skipping this timestep.")
            return

        input_td = self._prepare_input_tensor(obs_list)

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            output_td = self.multitask_policy(input_td)
        action = output_td[("agents", "action")]

        self._issue_commands_to_agents(action)
        self.step_count += 1

    def _all_states_received(self):
        return all(agent.state_received for agent in self.world.agents)

    def _reached_max_steps(self):
        return self.step_count >= self.max_steps

    def _handle_termination(self):
        self.get_logger().info(f"Robots reached {self.max_steps} steps. Stopping.")
        self.timer.cancel()
        self.stop_all_agents()
        self.prompt_for_new_instruction()

    def _collect_observations(self):
        obs_list = []

        for agent in self.world.agents:
            if not agent.state_buffer:
                self.get_logger().warn(f"No state in buffer for agent {agent.robot_id}")
                return

            latest_state = agent.state_buffer[-1]
            feat = torch.cat([
                latest_state["pos"],
                latest_state["vel"],
                latest_state["event"],
                latest_state["sentence_embedding"],
                latest_state["obs"],], dim=-1).float()   # shape (1, 7)
            obs_list.append(feat)

        return obs_list

    def _prepare_input_tensor(self, obs_list):
        obs_tensor = torch.cat(obs_list, dim=0)

        return TensorDict({
            ("agents", "observation"): obs_tensor,
        }, batch_size=[len(obs_list)])
    
    def clamp_velocity_to_bounds(self, action: torch.Tensor, agent) -> List[float]:
        """
        Clamp the velocity so the agent remains within the environment bounds,
        accounting for the agent's radius and timestep.
        """
        pos = agent.state.pos[0]  # shape: (2,)
        theta = agent.state.rot[0]  # shape: (1,)
        vel = action.clone()
        
        vel_norm = action[X]
        omega = action[Y]  
        
        vel[X] = vel_norm * torch.cos(theta)
        vel[Y] = vel_norm * torch.sin(theta)
        
        # Scale Action to deployment environment
        vel[X] = vel[X] * self.x_semidim / self.task_x_semidim
        vel[Y] = vel[Y] * self.y_semidim / self.task_y_semidim

        bounds_min = torch.tensor([-self.x_semidim, -self.y_semidim], device=action.device) + self.agent_radius
        bounds_max = torch.tensor([ self.x_semidim,  self.y_semidim], device=action.device) - self.agent_radius

        next_pos = pos + vel * self.action_dt

        # Compute clamped velocity based on how far the agent can move without crossing bounds
        below_min = next_pos < bounds_min
        above_max = next_pos > bounds_max

        # Adjust velocity where next position would violate bounds
        vel[below_min] = (bounds_min[below_min] - pos[below_min]) / self.action_dt
        vel[above_max] = (bounds_max[above_max] - pos[above_max]) / self.action_dt

        # Clamp to the agent's max velocity norm
        clamped_vel = TorchUtils.clamp_with_norm(vel, agent.v_range)
        return clamped_vel.tolist(), omega.item()

    def _wrap_to_pi(self, angle: float) -> float:
        """Return the equivalent angle in the range [-π, π)."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _issue_commands_to_agents(self, action_tensor):
        real_time_str = datetime.now().isoformat()

        for i, agent in enumerate(self.world.agents):

            cmd_vel, cmd_omega = self.clamp_velocity_to_bounds(action_tensor[i], agent)
            vel_n, vel_e = convert_xy_to_ne(*cmd_vel)

            yaw_now   = agent.state.rot.item()                      # current estimate from localisation
            yaw_ref   = self._wrap_to_pi(yaw_now + cmd_omega * self.action_dt)

            if USING_FREYJA:
                agent.reference_state.vn = vel_n
                agent.reference_state.ve = vel_e
                agent.reference_state.yaw = yaw_ref
                agent.reference_state.an = agent.a_range
                agent.reference_state.ae = agent.a_range
                agent.reference_state.header.stamp = self.get_clock().now().to_msg()
            else:
                agent.reference_state.linear.x = math.sqrt(vel_n**2 + vel_e**2);
                agent.reference_state.angular.z = cmd_omega;

            self.get_logger().info(
                f"Robot {agent.robot_id} - Commanded velocity ne: {vel_n}, {vel_e} - "
                f"Commanded yaw rate: {cmd_omega}"
                f"pos: [{agent.state.pos[0,X]}, {agent.state.pos[0,Y]}] - "
                f"yaw: [{agent.state.rot[0]}] - "
                f"vel: [{agent.state.vel[0,X]}, {agent.state.vel[0,Y]}]"
            )

            self.csv_writer.writerow([
                agent.mytime, real_time_str, agent.robot_id,
                cmd_vel, cmd_omega, vel_n, vel_e,
                agent.state.pos[0,X], agent.state.pos[0,Y],
                agent.state.vel[0,X], agent.state.vel[0,Y]
            ])
            self.log_file.flush()

            agent.pub.publish(agent.reference_state)
            agent.mytime += self.action_dt

    def stop_all_agents(self):
        if getattr(self, "timer", None): self.timer.cancel()
        for agent in self.world.agents:
            if agent.timer is not None:
                agent.timer.cancel()
            agent.state_buffer.clear()
            agent.send_zero_velocity()
    
    def _parse_goal_from_input(self, txt: str) -> torch.Tensor:
        """
        Expected formats:
        •  "3.0 -2.5"
        •  "3.0, -2.5"
        Returns a (1, 2) float32 tensor on the correct device.
        """
        # replace comma with space, split, take first two items
        try:
            n_str, e_str = txt.replace(",", " ").split()[:2]
            goal = torch.tensor([[float(n_str), float(e_str)]],
                                dtype=torch.float32,
                                device=self.device)
            return goal
        except Exception:
            raise ValueError(
                "Invalid goal format. Please enter two numbers, e.g. '1.2 -0.8'"
            )

    def prompt_for_new_task_instruction(self):

        new_sentence = input("Enter a new instruction for the agents: ")
        try:
            embedding = torch.tensor(self.llm.encode([new_sentence]), device=self.device).squeeze(0)
        except Exception as e:
            self.get_logger().error(f"Failed to encode instruction: {e}")
            return
        self.y = embedding

        # Reset step count and timers
        self.step_count = 0
        self.timer = self.create_timer(self.action_dt, self.timer_callback)
        for agent in self.world.agents:
            if agent.timer: agent.timer.cancel(); agent.timer = None
            agent.mytime = 0
            agent.timer = self.create_timer(agent.obs_dt, agent.collect_observation)
        self.get_logger().info("Starting agents with new instruction.")

    def prompt_for_new_speech_task_instruction(self):

        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        self.get_logger().info("Listening for new instruction... (or press Enter to type instead)")

        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print(f"\n Audio recieved, sending to Google Speech API""\n")
            new_sentence = recognizer.recognize_google(audio)
            print(f"\n[Speech Recognized] \"{new_sentence}\"\n")
            self.get_logger().info(f"Received spoken instruction: {new_sentence}")
        except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError) as e:
            self.get_logger().warn(f"Speech recognition failed: {e}")
            user_input = input("Speech recognition failed. Type instruction or press Enter to try again: ").strip()
            if user_input:
                new_sentence = user_input
            else:
                self.get_logger().info("Retrying speech recognition...")
                return self.prompt_for_new_speech_instruction()
        except Exception as e:
            self.get_logger().error(f"Unexpected error during speech recognition: {e}")
            return

        try:
            embedding = torch.tensor(self.llm.encode([new_sentence]), device=self.device).squeeze(0)
        except Exception as e:
            self.get_logger().error(f"Failed to encode instruction: {e}")
            return

        self.y = embedding

        # Reset step count and timers
        self.step_count = 0
        self.timer = self.create_timer(self.action_dt, self.timer_callback)
        for agent in self.world.agents:
            agent.mytime = 0
            if agent.timer: agent.timer.cancel(); agent.timer = None
            agent.timer = self.create_timer(agent.obs_dt, agent.collect_observation)
        self.get_logger().info("Starting agents with new instruction.")

def get_runtime_log_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log_dir = runtime_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

# Manually parse config_path and config_name from CLI
def extract_initial_config():
    config_path, config_name = None, None
    for arg in sys.argv:
        if arg.startswith("config_path="):
            config_path = arg.split("=", 1)[1]
        elif arg.startswith("config_name="):
            config_name = arg.split("=", 1)[1]
    return config_path, config_name

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


# Run script with:
# python deployment/debug/debug_policy/navigation_deployment.py restore_path=/path_to_checkpoint.pt
@hydra.main(version_base=None,config_path="../../conf",config_name="four_flags/deployment/single_agent_navigation")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))   # full merged config
    rclpy.init()

    cfg.experiment.restore_file = cfg.restore_path

    log_dir = get_runtime_log_dir()

    # Instantiate interface
    ros_interface_node = VmasModelsROSInterface(
        config=cfg,
        log_dir=log_dir
    )

    ros_interface_node.prompt_for_new_instruction()

    def sigint_handler(sig, frame):
        ros_interface_node.get_logger().info('SIGINT received. Stopping timer and sending zero velocity...')
        ros_interface_node.stop_all_agents()
        rclpy.spin_once(ros_interface_node, timeout_sec=0.5)
        ros_interface_node.destroy_node()
        ros_interface_node.log_file.close()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    rclpy.spin(ros_interface_node)


if __name__ == '__main__':
    main()
