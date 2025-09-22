# config_loader_condensed.py
from collections.abc import Mapping
from vmas.simulator.utils import ScenarioUtils  # kept in case you use it elsewhere

# --------------------------------------------------------------------------- #
# 1. PARAMS table (updated for FourRoomsSwitchesScenario)
# --------------------------------------------------------------------------- #
PARAMS = [
    # --- Arena --------------------------------------------------------------
    ("x_semidim", 1.0), ("y_semidim", 0.5),

    # --- Agents -------------------------------------------------------------
    ("n_agents", 3), ("agent_radius", 0.025), ("agent_spacing", 0.2),

    # --- Rooms / Gates / Targets -------------------------------------------
    ("gate_thickness", 0.075),
    ("switch_radius", 0.125),
    ("goal_radius", None),           # if None, scenario will default to agent_radius
    ("require_all_to_finish", True),

    # --- Rewards ------------------------------------------------------------
    ("shaping_factor", 100.0), ("collision_penalty", 0.0),
    ("edge_eps", 0.0), ("edge_penalty", 1.0),
    ("min_separation", 0.8),

    # --- Dynamics -----------------------------------------------------------
    ("use_velocity_controller", True), ("use_kinematic_model", False),
    ("agent_weight", 1.0), ("agent_v_range", 1.0), ("agent_a_range", 1.0),
    ("min_collision_distance", 0.1), ("linear_friction", 0.1),

    # --- Communication / GNN -----------------------------------------------
    ("use_gnn", False), ("comm_dim", 1),
    ("comms_radius", 1.0),

    # --- Language -----------------------------------------------------------
    ("embedding_size", 1024), ("use_embedding_ratio", 1.0),
    ("event_dim", 3), ("state_dim", 5),

    # --- Paths --------------------------------------------------------------
    ("data_json_path", "sequence_models/data/four_rooms/dataset_four_rooms.json"),
    ("decoder_model_path", ""),  # harmless if unused
    ("sequence_model_path", "sequence_models/four_rooms/four_rooms_language.pth"),
    ("policy_config_path", "../../conf"),
    ("policy_config_name", "config_stage_two.yaml"),
    ("policy_restore_path", "checkpoints/four_rooms/four_rooms_multitask/policy.pt"),
    
    # --- Evaluation Configs --------------------------------------------------
    ("initial_room", None), ("initialized_rnn", False), ("even_distribution", False),
]

# Expand every entry to canonical (dest, key, default) form
PARAMS = [
    (dest, key, default)
    for entry in PARAMS
    for dest, key, default in [(
        entry if isinstance(entry, tuple) and len(entry) == 3 else
        (entry[0], entry[0], entry[1]) if isinstance(entry, tuple) else
        (entry, entry, None)
    )]
]

# --------------------------------------------------------------------------- #
# 2. Generic loader
# --------------------------------------------------------------------------- #
def load_scenario_config(source, env):
    """
    Parameters
    ----------
    source : dict-like (kwargs) | object with attributes
    env    : your VMAS scenario environment
    """
    is_mapping = isinstance(source, Mapping)

    for dest_attr, key, default in PARAMS:
        if is_mapping:
            value = source.get(key, default)
        else:
            value = getattr(source, key, default)
        setattr(env, dest_attr, value)

    # --- derived attributes -------------------------------------------------
    # No passage_width or chamber geometry in the new scenario.
    env.agent_f_range = env.agent_a_range + env.linear_friction
    env.agent_u_range = (env.agent_v_range if env.use_velocity_controller else env.agent_f_range)
    env.viewer_zoom = 1
