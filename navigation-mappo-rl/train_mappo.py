from nav.environment import Environment
import gymnasium as gym
import yaml
import supersuit as ss
import os
from networks.actor_critic_network import ObservationEncoder
from rl.mappo import MAPPO
import sys

model_id = sys.argv[1]
config_file = sys.argv[2]

history_length = 4


os.makedirs(f"models/{model_id}", exist_ok=True)
os.makedirs(f"models/{model_id}/best_model", exist_ok=True)
os.makedirs(f"models/{model_id}/checkpoints", exist_ok=True)
os.makedirs(f"videos/{model_id}", exist_ok=True)
os.makedirs(f"logs/{model_id}", exist_ok=True)


config = yaml.safe_load(open(config_file))
env = Environment(config)
config = env.config.model_dump()  # get the config along with defaults
lidar_dim = env.config.num_rays
agent_states_dim = env.agent_states_dim

policy_kwargs = dict(
    features_extractor_class=ObservationEncoder,
    features_extractor_kwargs=dict(
        agent_states_dim=agent_states_dim,
        lidar_dim=lidar_dim,
        history_length=history_length,
        objects=3,
    ),
    state_features_output_dim=384,
)

n_agents = env.n_agents
env = ss.frame_stack_v1(env, history_length)
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8)
env.n_agents = n_agents
print(env.n_agents)

load_model = False
if os.path.exists(f"models/{model_id}/best_model/model.pth"):
    answer = input("Best model found, do you want to continue training? (y/n)")
    if answer.lower() == "y":
        load_model = True

model = MAPPO(
    env,
    eval_config=config,
    history_length=history_length,
    batch_size=128,
    policy_kwargs=policy_kwargs,
    learning_rate=5e-4,
    inference_interval=5,
    n_epochs=2,
    ent_coef=0.0,
    buffer_size=256,
    model_dir=f"models/{model_id}",
    video_dir=f"videos/{model_id}",
)

if load_model:
    model.load_model(f"models/{model_id}/best_model")

model.learn(total_timesteps=1e7)
