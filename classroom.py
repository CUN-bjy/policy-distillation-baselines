# original : https://github.com/DLR-RM/rl-baselines3-zoo

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import os, sys

import torch as th
import yaml
from stable_baselines3.common.utils import set_random_seed

import utils3.import_envs  # noqa: F401 pylint: disable=unused-import
from utils3 import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils2.replay_memory import Memory

################################################################
# to clear cv2 Import error
ros_pack_path = '/opt/ros/%s/lib/python2.7/dist-packages'%os.getenv('ROS_DISTRO')
if ros_pack_path in sys.path: sys.path.remove(ros_pack_path)
################################################################


def load_env_and_model(env_id,algo,folder):
    # get experiment id
    exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
    print(f"Loading latest experiment, id={exp_id}")

    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    # check & take get the model_path
    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if not found:
        raise ValueError(f"No model found for {algo} on {env_id}, path: {model_path}")

    # set random seed
    set_random_seed(0)

    # get stats_path & hyperparam_path
    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    # make gym environment
    env = create_test_env(
        env_id,
        n_envs=1,
        stats_path=stats_path,
        seed=0,
        log_dir=None,
        should_render=True,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    # Dummy buffer size as we don't need memory to enjoy the trained agent
    kwargs = dict(seed=0)
    kwargs.update(dict(buffer_size=1))

    # load pre-trained model
    model = ALGOS[algo].load(model_path, env=env, **kwargs)


    return env, model


def sample_generator(env, model, render=True, min_batch_size=10000,id_=0):
    log = dict()
    memory = Memory()

    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    num_episodes = 0
  
    # main loop to enjoy for n_timesteps..
    try:
        while num_steps < min_batch_size: 
            try:
                obs = env.reset()
                state = None
                episode_reward = 0.0
                episode_rewards, episode_lengths = [], []
                ep_len = 0

                for t in range(1000):
                    action, state = model.predict(obs, state=state, deterministic=True)
                    next_state, reward, done, _ = env.step(action)

                    episode_reward += reward[0]
                    ep_len += 1

                    if done:
                        # NOTE: for env using VecNormalize, the mean reward
                        # is a normalized reward when `--norm_reward` flag is passed
                        print(f"Episode Reward: {episode_reward:.2f}")
                        print("Episode Length", ep_len)
                        episode_rewards.append(episode_reward)
                        episode_lengths.append(ep_len)
                        episode_reward = 0.0
                        ep_len = 0
                        state = None

                    mask = 0 if done else 1
                    memory.push(obs, action, mask, next_state, reward)
                    obs = next_state

                    if render: 
                        env.render("human")
                    if done:
                        break
                # log stats
                num_steps += (t + 1)
                num_episodes += 1
                total_reward = sum(episode_rewards)
                min_reward = min(episode_rewards)
                max_reward = max(episode_rewards)
            except Exception as e:
                print(e)
    except KeyboardInterrupt:
        pass
    env.close()

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward

    return id_, memory, log

if __name__ == "__main__":

    # key parameters
    env_id = 'AntBulletEnv-v0'
    algo = 'td3'
    folder = "rl-trained-agents"
    
    env, teacher = load_env_and_model(env_id, algo, folder)
    sample_generator(env, teacher)