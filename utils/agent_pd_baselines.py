# reference : https://github.com/Mee321/policy-distillation & https://github.com/DLR-RM/rl-baselines3-zoo

# ---------------------------------------------
# Autor : Junyeob Baek, wnsdlqjtm@naver.com
# ---------------------------------------------

import os, yaml, sys

from utils.replay_memory import Memory
from utils.torch import *

from stable_baselines3.common.utils import set_random_seed

import utils2.import_envs  # noqa: F401 pylint: disable=unused-import
from utils2 import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams


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

        episode_rewards, episode_lengths = [], []
        while num_steps < min_batch_size: 
            obs = env.reset()
            state = None
            episode_reward = 0.0
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
                    env.render()
                if done:
                    break
            # log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward = sum(episode_rewards)
            min_reward = min(episode_rewards)
            max_reward = max(episode_rewards)
    except KeyboardInterrupt:
        pass
    # env.close()
    
    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward

    return id_, memory, log

class AgentCollection:

    def __init__(self, envs, policies, device, mean_action=False, render=False,  num_agents=1):
        self.envs = envs
        self.policies = policies
        self.device = device
        self.mean_action = mean_action
        self.render = render
        self.num_agents = num_agents
        self.num_teachers = len(policies)

    def collect_samples(self, min_batch_size, exercise=False):
        # print("collect_samples called!!")
        results = []
        for i in range(self.num_teachers):
            if not exercise:
                results.append(sample_generator(self.envs[i], self.policies[i], self.render, min_batch_size, i))
            else:
                results.append(self.exercise(self.envs[i], self.policies[i], self.render, min_batch_size, i))
        worker_logs = [None] * self.num_agents
        worker_memories = [None] * self.num_agents
        # print(len(result_ids))
        for result in results:
            pid, worker_memory, worker_log = result
            worker_memories[pid] = worker_memory
            worker_logs[pid] = worker_log

        # print("collect_samples done")
        return worker_memories, worker_logs

    def get_expert_sample(self, batch_size, deterministic=True):
        # print("get_expert_sample called!!")
        memories, logs = self.collect_samples(batch_size)
        teacher_rewards = [log['avg_reward'] for log in logs if log is not None]
        teacher_average_reward = np.array(teacher_rewards).mean()
        # TODO better implementation of dataset and sampling
        # construct training dataset containing pairs {X:state, Y:output of teacher policy}
        dataset = []
        for memory, policy in zip(memories, self.policies):
            batch = memory.sample()
            batched_state = np.array(batch.state).reshape(-1, policy.env.observation_space.shape[0])
            states = torch.from_numpy(batched_state).to(torch.float).to('cpu')
            act_dist = torch.from_numpy(policy.predict(states, deterministic=deterministic)[0])
            dataset += [(state, act_dist) for state, act_dist in zip(states, act_dist)]
        return dataset, teacher_average_reward

    def exercise(self, env, policy, render=True, min_batch_size=10000, pid=0):
        torch.randn(pid)
        log = dict()
        memory = Memory()
        num_steps = 0
        total_reward = 0
        min_reward = 1e6
        max_reward = -1e6
        num_episodes = 0

        while num_steps < min_batch_size:
            state = env.reset()
            reward_episode = 0

            for t in range(1000):
                state_var = tensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = policy.mean_action(state_var.to(torch.float))[0].numpy()
                next_state, reward, done, _ = env.step(action)
                reward_episode += reward

                mask = 0 if done else 1

                memory.push(state, action, mask, next_state, reward)

                if render:
                    env.render()
                if done:
                    break

                state = next_state

            # log states
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)
            print("reward_episode: %f"%reward_episode)
            print("num_steps: %d"%num_steps)


        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward

        return (pid, memory, log)