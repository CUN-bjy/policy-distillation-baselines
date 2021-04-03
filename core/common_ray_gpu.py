import torch
from utils2.torch import to_device
import numpy as np
import multiprocessing as mp
import ray

@ray.remote(num_gpus=1)
def estimate_advantages(memory, value_net, gamma, tau, device='cpu', dtype=torch.double, pid=None):
    torch.tensor(1).to('cuda')

    batch = memory.sample()
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)

    # with torch.no_grad():
    values = value_net(states).detach()
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = torch.zeros((rewards.size(0), 1), dtype=dtype)
    # advantages = tensor_type(rewards.size(0), 1)
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns, states, actions = to_device(device, advantages, returns, states, actions)

    return (pid, advantages, returns, states, actions)


def estimate_advantages_parallel(memories, value_net, gamma, tau, device='cpu', dtype=torch.float64, num_parallel_workers=mp.cpu_count()):
    # ray.init()
    result_ids = []
    for memory, pid in zip(memories, range(len(memories))):
        result_ids.append(estimate_advantages.remote(memory,
            value_net, gamma, tau, device, dtype, pid))

    advantages_list = [None]*len(memories)
    returns_list = [None]*len(memories)
    states_list = [None] * len(memories)
    actions_list = [None] * len(memories)
    for result_id in result_ids:
        pid, advantages, returns, states, actions = ray.get(result_id)
        advantages_list[pid] = advantages
        returns_list[pid] = returns
        states_list[pid] = states
        actions_list[pid] = actions

    return advantages_list, returns_list, states_list, actions_list

def estimate_advantages_parallel_noniid(memories, value_nets_list, gamma, tau, device='cpu', dtype=torch.float64, num_parallel_workers=mp.cpu_count()):
    # ray.init()
    result_ids = []
    for memory, value_net, pid in zip(memories, value_nets_list, range(len(memories))):
        result_ids.append(estimate_advantages.remote(memory,
            value_net, gamma, tau, device, dtype, pid))

    advantages_list = [None]*len(memories)
    returns_list = [None]*len(memories)
    states_list = [None] * len(memories)
    actions_list = [None] * len(memories)
    for result_id in result_ids:
        pid, advantages, returns, states, actions = ray.get(result_id)
        advantages_list[pid] = advantages
        returns_list[pid] = returns
        states_list[pid] = states
        actions_list[pid] = actions

    return advantages_list, returns_list, states_list, actions_list

