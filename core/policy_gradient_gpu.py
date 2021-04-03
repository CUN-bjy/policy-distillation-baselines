from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import torch
from torch.autograd import Variable
import scipy
import ray


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

@ray.remote(num_gpus=1)
def compute_PG(pid, policy_net, value_net, states, actions, returns, advantages):
    torch.tensor(1).to('cuda')

    """compute policy gradient and update value net by using samples in memory"""
    for param in policy_net.parameters():
        param.requires_grad = True
    for param in value_net.parameters():
        param.requires_grad = True

    # Original code uses the same LBFGS to optimize the value loss

    def get_value_loss(targets):
        def _get_value_loss(flat_params):
            vector_to_parameters(torch.Tensor(flat_params), value_net.parameters())
            for param in value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = value_net(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in value_net.parameters():
                value_loss += param.pow(2).sum() * 1e-3
            grads = torch.autograd.grad(value_loss, value_net.parameters())
            return value_loss.data.double().numpy(), parameters_to_vector(grads).double().numpy()

        return _get_value_loss

    # def get_value_grad(flat_params):
    value_net_curr_params = get_flat_params_from(value_net).double().numpy()

    value_net_update_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss(returns), value_net_curr_params, maxiter=25)
    # vector_to_parameters(torch.Tensor(value_net_update_params), value_net.parameters())

    log_probs = policy_net.get_log_prob(states.double(), actions)
    loss = -(advantages * torch.exp(log_probs - log_probs.detach())).mean()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = parameters_to_vector(grads)

    return pid, loss_grad, value_net_update_params


def compute_policy_gradient_parallel(policy_net, value_net, states_list, actions_list, returns_list, advantages_list):
    result_ids = []
    num_agents = len(advantages_list)
    for advantages, returns, states, actions, pid in zip(advantages_list, returns_list, states_list, actions_list, range(num_agents)):
        result_id = compute_PG.remote(pid, policy_net, value_net, states.float(), actions, returns.float(), advantages)
        result_ids.append(result_id)

    policy_gradients = [None]*num_agents
    value_net_update_params = [None]*num_agents

    for result_id in result_ids:
        pid, policy_gradient, value_net_update_param = ray.get(result_id)
        policy_gradients[pid] = policy_gradient.numpy()
        value_net_update_params[pid] = value_net_update_param

    return policy_gradients, value_net_update_params

def compute_policy_gradient_parallel_noniid(policy_net, value_nets_list, states_list, actions_list, returns_list, advantages_list):
    result_ids = []
    num_agents = len(advantages_list)
    for value_net, advantages, returns, states, actions, pid in zip(value_nets_list, advantages_list, returns_list, states_list, actions_list, range(num_agents)):
        result_id = compute_PG.remote(pid, policy_net, value_net, states.float(), actions, returns.float(), advantages)
        result_ids.append(result_id)

    policy_gradients = [None]*num_agents
    value_net_update_params = [None]*num_agents

    for result_id in result_ids:
        pid, policy_gradient, value_net_update_param = ray.get(result_id)
        policy_gradients[pid] = policy_gradient.numpy()
        value_net_update_params[pid] = value_net_update_param

    return policy_gradients, value_net_update_params