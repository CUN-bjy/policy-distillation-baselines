import torch
import numpy as np

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros


def to_device(device, *args):
    return [x.to(device) for x in args]


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                grads.append(zeros(param.view(-1).shape))
            else:
                grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def compute_flat_grad(output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(zeros(param.view(-1).shape, device=param.device, dtype=param.dtype))
        else:
            out_grads.append(grads[j].view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads


def flat(grads):
    flat_grads = []
    for grad in grads:
        flat_grads.append(grad.view(-1))
    flat_grads = torch.cat(flat_grads)
    return flat_grads


def get_update_direction_with_lo(grad_flat, current_net, lo):
    directions = []
    prev_ind = 0
    for param in current_net.parameters():
        flat_size = int(np.prod(list(param.size())))
        ndarray = grad_flat[prev_ind:prev_ind + flat_size].view(param.size()).detach()
        if ndarray.dim() > 1:  # inter-layer parameters
            ndarray = ndarray.numpy()
            direction_layer = lo.lo_oracle(-ndarray)
            direction_layer = torch.from_numpy(direction_layer).view(-1)
            direction_layer = param.view(-1) - direction_layer.double()
        else:  # parameters of activation functions
            direction_layer = ndarray
        directions.append(direction_layer)
        prev_ind += flat_size
    direction = torch.cat(directions)
    # print(torch.norm(-grad_flat - direction))
    return direction

def get_update_direction_with_lo2(grad_flat, net, lo, cur_params):
    directions = []
    prev_ind = 0
    for param in net.parameters():
        flat_size = int(np.prod(list(param.size())))
        ndarray = grad_flat[prev_ind:prev_ind + flat_size].view(param.size()).detach()
        cur_param = cur_params[prev_ind:prev_ind + flat_size]
        if ndarray.dim() > 1:  # inter-layer parameters
            ndarray = ndarray.numpy()
            direction_layer = lo.lo_oracle(-ndarray)
            direction_layer = torch.from_numpy(direction_layer).view(-1)
            direction_layer = cur_param - direction_layer.double()
        else:  # parameters of activation functions
            direction_layer = ndarray
        directions.append(direction_layer)
        prev_ind += flat_size
    direction = torch.cat(directions)
    # print(torch.norm(-grad_flat - direction))
    return direction