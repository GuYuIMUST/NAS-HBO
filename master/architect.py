import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import utils

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.objs_search = utils.AvgrageMeter()
        self.top1_search = utils.AvgrageMeter()

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
            self.optimizer.step()
        else:
            loss, loss1, loss2 = self._backward_step(input_valid, target_valid)
            self.optimizer.step()
            return loss, loss1, loss2

    def _backward_step(self, input_valid, target_valid):
        flatten_tensor1 = torch.flatten(torch.sigmoid(self.model.alphas_normal))
        flatten_tensor2 = torch.flatten(torch.sigmoid(self.model.alphas_reduce))
        flatten_tensor3 = torch.flatten(torch.sigmoid(self.model.betas_normal))
        flatten_tensor4 = torch.flatten(torch.sigmoid(self.model.betas_reduce))
        aux_input = torch.cat([flatten_tensor1, flatten_tensor2, flatten_tensor3, flatten_tensor4], dim=0)

        loss, loss1, loss2 = self.model._loss2(input_valid, target_valid, aux_input)
        loss.backward()
        return loss, loss1, loss2

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)  # guyu momtum

        flatten_tensor1 = torch.flatten(torch.sigmoid(unrolled_model.alphas_normal))
        flatten_tensor2 = torch.flatten(torch.sigmoid(unrolled_model.alphas_reduce))
        flatten_tensor3 = torch.flatten(torch.sigmoid(unrolled_model.betas_normal))
        flatten_tensor4 = torch.flatten(torch.sigmoid(unrolled_model.betas_reduce))
        aux_input = torch.cat([flatten_tensor1, flatten_tensor2, flatten_tensor3, flatten_tensor4], dim=0)
        unrolled_loss, loss1, loss2 = unrolled_model._loss2(input_valid, target_valid, aux_input)
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(v, alpha=R)
        R = R.to('cpu')
        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
