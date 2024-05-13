import math
import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.sgd import SGD

from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
                        _differentiable_doc, _foreach_doc, _maximize_doc)

from main import ANFIS



class FuzzySGD(SGD):
    def __init__(self, anfis, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize: bool = False, foreach: Optional[bool] = None,
                 differentiable: bool = False):
        
        if not isinstance(anfis, ANFIS):
            raise ValueError(f"You need to supply the fuzzy optimizer with an ANFIS system, but you provided {type(anfis)}.")
        
        self.anfis = anfis

        defaults = dict(lr=lr, momentum=momentum,
                        dampening=dampening, weight_decay=weight_decay,
                        nesterov=nesterov, maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
    
        super().__init__(anfis.parameters(), **defaults)

    def sgd(self,
        params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
        r"""Functional API that performs SGD algorithm computation.

        See :class:`~torch.optim.SGD` for details.
        """

        if foreach is None:
            # why must we be explicit about an if statement for torch.jit.is_scripting here?
            # because JIT can't handle Optionals nor fancy conditionals when scripting
            if not torch.jit.is_scripting():
                _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
            else:
                foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        if foreach and not torch.jit.is_scripting():
            func = self._multi_tensor_sgd
        else:
            func = self._single_tensor_sgd

        func(params,
            d_p_list,
            momentum_buffer_list,
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            has_sparse_grad=has_sparse_grad,
            maximize=maximize)
    
    
    def get_conversion_number(self, x):
        if x == 0:
            return 1.0
        
        x = math.floor(math.log10(abs(x)))
        if x > 0:
            return float("1" + "0" * x)
        else:
            return float("0." + "0" * abs(x) + "1")
        
        
    def _single_tensor_sgd(self,
                        params: List[Tensor],
                        d_p_list: List[Tensor],
                        momentum_buffer_list: List[Optional[Tensor]],
                        *,
                        weight_decay: float,
                        momentum: float,
                        lr: float,
                        dampening: float,
                        nesterov: bool,
                        maximize: bool,
                        has_sparse_grad: bool):
        print("--------------------------------------")
        for i, param in enumerate(params):

            range_universe = None
            for _, universe in self.anfis.inputs.items():
                for _, function in universe.functions.items():
                    for param_a in function.parameters():
                        if param is param_a:
                            range_universe = (universe.max - universe.min) / universe.max

            if range_universe is None and self.anfis.system_type != "Takagi-Sugeno":
                for _, universe in self.anfis.outputs.items():
                    for _, function in universe.functions.items():
                        for param_c in function.parameters():
                            if param is param_c:
                                range_universe = (universe.max - universe.min) / universe.max

            d_p = d_p_list[i] if not maximize else -d_p_list[i]

            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            print("1", d_p.item())
            d_p_cn = self.get_conversion_number(d_p) * 10
            range_universe_cn = self.get_conversion_number(range_universe)
            print("RANGO", range_universe_cn)

            d_p = (range_universe_cn / d_p_cn) * d_p

            print("d_p: ", d_p.item())
            print("2", d_p.item()*lr)
            print("PARAM", param.item())
            print(" ")

            param.add_(d_p, alpha=lr)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            self.sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
    





    