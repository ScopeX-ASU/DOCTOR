"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:23:19
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-28 22:57:34
"""

from typing import Callable, Dict, Optional

import numpy as np
import torch
from pyutils.general import logger
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn  # , set_deterministic
from torchonn.op.mrr_op import *

from .layers.mrr_conv2d import AddDropMRRBlockConv2d
from .layers.mrr_linear import AddDropMRRBlockLinear

__all__ = ["SparseBP_Base"]


class SparseBP_Base(nn.Module):
    _conv_linear = (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)
    _linear = (AddDropMRRBlockLinear,)
    _conv = (AddDropMRRBlockConv2d,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self, random_state: int = None) -> None:
        for name, m in self.named_modules():
            if isinstance(m, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def backup_phases(self) -> None:
        self.phase_backup = {}
        for layer_name, layer in self.fc_layers.items():
            self.phase_backup[layer_name] = {
                "weight": layer.weight.data.clone()
                if layer.weight is not None
                else None
            }

    def restore_phases(self) -> None:
        for layer_name, layer in self.fc_layers.items():
            backup = self.phase_backup[layer_name]
            for param_name, param_src in backup.items():
                param_dst = getattr(layer, param_name)
                if param_src is not None and param_dst is not None:
                    param_dst.data.copy_(param_src.data)

    def set_phase_variation(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                layer.set_phase_variation(flag)

    def set_global_temp_drift(self, flag: bool = True):
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                layer.set_global_temp_drift(flag)

    def set_crosstalk_noise(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                layer.set_crosstalk_noise(flag)

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                layer.set_weight_noise(noise_std)

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.set_weight_bitwidth(w_bit)

    def get_num_device(self) -> Dict[str, int]:
        total_mrr = 0  # total_mzi = 0
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                total_mrr += layer.in_channel_pad * layer.out_channel_pad
        return {"mrr": total_mrr}

    def load_parameters(self, param_dict: Dict[str, Dict[str, Tensor]]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {layer_name: {param_name: param_tensor, ...}, ...}
        """
        for layer_name, layer_param_dict in param_dict.items():
            self.layers[layer_name].load_parameters(layer_param_dict)

    def build_obj_fn(self, X: Tensor, y: Tensor, criterion: Callable) -> Callable:
        def obj_fn(X_cur=None, y_cur=None, param_dict=None):
            if param_dict is not None:
                self.load_parameters(param_dict)
            if X_cur is None or y_cur is None:
                data, target = X, y
            else:
                data, target = X_cur, y_cur
            pred = self.forward(data)
            return criterion(pred, target)

        return obj_fn

    def enable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.enable_fast_forward()

    def disable_fast_forward(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.disable_fast_forward()

    def sync_parameters(self, src: str = "weight") -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.sync_parameters(src=src)

    def build_weight(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockConv2d, AddDropMRRBlockLinear)):
                layer.build_weight()

    def print_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.print_parameters()

    def gen_mixedtraining_mask(
        self,
        sparsity: float,
        prefer_small: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        return {
            layer_name: layer.gen_mixedtraining_mask(
                sparsity, prefer_small, random_state
            )
            for layer_name, layer in self.named_modules()
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d))
        }

    def switch_mode_to(self, mode: str) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.switch_mode_to(mode)

    def assign_random_phase_bias(self, random_state: Optional[int] = 42) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.assign_random_phase_bias(random_state)

    def clear_phase_bias(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.clear_phase_bias()

    def set_noisy_identity(self, flag: bool = True) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.set_noisy_identity(flag)

    def get_power(self, mixedtraining_mask: Optional[Tensor] = None) -> float:
        power = sum(
            layer.get_power(mixedtraining_mask[layer_name])
            for layer_name, layer in self.fc_layers.items()
        )
        return power

    def gen_deterministic_gradient_mask(
        self, bp_feedback_sparsity: Optional[float] = None
    ) -> None:
        for layer in self.fc_layers.values():
            layer.gen_deterministic_gradient_mask(
                bp_feedback_sparsity=bp_feedback_sparsity
            )

    def gen_uniform_gradient_mask(
        self, bp_feedback_sparsity: Optional[float] = None
    ) -> None:
        for layer in self.fc_layers.values():
            layer.gen_uniform_gradient_mask(bp_feedback_sparsity=bp_feedback_sparsity)

    def set_random_input_sparsity(
        self, bp_input_sparsity: Optional[float] = None
    ) -> None:
        for layer in self.fc_layers.values():
            layer.set_random_input_sparsity(bp_input_sparsity)

    def set_bp_feedback_sampler(
        self,
        forward_sparsity: float,
        backward_sparsity: float,
        alg: str = "topk",
        normalize: bool = False,
        random_state: Optional[int] = None,
    ):
        for layer in self.modules():
            # if(isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))):
            # Linear is not the bottleneck but critical to performance. Recommend not to sample Linear layer!
            if isinstance(layer, (AddDropMRRBlockConv2d,)):
                layer.set_bp_feedback_sampler(
                    forward_sparsity, backward_sparsity, alg, normalize, random_state
                )

    def set_bp_input_sampler(
        self,
        sparsity: float,
        spatial_sparsity: float,
        column_sparsity: float,
        normalize: bool = False,
        random_state: Optional[int] = None,
        sparsify_first_conv: bool = True,
    ):
        counter = 0
        for layer in self.modules():
            if isinstance(layer, AddDropMRRBlockLinear):
                layer.set_bp_input_sampler(sparsity, normalize, random_state)
            elif isinstance(layer, AddDropMRRBlockConv2d):
                if counter == 0:
                    # always donot apply spatial sampling to first conv.
                    # first conv is not memory bottleneck, not runtime bottleneck, but energy bottleneck
                    if sparsify_first_conv:
                        layer.set_bp_input_sampler(
                            0, column_sparsity, normalize, random_state
                        )
                    else:
                        layer.set_bp_input_sampler(0, 0, normalize, random_state)
                    counter += 1
                else:
                    layer.set_bp_input_sampler(
                        spatial_sparsity, column_sparsity, normalize, random_state
                    )

    def set_bp_rank_sampler(
        self,
        bp_rank: int,
        alg: str = "topk",
        sign: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        for layer in self.modules():
            if isinstance(layer, (AddDropMRRBlockLinear, AddDropMRRBlockConv2d)):
                layer.set_bp_rank_sampler(bp_rank, alg, sign, random_state)

    def stack_all_target_dict(self) -> torch.Tensor:
        weight_dict = {}
        weights = []
        for mode in ["conv", "linear"]:
            for layer in self.modules():
                if isinstance(
                    layer,
                    AddDropMRRBlockConv2d if mode == "conv" else AddDropMRRBlockLinear,
                ):
                    weights.append(layer.weight.data.flatten(0, 1))
            weights = torch.cat(weights, dim=0)
            weight_dict[mode] = weights
        return weight_dict

    def set_noise_schedulers(
        self,
        scheduler_dict={
            "phase_variation_scheduler": None,
            "global_temp_scheduler": None,
            "crosstalk_scheduler": None,
        },
    ):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                for scheduler_name, scheduler in scheduler_dict.items():
                    setattr(layer, scheduler_name, scheduler)

        for scheduler_name, scheduler in scheduler_dict.items():
            setattr(self, scheduler_name, scheduler)

    def reset_noise_schedulers(self):
        self.phase_variation_scheduler.reset()
        self.global_temp_scheduler.reset()
        self.crosstalk_scheduler.reset()

    def step_noise_scheduler(self, T=1):
        if self.phase_variation_scheduler is not None:
            for _ in range(T):
                self.phase_variation_scheduler.step()

        if self.global_temp_scheduler is not None:
            for _ in range(T):
                self.global_temp_scheduler.step()

    def backup_ideal_weights(self):
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._ideal_weight = layer.weight.detach().clone()
                # layer._ideal_phase = layer.phase.detach().clone()

    def cycles(self, x_size):
        x = torch.randn(x_size, device=self.device)
        self.eval()

        def hook(m, inp, out):
            m._input_shape = inp[0].shape

        handles = []
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                handle = layer.register_forward_hook(hook)
                handles.append(handle)
        with torch.no_grad():
            self.forward(x)
        cycles = 0
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                cycles += layer.cycles(layer._input_shape, probe=False)
        for handle in handles:
            handle.remove()
        return cycles

    def probe_cycles(self, num_vectors=None):
        cycles = 0
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                cycles += layer.cycles(probe=True, num_vectors=num_vectors)
        return cycles

    def gen_sparsity_mask(self, sparsity=1.0, mode="topk"):
        # top sparsity% will be calibrated
        R, C, k = 4, 4, 8
        self._sparsity = sparsity
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                salience = layer.weight._salience
                P, Q = salience.shape[0:2]
                shape = int(np.ceil(P / R) * R), int(np.ceil(Q / C) * C)
                salience = torch.nn.functional.pad(
                    salience,
                    (0, 0, 0, 0, 0, shape[1] - Q, 0, shape[0] - P),
                    mode="constant",
                    value=0,
                )
                salience = (
                    salience.reshape(shape[0] // R, R, shape[1] // C, C, k, k)
                    .permute(0, 2, 1, 3, 4, 5)
                    .flatten(2)
                    .norm(1, dim=-1)
                )

                if mode == "topk":
                    threshold = torch.quantile(
                        salience.flatten(), q=1 - sparsity, dim=0
                    )
                    mask = salience >= threshold

                elif mode == "IS":
                    mask = torch.zeros_like(salience.flatten())
                    sample_IS = np.random.choice(
                        a=len(list(salience.cpu().detach().numpy().flatten())),
                        size=round(max(1, salience.numel() * sparsity)),
                        replace=False,
                        p=salience.cpu().detach().numpy().flatten()
                        / salience.cpu().detach().numpy().flatten().sum(),
                    )

                    for i in range(len(sample_IS)):
                        mask[sample_IS[i]] = 1
                    mask = mask.view_as(salience)

                elif mode == "uniform":
                    mask = torch.zeros_like(salience.flatten())
                    sample_uni = np.random.choice(
                        a=len(list(salience.cpu().detach().numpy().flatten())),
                        size=round(max(1, salience.numel() * sparsity)),
                        replace=False,
                        p=np.ones_like(salience.cpu().detach().numpy().flatten())
                        / len(salience.cpu().detach().numpy().flatten()),
                    )

                    for i in range(len(sample_uni)):
                        mask[sample_uni[i]] = 1
                    mask = mask.view_as(salience)

                else:
                    raise NotImplementedError

                mask = (
                    mask[:, None, :, None]
                    .repeat(1, R, 1, C)
                    .reshape(shape[0], shape[1])[:P, :Q]
                )
                layer.weight._sparsity_mask = mask
                layer.weight._sparsity = sparsity

    def gen_weight_salience(self, mode="first_grad"):
        assert mode in {"magnitude", "first_grad", "second_grad"}

        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                if mode == "magnitude":
                    layer.weight._salience = layer.weight.abs()
                elif mode == "first_grad":
                    layer.weight._salience = layer.weight._first_grad.abs()
                elif mode == "second_grad":
                    layer.weight._salience = layer.weight._second_grad.abs()
                else:
                    raise ValueError(f"Unknown mode {mode}")

    def map_to_hardware(
        self,
        input_shape,
        lr=1e-2,
        num_steps=100,
        stop_thres=None,
        average_times=5,
        criterion="nmae",
        verbose: bool = True,
        sparsity: float = 1.0,
        sparsity_mode: str = "uniform",
        validation_callback=None,
    ):
        assert criterion in {
            "mse",
            "mae",
            "nmse",
            "nmae",
            "first-order",
            "second-order",
        }
        if criterion == "mse":
            loss_fn = torch.nn.functional.mse_loss
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "mae":
            loss_fn = torch.nn.functional.l1_loss
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "nmse":
            loss_fn = (
                lambda x, target: x.sub(target).square().sum() / target.square().sum()
            )
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "nmae":
            loss_fn = lambda x, target: x.sub(target).norm(p=1) / target.norm(p=1)
            one_prob_cycles = self.probe_cycles() * average_times
        elif criterion == "first-order":
            loss_fn = lambda x, target, grad: (
                grad * (x - target)
            ).sum().abs() + torch.nn.functional.l1_loss(x, target)
            one_prob_cycles = (
                self.probe_cycles(num_vectors=1) + self.probe_cycles() * average_times
            )  # extra 2 cycle
        elif criterion == "second-order":

            def loss_fn(x, target, grad, second_grad):
                error = x - target
                return (
                    (grad * error).sum() + 0.5 * (error.square() * second_grad).sum()
                ).abs() + +torch.nn.functional.l1_loss(x, target)

            one_prob_cycles = (
                self.probe_cycles(num_vectors=2) + self.probe_cycles() * average_times
            )
        one_prob_cycles = int(round(one_prob_cycles * getattr(self, "_sparsity", 1)))
        metric_fn = lambda x, target: x.sub(target).norm(p=1) / target.norm(p=1)

        if verbose:
            logger.info("Mapping ideal weights to noisy hardware")
            logger.info("Backup ideal weights...")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, num_steps, eta_min=0.5 * lr
        )
        one_inference_cycles = self.cycles(input_shape)
        cycle_count = 0
        total_cycle = 0
        total_steps = 0
        if verbose:
            logger.info(
                f"lr: {lr:.2e} average_times: {average_times} #Cycles per inference: {one_inference_cycles:5d}. #Cycles per probe: {one_prob_cycles:5d}."
            )

        for step in range(num_steps):
            loss = []
            metric = []
            self.gen_sparsity_mask(sparsity, sparsity_mode)
            for layer in self.modules():
                if isinstance(layer, self._conv_linear):
                    noisy_weight = (
                        sum(
                            [
                                layer.build_weight(enable_ste=True, enable_remap=True)
                                for _ in range(average_times)
                            ]
                        )
                        / average_times
                    )
                    if hasattr(layer.weight, "_sparsity_mask"):
                        sparsity_mask = (
                            layer.weight._sparsity_mask.flatten().nonzero().flatten()
                        )
                        noisy_weight = noisy_weight.flatten(0, 1)[sparsity_mask]
                        ideal_weight = layer._ideal_weight.flatten(0, 1)[sparsity_mask]

                    else:
                        sparsity_mask = None
                        ideal_weight = layer._ideal_weight

                    if criterion == "first-order":
                        grad = (
                            layer.weight._first_grad
                        )  # need to be preloaded and stored
                        if sparsity_mask is not None:
                            grad = grad.flatten(0, 1)[sparsity_mask]
                        loss.append(loss_fn(noisy_weight, ideal_weight, grad))
                    elif criterion == "second-order":
                        grad = layer.weight._first_grad
                        second_grad = layer.weight._second_grad
                        if sparsity_mask is not None:
                            grad = grad.flatten(0, 1)[sparsity_mask]
                            second_grad = second_grad.flatten(0, 1)[sparsity_mask]
                        loss.append(
                            loss_fn(noisy_weight, ideal_weight, grad, second_grad)
                        )
                    else:
                        loss.append(loss_fn(noisy_weight, ideal_weight))

                    metric.append(
                        metric_fn(noisy_weight.detach(), ideal_weight).cpu().numpy()
                    )

            cycle_count += one_prob_cycles
            total_cycle += one_prob_cycles
            if cycle_count >= one_inference_cycles:
                T = int(cycle_count // one_inference_cycles)
                self.step_noise_scheduler(T)
                total_steps += T
                cycle_count -= T * one_inference_cycles
            loss = sum(loss) / len(loss)
            if validation_callback is not None:
                logger.info(
                    f"Step: {step}, #Cycle: {total_cycle}, {criterion} loss: {loss.item():.2e}"
                )
                validation_callback(step, total_cycle, loss.item())

            if verbose and (step % 5 == 0 or step == num_steps - 1):
                logger.info(
                    f"Step: {step:4d} cycle: {total_cycle:.3e}({cycle_count:4d} / {one_inference_cycles:4d}, {total_steps:3d} noise steps) {criterion} loss: {loss.item():.2e} NMAE: (mean) {np.mean(metric):.2e} (std) {np.std(metric):.2e}"
                )

            if stop_thres is not None and loss.mean() < stop_thres:
                logger.info(
                    f"Break: Step: {step:4d} cycle: {total_cycle:.3e}({cycle_count:4d} / {one_inference_cycles:4d}, {total_steps:3d} noise steps) {criterion} loss: {loss.item():.2e} NMAE: (mean) {np.mean(metric):.2e} (std) {np.std(metric):.2e}"
                )
                break

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

        optimizer.zero_grad()
        self.sync_parameters(src="weight")

        self.global_temp_scheduler.record_current_temp()
        if verbose:
            logger.info(
                f"Finish calibration, current temp: {self.global_temp_scheduler.T} K"
            )
        return total_cycle

    def remap(
        self,
        input_shape,
        flag: bool,
        alg: str = "LAP",
        salience_mode: str = "heuristic",
        average_times: int = 1,
        tolerance: float = 1,
        verbose: bool = True,
        enable_step: bool = True,
    ):
        # perform intra tile remapping in layer here
        # input: layer.weight partitioned into (bs_x, bs_y) batches, with each batch taking (R,C,K,K) weights and remapping them on tile
        # For each (R,C,K,K) weight bank, we only change the R indexs, e.g. (1,2,3,4) -> (3,2,1,4) to minimize the E|W_tilde - layer.weight|
        total_cycles = 0
        cycle_count = 0
        one_inference_cycles = self.cycles(input_shape)
        if flag:
            for layer in self.modules():
                if isinstance(layer, self._conv_linear):
                    _, _, cycles = layer.remap_intra_tile(
                        alg=alg,
                        salience_mode=salience_mode,
                        average_times=average_times,
                        tolerance=tolerance,
                    )
                    cycle_count += cycles
                    total_cycles += cycles
                    if cycle_count >= one_inference_cycles:
                        T = int(cycle_count // one_inference_cycles)
                        if enable_step:
                            self.step_noise_scheduler(T)
                        cycle_count -= T * one_inference_cycles

            self.global_temp_scheduler.record_current_temp()
            if verbose:
                logger.info(
                    f"Finish remapping, total cycles: {total_cycles}, current temp: {self.global_temp_scheduler.T} K"
                )

        return total_cycles

    def is_map_from_temp(self) -> bool:
        is_remap = False
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                temp_drift = (
                    layer.global_temp_scheduler.get_global_temp()
                    - layer.global_temp_scheduler.T0
                )
                if temp_drift > 0.2:
                    is_remap = True
        return is_remap

    def probe_weight_error(self) -> Tensor:
        W_til, W = [], []
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, self._conv_linear):
                    W_til.append(
                        layer.build_weight(enable_ste=True, enable_remap=True)
                        .detach()
                        .flatten()
                    )
                    W.append(layer._ideal_weight.flatten())
            W_til = torch.cat(W_til)
            W = torch.cat(W)
            error = W_til.sub(W).norm(p=1) / W.norm(p=1)
        return error.item()

    def set_enable_ste(self, enable_ste: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._enable_ste = enable_ste

    def set_noise_flag(self, noise_flag: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer._noise_flag = noise_flag

    def set_enable_remap(self, enable_remap: bool) -> None:
        for layer in self.modules():
            if isinstance(layer, self._conv_linear):
                layer.set_enable_remap(enable_remap)

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
