"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:35:39
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-04 16:30:46
"""

import os
import random
import sys
from functools import lru_cache
from typing import Callable, List, Optional

import einops
import numpy as np
import torch
import tqdm
from pyutils.compute import (
    gen_gaussian_filter2d,
    merge_chunks,
    partition_chunks,
)
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor
from torch.types import _size

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))


__all__ = [
    "DeterministicCtx",
    "PhaseVariationScheduler",
    "GlobalTemperatureScheduler",
    "CrosstalkScheduler",
    "calculate_grad_hessian",
]


def apply_remap_weight(self, weight, col_ind, require_size=[4, 4, 8, 8]):
    weight = self.layer_weight_partition_chunk(
        weight, require_size=require_size
    )  # [b0, b1, R, C, K, K]
    # print(self.weight.shape, weight.shape, self.col_ind.shape)
    weight = weight.flatten(0, 1)[
        torch.arange(weight.shape[0] * weight.shape[1])[..., None],
        col_ind.flatten(0, 1),
    ].reshape(weight.shape)
    weight = self.layer_weight_merge_chunk(weight)[
        : self.grid_dim_y, : self.grid_dim_x
    ]  # [P,Q,K,K]
    return weight


def unapply_remap_weight(self, weight, col_ind, require_size=[4, 4, 8, 8]):
    weight = self.layer_weight_partition_chunk(
        weight, require_size=require_size
    )  # [b0, b1, R, C, K, K]
    weight.flatten(0, 1)[
        torch.arange(weight.shape[0] * weight.shape[1])[..., None],
        col_ind.flatten(0, 1),
    ] = weight.flatten(0, 1).clone()
    weight = self.layer_weight_merge_chunk(weight)[
        : self.grid_dim_y, : self.grid_dim_x
    ]  # [P,Q,K,K]
    return weight


def apply_remap_noise(noise_map, col_ind):
    ## noise_map: [b0, b1, R, C, K, K]
    ## col_ind: [b0, b1, R]
    ## [0, 1, 1, 2] means W0 -> T0, W1 -> T1, W2 -> T1, W3 -> T2
    noise_map = noise_map.flatten(0, 1)[
        torch.arange(noise_map.shape[0] * noise_map.shape[1])[..., None],
        col_ind.flatten(0, 1),
    ].reshape(noise_map.shape)

    return noise_map  # [b0, b1 ,R, C, K, K]


class PhaseVariationScheduler(object):
    def __init__(
        self,
        size: _size = [
            4,
            4,
            8,
            8,
        ],  # this one should be the architecture dimension, [R, C, K, K], not workload dimension [P, Q, K, K]
        T_max: int = 1000,  # total number of steps
        mean_schedule_fn: Callable = lambda: 0.02,  # a function that returns a mean value for a given step
        std_schedule_fn: Callable = lambda: 0.01,  # a function that returns a std value for a given step
        smoothing_kernel_size: int = 5,  # kernel size for the gaussian filter
        smoothing_factor: float = 0.05,  # how smooth is the distribution
        smoothing_mode: str = "core",  # smoothing mode, core: smooth core-by-core, arch: smooth over all cores
        min_std: float = 0.001,
        momentum: float = 0.9,  # how much is on previous noise std distribution, momenutm * old_map + (1-momentum) * new_map
        noise_scenario_src: str = "",  # corner, edge cases
        noise_scenario_tgt: str = "",
        random_state: int = 0,
        device="cuda:0",
    ) -> None:
        """
        Each device has a zero-mean random phase noise, the phase noise follows N(0, std_i^2) for the i-th device
        Then we need a tensor `noise_std_map` with the same shape as `phase` for each device.
        The noise intensity for each device will gradually drift to an unknown direction.
        To create a random std drift curve, e.g., std_i=0.01 -> std_i=0.008 -> std_i=0.012 -> std_i=0.011 -> std_i=0.009
        we construct a random process, std_i = momentum * std_i_old + (1 - momentum) * std_i_new
        , where std_i_new is randomly sampled from a Gaussian distribution N(std_mean_i, std_std_i),
        std_i_new are spatially smooth across all devices, therefore we apply gaussian filter to smooth `noise_std_map`.
        std_mean_i is controlled by mean_schedule_fn, std_std_i is controlled by std_schedule_fn.
        For example, if std_mean increases, it means the average noise intensity increases across all devices. Maybe the environment gets worse or background noises become larger.
        For example, if std_std increases, it means the noise intensity becomes more diverse across all devices. Maybe there is some local perturbation that makes devices behave diversely.

        """
        # std of the phase noise follows Gaussian distribution ~ N(noise_std_mean, noise_std_std^2)
        super().__init__()
        self.size = size
        self.T_max = T_max
        self.mean_schedule_fn = mean_schedule_fn
        self.std_schedule_fn = std_schedule_fn
        self.smoothing_kernel_size = smoothing_kernel_size
        assert (
            smoothing_kernel_size == 0 or smoothing_kernel_size % 2 == 1
        ), "Must have 0 or odd size of kernel"
        self.smoothing_factor = smoothing_factor
        self.smoothing_mode = smoothing_mode
        self.momentum = momentum
        self.min_std = min_std
        self.noise_scenario_src = noise_scenario_src
        self.noise_scenario_tgt = noise_scenario_tgt

        self.random_state = random_state
        self.device = device
        self.core_noise_mean_map = None

        if self.smoothing_factor > 0 and self.smoothing_kernel_size > 0:
            self.gaussian_filter = gen_gaussian_filter2d(
                self.smoothing_kernel_size,
                std=self.smoothing_factor,
                center_one=False,
                device=self.device,
            )[None, None, ...].to(device)
            # print(self.gaussian_filter)
            # exit(0)
            pad = self.smoothing_kernel_size // 2
            self.padder = torch.nn.ReflectionPad2d((pad, pad, pad, pad))
        else:
            self.gaussian_filter = None
        self.noises = None

        self.reset()

    def reset(self):
        self._step = 0
        self.noise_std_mean = self.mean_schedule_fn(
            0
        )  # the mean of the phase noise std
        self.noise_std_std = self.std_schedule_fn(0)  # the std of the phase noise std
        self.noise_std_map = None
        self.noises = None
        self.noise_scenario_transition()
        self.update_noise_std_map()

    def step(self):
        # one time step to change noise distribution
        self._step += 1  # enable periodic scheduling
        self.noise_std_mean = self.mean_schedule_fn(
            (self._step % self.T_max) / self.T_max  # enable periodic scheduling
        )  # normalized value to query the mean schedule function
        self.noise_std_std = self.std_schedule_fn(
            (self._step % self.T_max) / self.T_max  # enable periodic scheduling
        )  # normalized value to query the std schedule function
        self.update_noise_std_map()
        self.noise_scenario_transition()

    def noise_scenario_transition(self):
        if self.noise_scenario_tgt == "edge":
            target_core_noise_mean_map = torch.tensor(
                [
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                    [0.008, 0.006, 0.004, 0.002],
                ],
                device=self.device,
            )
        elif self.noise_scenario_tgt == "corner":
            target_core_noise_mean_map = torch.tensor(
                [
                    [0.008, 0.006, 0.006, 0.004],
                    [0.006, 0.006, 0.004, 0.004],
                    [0.006, 0.004, 0.004, 0.002],
                    [0.004, 0.004, 0.002, 0.002],
                ],
                device=self.device,
            )
        elif self.noise_scenario_tgt == "upper_edge":
            target_core_noise_mean_map = torch.tensor(
                [
                    [4, 4, 4, 4],
                    [0.4, 0.4, 0.4, 0.4],
                    [0.04, 0.04, 0.04, 0.04],
                    [0.004, 0.004, 0.004, 0.004],
                ],
                device=self.device,
            )

        core_noise_mean_map = self._generate_core_noise_mean_map()
        if self.core_noise_mean_map is None:
            self.core_noise_mean_map = core_noise_mean_map
        else:
            self.core_noise_mean_map = (
                self.momentum * self.core_noise_mean_map
                + (1 - self.momentum) * target_core_noise_mean_map
            )

    def _generate_core_noise_mean_map(self) -> Tensor:
        core_noise_mean_map = torch.zeros(self.size[:-2])
        if self.noise_scenario_src == "corner":
            self.core_noise_mean_map = torch.tensor(
                [
                    [0.0008, 0.0006, 0.0006, 0.0004],
                    [0.0006, 0.0006, 0.0004, 0.0004],
                    [0.0006, 0.0004, 0.0004, 0.0002],
                    [0.0004, 0.0004, 0.0002, 0.0002],
                ]
            )
        elif self.noise_scenario_src == "edge":
            self.core_noise_mean_map = torch.tensor(
                [
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                    [0.0008, 0.0006, 0.0004, 0.0002],
                ]
            )
        else:
            raise NotImplementedError

        core_noise_mean_map = self.core_noise_mean_map / 2
        self.core_noise_mean_map = core_noise_mean_map.to(self.device)
        return core_noise_mean_map.to(self.device)

    def _generate_noise_std_map(self):
        # determinstic, do not affect external random state, different across steps
        # this is device-wise noise std map. Each MRR has a noise_std, representing its noise intensity
        # this can create difference within each core for intra-core remapping
        noise_std_map = torch.normal(
            self.noise_std_mean,
            self.noise_std_std,
            size=self.size,
            generator=torch.Generator(device=self.device).manual_seed(
                self.random_state + self._step
            ),
            device=self.device,
        ).clamp_min_(self.min_std)
        # the std needs to be at least some small value, if std is zero, then it is not random at all.
        ## we assume each core has a background noise intensity(std) specific to this core.
        ## this core-wise std will be added to the noise_std_map. then different cores can have different noise intensity
        ## this core-wise noise intensity leads to unbalanced/uneven noise levels across c x r cores. Then enable inter-core remapping.
        # self._generate_core_noise_mean_map()

        core_noise_std_map = (
            torch.normal(
                mean=self.core_noise_mean_map,  # core-wise std level
                std=self.noise_std_std,  # core-wise std diversity
                # size=self.size[:-2], # std_mean for this core, approximated by the std_mean averaged across kxk rings
                generator=torch.Generator(device=self.device).manual_seed(
                    self.random_state + self._step
                ),
                # device=self.device,
            )
            .clamp_min_(self.min_std)
            .to(self.device)[..., None, None]
        )  # [c,r,1,1]  # the std needs to be at least some small value, if std is zero, then it is not random at all.

        # =========================================================  core-wise noise_mean_map  ========================================================#
        ## core-wise noise_mean_map, different core has different noise intensity, and we define 2 modes for noise_mean distribution
        ## 1: corner mode, noise intensity is most significant at left-up core, and smoothly distributed along x- and y- axis
        ## 2: edge mode, noise intensity is most significant at left column and distributed along x-axis

        noise_std_map = (core_noise_std_map + noise_std_map) / 2
        if self.gaussian_filter is not None:
            # we assume the noise intensity (i.e., std) distribution is smooth locally
            if self.smoothing_mode == "core":
                noise_std_map = torch.nn.functional.conv2d(
                    self.padder(noise_std_map).flatten(0, 1).unsqueeze(1),
                    self.gaussian_filter,
                    padding="valid",
                ).view_as(noise_std_map)
            elif self.smoothing_mode == "arch":
                noise_std_map = partition_chunks(
                    torch.nn.functional.conv2d(
                        self.padder(merge_chunks(noise_std_map)[None, None]),
                        self.gaussian_filter,
                        padding="valid",
                    )[0, 0],
                    out_shape=[4, 4, 8, 8],
                )
        return noise_std_map

    def update_noise_std_map(self):
        noise_std_map = self._generate_noise_std_map()
        if self.noise_std_map is None:
            self.noise_std_map = noise_std_map
        else:
            # every time step, we gradually update the noise std map to another random map, the momentum controls how much we keep the old map
            self.noise_std_map = (
                self.momentum * self.noise_std_map + (1 - self.momentum) * noise_std_map
            )

    def sample_noise(self, size=None, enable_remap: bool = False, col_ind=None):
        ## size [P, Q, k, k]: the workload size you want to map to this [R, C, K, K] multi-core MRR accelerator
        ## If size is None, then the workload is assumed to be [R, C, K, K]
        ## need to return [P, Q, k, k] phase noises for this workload
        ## assume the archiecture is [R, C, k, k]

        # when size=self.size, i.e., batch = [1, 1], then P=R, Q=C, i.e., each block in the layer weight matrix is mapped to a photonic core.
        # when batch = [u, v], we assume u=\ceil{P/R}, v=\ceil{Q/C}, i.e., the matrix needs to be partition into multiple RkxCk blocks and mapped sequentially to the same accelerator.
        size = size or self.size
        batch = (
            int(np.ceil(size[0] / self.size[0])),
            int(np.ceil(size[1] / self.size[1])),
        )

        # we assume the phase noise has zero mean, only std is determined by the noise_std_map
        # the P, Q, K, K workload will be chunked into u-by-v chunks (with same padding), each chunk is R, C, K, K, and thus can be mapping to the arch.
        # The u-by-v chunks require u-by-v times inference. The u-by-v inferences will see the same noise distribution, but different noise samples.
        # noise_std_map = einops.repeat(self.noise_std_map, "r c k l-> (u r) (v c) k l", u=batch[0], v=batch[1])[:size[0], :size[1]]
        noise_std_map = einops.repeat(
            self.noise_std_map, "r c k l-> u v r c k l", u=batch[0], v=batch[1]
        )
        if enable_remap and col_ind is not None:
            ## we remap noise distribution
            noise_std_map = apply_remap_noise(noise_std_map, col_ind=col_ind)
        noise_std_map = (
            noise_std_map.permute(0, 2, 1, 3, 4, 5)
            .flatten(0, 1)
            .flatten(1, 2)[: size[0], : size[1]]
        )

        noises = torch.normal(
            mean=0.0, std=noise_std_map
        )  # n ~ N(0, noise_std_map^2) different device has different std
        # noises = torch.normal(
        #     mean=0.0, std=self.noise_std_map
        # )  # n ~ N(0, noise_std_map^2) different device has different std
        self.noises = noises  ## add this to record the noise sampled.
        return noises


class GlobalTemperatureScheduler(object):
    def __init__(
        self,
        size=[4, 4, 8, 8],
        T_max: int = 1000,  # total number of steps
        n_g: float = 4.3,  # Bogaerts et al. 2012
        n_eff: float = 1.89,  # Bogaerts et al. 2012, TM Mode
        dwl_dT: float = 0.102,  # Bogaerts et al. 2012, TM Mode, d wavelength / d T. unit nm / K
        schedule_fn: Callable = lambda: 300,  # a function that returns a temperature in K unit, bu default is room temp
        T0: float = 300,  # initial room temperature
        lambda_res: List | Tensor | np.ndarray = [],
        L_list: List | Tensor | np.ndarray = [],
        hotspot_mode: str = "uniform",
        device="cuda:0",
    ) -> None:
        """
        just gradually set global temperature based on schedule_fn
        """
        # std of the phase noise follows Gaussian distribution ~ N(noise_std_mean, noise_std_std^2)
        super().__init__()
        self.size = size
        self.T_max = T_max
        self.schedule_fn = schedule_fn
        self.n_g = n_g
        self.n_eff = n_eff
        self.dwl_dT = dwl_dT
        self.T0 = T0
        self._last_T = T0
        self.L_list = L_list
        assert hotspot_mode in {"uniform", "corner"}
        self.hotspot_mode = hotspot_mode
        self.lambda_res = lambda_res
        if isinstance(lambda_res, list):
            self.lambda_res = torch.tensor(lambda_res, device=device)
        elif isinstance(lambda_res, np.ndarray):
            self.lambda_res = torch.from_numpy(lambda_res).to(device)

        if self.lambda_res.device != device:
            self.lambda_res = self.lambda_res.to(device)

        if isinstance(L_list, list):
            self.L_list = torch.tensor(L_list, device=device)
        elif isinstance(L_list, np.ndarray):
            self.L_list = torch.from_numpy(L_list).to(device)

        if self.L_list.device != device:
            self.L_list = self.L_list.to(device)

        self.device = device
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self.T = self.schedule_fn(0)

    def step(self) -> None:
        self._step += 1
        self.T = self.schedule_fn(self._step / self.T_max)

    def get_global_temp(self) -> float:
        return self.T

    def record_current_temp(self):
        self._last_T = self.T

    def get_hotspot_map(self) -> Tensor:
        if self.hotspot_mode == "uniform":
            hotspot_map = torch.ones(self.size[0:2], device=self.device)
        elif self.hotspot_mode == "corner":
            X, Y = torch.meshgrid(
                torch.arange(self.size[0], device=self.device),
                torch.arange(self.size[1], device=self.device),
            )
            hotspot_map = torch.exp(-1 * (X.square() + Y.square()).sqrt())
        else:
            raise NotImplementedError
        return hotspot_map

    def get_phase_drift(
        self, phase, T, enable_remap: bool = False, col_ind=None
    ) -> Tensor:
        """
        temperature drift will trigger lambda shift, i.e., delta_lambda, we assume lambda is linear to T, then
        delta_lambda = delta_T * d lambda / dT

        delta_lambda means there is a change on the neff, i.e., delta_neff
        delta_neff = delta_lambda * n_g / lambda_res

        the neff change leads to extra round-trip phase shift, e.g., delta_phi
        delta_phi = delta_neff * 2pi * R / lambda_res * 2pi
        The temperature change induced phase drift is only a function of T and wavelengths/Radius.
        For this [R,C,K,K] MRR weight bank architecture, only K different wavelengths/Radius, T is global.
        return delta_Phi [Tensor]: [K]-shaped tensor, each element is the phase drift for each wavelength/Radius.
        This can be naturally broadcast to [P,Q,K,K] workload (corresponding to the last dimension).
        """

        n_g = self.n_g  # Bogaerts et al. 2012
        n_eff = self.n_eff  # Bogaerts et al. 2012, TM Mode
        # delta_lambda = (T - self.T0) * self.dwl_dT
        hotspot_map = self.get_hotspot_map()[..., None, None]  # [R, C, 1, 1]
        delta_T = (T - self.T0) * hotspot_map
        delta_lambda = delta_T * self.dwl_dT  # [R, C, 1, 1]
        K = phase.shape[-1]
        lambda_res = self.lambda_res[
            self.lambda_res.shape[0] // 2 - K // 2 : self.lambda_res.shape[0] // 2
            - K // 2
            + K
        ]  # pass the central k wavelengths
        L_list = self.L_list[
            self.L_list.shape[0] // 2 - K // 2 : self.L_list.shape[0] // 2 - K // 2 + K
        ]
        delta_neff = delta_lambda * n_g / lambda_res  # [R, C, 1, k]
        delta_Phi = delta_neff * L_list * 1000 / lambda_res * 2 * np.pi  # [R, C, 1, k]

        size = phase.shape  # [P,Q,K,K]
        batch = (
            int(np.ceil(size[0] / self.size[0])),
            int(np.ceil(size[1] / self.size[1])),
        )

        # we assume the phase noise has zero mean, only std is determined by the noise_std_map
        # the P, Q, K, K workload will be chunked into u-by-v chunks (with same padding), each chunk is R, C, K, K, and thus can be mapping to the arch.
        # The u-by-v chunks require u-by-v times inference. The u-by-v inferences will see the same noise distribution, but different noise samples.
        delta_Phi = einops.repeat(
            delta_Phi, "r c k l-> u v r c k l", u=batch[0], v=batch[1]
        )
        if enable_remap and col_ind is not None:
            ## we remap noise distribution
            delta_Phi = apply_remap_noise(delta_Phi, col_ind=col_ind)
        delta_Phi = (
            delta_Phi.permute(0, 2, 1, 3, 4, 5)
            .flatten(0, 1)
            .flatten(1, 2)[: size[0], : size[1]]
        )

        return delta_Phi  # [P, Q, 1, k]


class CrosstalkScheduler(object):
    def __init__(
        self,
        # Size = [4,4,8,8],
        crosstalk_coupling_factor: float = 4.8,
        interv_h: float = 60.0,
        interv_v: float = 200.0,
        cutoff_calue: float = 1e-3,
        device="cuda:0",
    ) -> None:
        super().__init__()
        self.crosstalk_coupling_factor = crosstalk_coupling_factor
        self.interv_h = interv_h
        self.interv_v = interv_v
        self.cutoff_calue = cutoff_calue
        self.vh_coeff = self.interv_v / self.interv_h
        self.device = device

        self.crosstalk_mask = None

    @lru_cache(maxsize=8)
    def get_crosstalk_matrix(self, size) -> Tensor:
        k1, k2 = size[-2], size[-1]
        X, Y = torch.meshgrid(
            torch.arange(k1, device=self.device), torch.arange(k2, device=self.device)
        )
        X, Y = X.flatten().float(), Y.flatten().float()
        distance = (
            X.unsqueeze(1).sub(X.unsqueeze(0)).square() * self.interv_v**2
            + Y.unsqueeze(1).sub(Y.unsqueeze(0)).square() * self.interv_h**2
        ).sqrt()
        # print(X.shape, distance.shape)
        self.crosstalk_mask = torch.exp(-self.crosstalk_coupling_factor * distance)
        return self.crosstalk_mask


class DeterministicCtx:
    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = random_state

    def __enter__(self):
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        self.torch_cuda_random_state = torch.cuda.get_rng_state()
        set_torch_deterministic(self.random_state)
        return self

    def __exit__(self, *args):
        random.setstate(self.random_state)
        np.random.seed(self.numpy_random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        torch.cuda.set_rng_state(self.torch_cuda_random_state)


def calculate_grad_hessian(
    model, train_loader, criterion, num_samples=10, device="cuda:0"
):
    ## average gradients and second order gradients will be stored in weight._first_grad and weight._second_grad
    is_train = model.training
    model.train()
    bn_state = None
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            bn_state = m.training
            m.eval()
    params = []
    for m in model.modules():
        if isinstance(m, model._conv_linear):
            # print(m)
            m.weight._first_grad = 0
            m.weight._second_grad = 0
            params.append(m.weight)
    generator = torch.Generator(params[0].device).manual_seed(0)

    for idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward(create_graph=True)
        ## record the gradient
        grads = []
        for p in params:
            if p.grad is not None:
                ## accumulate gradients and average across all batches
                p._first_grad += p.grad.data / len(train_loader)
                grads.append(p.grad)

        # compute second order gradient
        for _ in range(num_samples):
            zs = [
                torch.randint(0, 2, p.size(), generator=generator, device=p.device)
                * 2.0
                - 1.0
                for p in params
            ]  # Rademacher distribution {-1.0, 1.0}
            h_zs = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                only_inputs=True,
                retain_graph=num_samples - 1,
            )
            for h_z, z, p in zip(h_zs, zs, params):
                ## accumulate second order gradients
                p._second_grad += h_z * z / (num_samples * len(train_loader))
        model.zero_grad()

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train(bn_state)
    model.train(is_train)
