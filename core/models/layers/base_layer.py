"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-11-14 16:53:34
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-11-17 15:34:49
"""

from typing import Any, Dict

import numpy as np
import torch
from pyutils.compute import add_gaussian_noise
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.types import Device

__all__ = ["ONNBaseLayer"]


class ONNBaseLayer(nn.Module):
    def __init__(self, *args, device: Device = torch.device("cpu"), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # cuda or cpu, defaults to cpu
        self.device = device

    def build_parameters(self) -> None:
        raise NotImplementedError

    def reset_parameters(self) -> None:
        raise NotImplementedError

    @classmethod
    def from_layer(cls, layer: nn.Module, *args, **kwargs) -> nn.Module:
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_fast_forward(self) -> None:
        self.fast_forward_flag = True

    def disable_fast_forward(self) -> None:
        self.fast_forward_flag = False

    # phase variation
    def set_phase_variation(self, flag: bool = False) -> None:
        self._enable_phase_variation = flag

    # tenperature drift
    def set_global_temp_drift(self, flag: bool = False) -> None:
        self._enable_global_temp_drift = flag

    # crosstalk
    def set_crosstalk_noise(self, flag: bool = False) -> None:
        self._enable_crosstalk = flag

    def set_weight_noise(self, noise_std: float = 0.0) -> None:
        self.weight_noise_std = noise_std

    def set_weight_bitwidth(self, w_bit: int) -> None:
        self.w_bit = w_bit

    def set_input_bitwidth(self, in_bit: int) -> None:
        self.in_bit = in_bit

    def load_parameters(self, param_dict: Dict[str, Any]) -> None:
        """
        description: update parameters based on this parameter dictionary\\
        param param_dict {dict of dict} {param_name: param_tensor, ...}
        """
        for name, param in param_dict.items():
            getattr(self, name).data.copy_(param)

    def switch_mode_to(self, mode: str) -> None:
        self.mode = mode

    def set_enable_ste(self, enable_ste: bool) -> None:
        self._enable_ste = enable_ste

    def set_enable_remap(self, enable_remap: bool) -> None:
        self._enable_remap = enable_remap

    def set_noise_flag(self, noise_flag: bool) -> None:
        self._noise_flag = noise_flag

    def _add_phase_variation(
        self, x, src: float = "weight", enable_remap: bool = False
    ) -> None:
        # this function can handle both phase noise injection to phase tensors and weight tensors
        if (not self._enable_phase_variation) or self.phase_variation_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight(x)
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        noise = self.phase_variation_scheduler.sample_noise(
            size=phase.shape, enable_remap=enable_remap, col_ind=self.col_ind
        )

        phase = phase + noise

        if src == "weight":
            x = self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phase)).mul(
                S_scale[..., None]
            )
        else:
            x = phase

        return x

    def _add_global_temp_drift(
        self, x, src: float = "weight", enable_remap: bool = False
    ) -> None:
        if (not self._enable_global_temp_drift) or self.global_temp_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight(x)
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        T = self.global_temp_scheduler.get_global_temp()
        noise = self.global_temp_scheduler.get_phase_drift(
            phase, T, enable_remap=enable_remap, col_ind=self.col_ind
        )
        phase = phase + noise

        if src == "weight":
            x = self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phase)).mul(
                S_scale[..., None]
            )
        else:
            x = phase

        return x

    def _add_crosstalk_noise(self, x, src: str = "weight") -> None:
        if (not self._enable_crosstalk) or self.crosstalk_scheduler is None:
            return x  #  no noise injected

        if src == "weight":
            phase, S_scale = self.build_phase_from_weight(x)
        elif src == "phase":
            phase = x
        else:
            raise NotImplementedError

        self.crosstalk_scheduler.get_crosstalk_matrix(self.phase.shape)
        crosstalk_coupling_matrix = self.crosstalk_scheduler.crosstalk_mask
        phase_shape = phase.shape

        phase_ct = torch.matmul(
            crosstalk_coupling_matrix,
            phase.view(phase_shape[0], phase_shape[1], -1).unsqueeze(3),
        ).view(phase_shape[0], phase_shape[1], phase_shape[2], -1)

        if src == "weight":
            x = self.mrr_tr_to_weight(self.mrr_roundtrip_phase_to_tr(phase_ct)).mul(
                S_scale[..., None]
            )
        else:
            x = phase_ct

        return x

    def build_weight(
        self,
        weight=None,
        flag: bool = True,
        enable_ste: bool = False,
        enable_remap: bool = False,
    ) -> Tensor:
        if self.mode == "weight":
            weight = weight if weight is not None else self.weight

            if flag:
                if enable_ste:
                    weight_tmp = weight.detach()
                else:
                    weight_tmp = weight

                phase, S_scale = self.build_phase_from_weight(weight_tmp)
                # step 1 add random phase variation
                phase = self._add_phase_variation(
                    phase, src="phase", enable_remap=enable_remap
                )

                # step 2 add global temperature drift
                phase = self._add_global_temp_drift(
                    phase, src="phase", enable_remap=enable_remap
                )

                # step 3 add thermal crosstalk
                phase = self._add_crosstalk_noise(phase, src="phase")

                # reconstruct noisy weight
                weight_noisy = self.mrr_tr_to_weight(
                    self.mrr_roundtrip_phase_to_tr(phase)
                ).mul(S_scale[..., None])
                if enable_ste:
                    weight = (
                        weight_noisy - weight
                    ).detach() + weight  # cut off gradient, only flow through weight
                else:
                    weight = weight_noisy

        elif self.mode == "phase":
            if self.w_bit < 16:
                phase = self.phase_quantizer(self.phase)
            else:
                phase = self.phase

            if self.phase_noise_std > 1e-5:
                phase = add_gaussian_noise(
                    phase,
                    0,
                    self.phase_noise_std,
                    trunc_range=(-2 * self.phase_noise_std, 2 * self.phase_noise_std),
                )

            weight = self.build_weight_from_phase(phase)
        elif self.mode == "voltage":
            raise NotImplementedError
        else:
            raise NotImplementedError
        if self.weight_noise_std > 1e-6:
            weight = weight * (1 + torch.randn_like(weight) * self.weight_noise_std)
        return weight

    def layer_weight_partition_chunk(
        self, X: Tensor, require_size: torch.Size = [4, 4, 8, 8], complex: bool = False
    ) -> Tensor:
        """this function is used to partition layer weight into our required [R,C,K,K] size"""
        if isinstance(X, torch.Tensor):
            R, C = require_size[0], require_size[1]
            P, Q, k = X.shape[0:3]
            shape = int(np.ceil(P / R) * R), int(np.ceil(Q / C) * C)  # [P_pad, Q_pad]
            X = torch.nn.functional.pad(
                X,
                (0, 0, 0, 0, 0, shape[1] - Q, 0, shape[0] - P),
                mode="constant",
                value=0,
            )  # [P_pad, Q_pad, k, k]
            X = X.reshape(shape[0] // R, R, shape[1] // C, C, k, k).permute(
                0, 2, 1, 3, 4, 5
            )  # [b0, b1, R,C,K,K]

            return X

        else:
            raise NotImplementedError

    def layer_weight_merge_chunk(self, x: Tensor, complex: bool = False) -> Tensor:
        if not complex:
            bc_x, bc_y, R, C, K = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
            x = x.permute(0, 2, 1, 3, 4, 5)  # [bc_x, R, bc_y, C, K, K]
            x = x.reshape(bc_x * R, bc_y * C, K, K)  # [P_pad, Q_pad, K, K]
        else:
            bc_x, bc_y, R, C, K = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
            x = x.permute(0, 2, 1, 3, 4, 5, 6)  # [bc_x, R, bc_y, C, K, K]
            x = x.reshape(bc_x * R, bc_y * C, K, K, 2)  # [P_pad, Q_pad, K, K]

        return x

    def remap_intra_tile(
        self,
        alg: str = "LAP",
        salience_mode="first_grad",
        average_times: int = 1,
        tolerance: float = 1,
    ):
        """Remap for [R,C,K,K] cores, map noise
        This function only solves row_ind and col_ind, it will not apply those indices to weights.
        """
        assert alg in {"LAP", "heuristic"}
        assert salience_mode in {"first_grad", "second_grad", "none"}
        self.row_ind, self.col_ind = [], []
        layer_weight = self._ideal_weight

        size = self.phase_variation_scheduler.size
        self.batch_weight_cores = self.layer_weight_partition_chunk(
            self.weight.data,
            require_size=size,  # [b0, b1, R, C, k, k]
        )

        self.batch_ideal_weight_core = self.layer_weight_partition_chunk(
            layer_weight, require_size=size
        )  # [b0, b1, R, C, k, k]

        if salience_mode == "none":
            first_salience = second_salience = None
        elif salience_mode == "first_grad":
            first_salience = self.layer_weight_partition_chunk(
                self.weight._first_grad, require_size=size
            )
            second_salience = None
        elif salience_mode == "second_grad":
            first_salience = self.layer_weight_partition_chunk(
                self.weight._first_grad, require_size=size
            )
            second_salience = self.layer_weight_partition_chunk(
                self.weight._second_grad, require_size=size
            )
        else:
            raise NotImplementedError

        # generate epsilon matrix [b0, b1, R, R]
        # we only need shift weight row-wise (R) and parallel prob
        # here we assume average times = 1 based on our previuos experiments
        all_weights = []
        all_first_salience = []
        all_second_salience = []
        ## [W1, W2, ..., WR]
        ## [W2, W3, ..., W1]
        ## [WR, WR-1, ..., W1]
        for r in range(size[0]):
            shifted_weights = self.layer_weight_merge_chunk(
                self.batch_weight_cores.data.roll(-r, dims=2)
            )[
                : self.grid_dim_y, : self.grid_dim_x
            ]  # [b0, b1, <R>, C, K, k] -> shift and merge -> [P, Q, K, K]
            shifted_weights = self.build_weight(
                weight=shifted_weights,
                flag=True,
                enable_ste=False,
                enable_remap=False,
            )
            shifted_weights = self.layer_weight_partition_chunk(
                shifted_weights, require_size=size
            ).roll(r, dims=2)  # [b0, b1, R, C, k, k]
            all_weights.append(shifted_weights)  # [b0, b1, R, C, k, k]
            if first_salience is not None:
                all_first_salience.append(first_salience.roll(-r, dims=2))
            if second_salience is not None:
                all_second_salience.append(second_salience.roll(-r, dims=2))
        all_weights = torch.stack(all_weights, dim=2)  # [b0, b1, <R>, R, C, k, k]

        if len(all_first_salience) > 0:
            all_first_salience = torch.stack(
                all_first_salience, dim=2
            )  # [b0, b1, <R>, R, C, k, k]
        if len(all_second_salience) > 0:
            all_second_salience = torch.stack(
                all_second_salience, dim=2
            )  # [b0, b1, <R>, R, C, k, k]

        if salience_mode == "none":
            epsilon_matrix = all_weights.sub(
                self.batch_ideal_weight_core.unsqueeze(2)
            ).norm(p=1, dim=(-3, -2, -1))  # [b0, b1, <R>, R]
        elif salience_mode == "first_grad":
            epsilon_matrix = (
                all_weights.sub(self.batch_ideal_weight_core.unsqueeze(2))
                .mul(all_first_salience)
                .sum(dim=[-1, -2, -3])
                .abs()
            )  # [b0, b1, <R>, R]
        elif salience_mode == "second_grad":
            err = all_weights.sub(self.batch_ideal_weight_core.unsqueeze(2))
            epsilon_matrix = (
                err.mul(all_first_salience).sum(dim=[-1, -2, -3])
                + 0.5 * err.square().mul(all_second_salience).sum(dim=[-1, -2, -3])
            ).abs()
            # [b0, b1, <R>, R]

            for col in range(1, epsilon_matrix.shape[-1]):
                epsilon_matrix[..., col] = epsilon_matrix[..., col].roll(col, dims=-1)

            epsilon_matrix = epsilon_matrix.data

        else:
            raise NotImplementedError

        # apply threshold based on tolerance and regenerate epsilon_matrix and tile_indices
        tile_errors = epsilon_matrix.mean(dim=-2)  # [b0, b1, R]
        tile_min_errors = tile_errors.min(dim=-1)[0]  # [b0, b1]
        tile_indices = torch.zeros_like(
            tile_errors
        )  # [b0, b1, R], important, used to reinterprete tile remapping
        max_workload_assigned = torch.zeros_like(tile_min_errors).to(
            torch.int32
        )  # [b1, b0]
        for b0 in range(epsilon_matrix.shape[0]):
            for b1 in range(epsilon_matrix.shape[1]):
                tile_err = tile_errors[
                    b0, b1
                ]  # e.g., R=5, [0.3, 0.02, 0.05, 0.2, 0.03]
                tile_mask = tile_err <= max(tolerance, tile_min_errors[b0, b1])
                good_tiles = torch.nonzero(tile_mask)[:, 0]
                # duplicate good tiles to fill the whole rows
                tile_times, workload_left = divmod(size[0], len(good_tiles))
                # every tile will do tile_times by default
                # then the rest workload_left will be spread to lowest error tiles.
                selected_tiles = torch.argsort(tile_err[good_tiles])[
                    :workload_left
                ]  # [0.02, 0.05, 0.03] -> [0, 2, 1] -> [0, 2]
                workload_assigned = torch.tensor(
                    [tile_times] * len(good_tiles), device=self.device
                )
                workload_assigned[selected_tiles] += (
                    1  # [1, 1, 1] + [1, 0, 1] -> [2, 1, 2] total workload
                )
                max_workload_assigned[b0, b1] = workload_assigned.max()
                tile_index = []
                # [1, 2, 4] repeat [2, 1, 2] -> [1, 1, 2, 4, 4]
                for idx, assigned in zip(good_tiles, workload_assigned):
                    tile_index.extend([idx] * assigned)
                tile_indices[b0, b1] = torch.tensor(tile_index, device=self.device)
                epsilon_matrix[b0, b1].copy_(epsilon_matrix[b0, b1, :, tile_index])
        epsilon_matrix = epsilon_matrix.detach().cpu().numpy()
        self.max_workload_assigned = max_workload_assigned

        # we need to solve b0 x b1 linear assignment problem
        self.row_ind = []
        for b0 in range(epsilon_matrix.shape[0]):
            row_ind_list, col_ind_list = [], []
            for b1 in range(epsilon_matrix.shape[1]):
                row_ind, col_ind = linear_sum_assignment(epsilon_matrix[b0, b1])
                row_ind_list.append(torch.from_numpy(row_ind).to(self.device))

                # need to reinterprete col_ind based on tile_indices
                # e.g., tile_indices [1, 1, 2, 4, 4]
                # col_ind            [0, 3, 2, 4, 1]
                # actual col_ind     [1, 4, 2, 4, 1]
                col_ind = tile_indices[b0, b1][col_ind]
                col_ind_list.append(col_ind)

            self.row_ind.append(torch.stack(row_ind_list))
            self.col_ind.append(torch.stack(col_ind_list))

        self.row_ind = torch.stack(self.row_ind).long()
        self.col_ind = torch.stack(self.col_ind).long()

        cycles = (
            (size[0] * size[-1] + size[0] ** 3)
            * epsilon_matrix.shape[0]
            * epsilon_matrix.shape[1]
        )  # Rk + R^3
        return self.row_ind, self.col_ind, cycles

    def forward(self, x):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""
