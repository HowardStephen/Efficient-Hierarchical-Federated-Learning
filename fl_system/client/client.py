# @PATH: fl_system/client/client.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Federated learning client. Implements FAFA-R algorithm lines 4-9.
# Local training, computes g_i, s_i (line 6), tau_i; uploads (g_i, tau_i, s_i).
# s_i = exp(-gamma * ||g_i - g_bar^(t-1)||^2); g_bar^(t-1) broadcast by cloud.
from __future__ import annotations

"""
Federated client: local training, computes g_i, s_i (line 6), tau_i; uploads (g_i, tau_i, s_i).
s_i = exp(-gamma * ||g_i - g_bar^(t-1)||^2), where g_bar^(t-1) is broadcast by the cloud.
"""

import copy
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import ClientConfig, HeterogeneityConfig
from ..utils.model_utils import state_dict_subtract
from ..aggregation import compute_similarity


class FederatedClient:
    """Federated learning client."""

    def __init__(
        self,
        client_id: int,
        local_data: TensorDataset,
        model: nn.Module,
        client_config: ClientConfig,
        heterogeneity_config: Optional[HeterogeneityConfig] = None,
        compute_power: Optional[float] = None,
        bandwidth_mbps: Optional[float] = None,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.local_data = local_data
        self.model = copy.deepcopy(model)
        self.client_config = client_config
        self.heterogeneity_config = heterogeneity_config or HeterogeneityConfig(enabled=False)
        self.device = device

        if self.heterogeneity_config.enabled:
            import random
            low, high = self.heterogeneity_config.compute_power_range
            self.compute_power = compute_power if compute_power is not None else random.uniform(low, high)
            low_b, high_b = self.heterogeneity_config.bandwidth_range
            self.bandwidth_mbps = bandwidth_mbps if bandwidth_mbps is not None else random.uniform(low_b, high_b)
        else:
            self.compute_power = 1.0
            self.bandwidth_mbps = 10.0

        self.model.to(device)
        self._dataloader: Optional[DataLoader] = None

    def _get_dataloader(self) -> DataLoader:
        if self._dataloader is None:
            self._dataloader = DataLoader(
                self.local_data,
                batch_size=self.client_config.batch_size,
                shuffle=True,
                drop_last=False,
            )
        return self._dataloader

    def set_model_state(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict, strict=True)

    def get_model_state(self) -> dict[str, Any]:
        return copy.deepcopy(self.model.state_dict())

    def local_train(self) -> tuple[dict[str, Any], float, int]:
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.client_config.learning_rate,
            momentum=0.9,
        )
        criterion = nn.CrossEntropyLoss()
        dataloader = self._get_dataloader()
        num_samples = len(self.local_data)

        start = time.perf_counter()
        for _ in range(self.client_config.local_epochs):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        actual_time = time.perf_counter() - start
        simulated_time = actual_time / self.compute_power
        return self.get_model_state(), simulated_time, num_samples

    def local_train_with_aggregation(
        self,
        global_state: dict[str, Any],
        g_bar_prev: Optional[dict[str, torch.Tensor]] = None,
        gamma: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], float, float, int]:
        """
        Algorithm lines 4-9: compute g_i, s_i, tau_i.

        Args:
            global_state: w^(t-1)
            g_bar_prev: g_bar^(t-1) previous round global average update; None for first round
            gamma: temperature parameter gamma

        Returns:
            (g_i, s_i, tau_i, num_samples)
        """
        self.set_model_state(global_state)
        w_global = {k: v.to(self.device).clone() for k, v in global_state.items()}

        w_local, training_time, num_samples = self.local_train()

        # Line 5: g_i = w^(t-1) - w_local
        g_i = state_dict_subtract(w_global, w_local)

        model_size = sum(p.numel() * p.element_size() for p in g_i.values())
        comm_delay = self.get_communication_delay(model_size)
        tau_i = training_time + comm_delay

        # Line 6: s_i = exp(-gamma * ||g_i - g_bar^(t-1)||^2)
        g_bar_cpu = {k: v.cpu().clone() for k, v in g_bar_prev.items()} if g_bar_prev else None
        g_i_cpu = {k: v.cpu().clone() for k, v in g_i.items()}
        s_i = compute_similarity(g_i_cpu, g_bar_cpu, gamma)

        return g_i_cpu, s_i, tau_i, num_samples

    def get_communication_delay(self, model_size_bytes: int) -> float:
        bandwidth_bytes_per_sec = self.bandwidth_mbps * 1e6 / 8
        return model_size_bytes / bandwidth_bytes_per_sec

    @property
    def num_samples(self) -> int:
        return len(self.local_data)


def create_clients(
    num_clients: int,
    data_by_client: dict[int, TensorDataset],
    model: nn.Module,
    client_config: ClientConfig,
    heterogeneity_config: Optional[HeterogeneityConfig] = None,
    device: str = "cpu",
) -> list[FederatedClient]:
    clients = []
    for i in range(num_clients):
        if i not in data_by_client:
            raise KeyError(f"Client {i} has no data in data_by_client")
        c = FederatedClient(
            client_id=i,
            local_data=data_by_client[i],
            model=model,
            client_config=client_config,
            heterogeneity_config=heterogeneity_config,
            device=device,
        )
        clients.append(c)
    return clients
