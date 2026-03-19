# @PATH: fl_system/client/__init__.py
# @DATE: 03.18.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Federated learning client module.
# Exports FederatedClient and create_clients.

from .client import FederatedClient, create_clients

__all__ = ["FederatedClient", "create_clients"]