# @PATH: fl_system/models/__init__.py
# @DATE: 03.19.2026
# @AUTHOR: Howard
# @E-MAIL: QSX20251439@student.fjnu.edu.cn
#
# Model definitions for federated learning.
# Exports VGGLite (CIFAR), FEMNISTLeNet (FEMNIST), and CIFARCNN alias.

from .vgg_lite import VGGLite
from .lenet import FEMNISTLeNet

# Backward compatibility alias
CIFARCNN = VGGLite

__all__ = ["VGGLite", "FEMNISTLeNet", "CIFARCNN"]