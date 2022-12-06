"""
RTCI: Real-time configuration interaction using split-operator propagation in python
"""

# core imports
from .ci import ci
from rtci.prop.prop import *

__all__ = ['ci','prop','utils']
