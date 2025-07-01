"""
Agent package for PiDog robot control and interaction.

This package contains modules for controlling and interacting with the PiDog robot,
including speech recognition, model interactions, and action management.
"""

from .agentic_dog import main
from .model_helper import ModelHelper
from .action_flow import ActionFlow

__version__ = '0.1.0'

__all__ = [
    'main',
    'ModelHelper',
    'ActionFlow',
] 