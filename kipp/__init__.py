"""Kipp: audible in-breaths and disfluent breaths in Swedish speech. See README.md."""
from .detector import BreathDetector, load_audio

__version__ = "1.0.0"
__all__ = ["BreathDetector", "load_audio"]
