"""Kipp: breaths in Swedish speech (BreathDetector), how much air each brings in (kipp.volume) and where a speaker breathes in a text
(kipp.lung). See README.md."""
from .detector import BreathDetector, load_audio

__version__ = "1.3.0"
__all__ = ["BreathDetector", "load_audio"]
