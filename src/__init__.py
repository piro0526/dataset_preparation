"""
PodcastFillers Speaker Diarization Module

This module provides tools for processing PodcastFillers dataset to create
stereo audio files suitable for Moshi fine-tuning.
"""

__version__ = "0.1.0"

from .data_loader import PodcastFillersLoader
from .speaker_diarization import SpeakerDiarizer, SpeakerSegment
from .audio_separator import AudioSeparator
from .stereo_generator import StereoAudioGenerator
from .pipeline import DiarizationPipeline

__all__ = [
    "PodcastFillersLoader",
    "SpeakerDiarizer",
    "SpeakerSegment",
    "AudioSeparator",
    "StereoAudioGenerator",
    "DiarizationPipeline",
]
