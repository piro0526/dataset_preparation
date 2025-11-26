"""
Audio separation by speaker using masking techniques.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import signal

from .speaker_diarization import SpeakerSegment


class AudioSeparator:
    """
    Separates audio by speaker using time-domain masking.

    This class creates separate audio tracks for each speaker
    based on diarization segments.
    """

    def __init__(
        self,
        fade_duration: float = 0.01,
        overlap_strategy: str = "mix"
    ):
        """
        Initialize audio separator.

        Args:
            fade_duration: Duration of fade in/out at segment boundaries (seconds)
            overlap_strategy: How to handle overlapping speech
                - 'mix': Mix both speakers
                - 'primary': Keep primary speaker only
                - 'first': Keep first speaker only
        """
        self.fade_duration = fade_duration
        self.overlap_strategy = overlap_strategy

    def separate_by_speaker(
        self,
        audio: np.ndarray,
        segments: List[SpeakerSegment],
        sr: int
    ) -> Dict[str, np.ndarray]:
        """
        Separate audio into individual speaker tracks.

        Args:
            audio: Audio data as numpy array (mono), shape: (num_samples,)
            segments: List of speaker segments
            sr: Sample rate

        Returns:
            Dictionary mapping speaker_id to audio array
            Each audio array has same length as input
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio, got shape {audio.shape}")

        # Get unique speakers
        speakers = list(set(s.speaker_id for s in segments))

        # Initialize output arrays
        separated = {speaker: np.zeros_like(audio) for speaker in speakers}

        # Create masks for each segment
        fade_samples = int(self.fade_duration * sr)

        for segment in segments:
            start_idx = int(segment.start * sr)
            end_idx = int(segment.end * sr)

            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(audio), end_idx)

            if start_idx >= end_idx:
                continue

            # Create mask with fade in/out
            segment_length = end_idx - start_idx
            mask = np.ones(segment_length)

            # Apply fade in
            if fade_samples > 0 and segment_length > fade_samples:
                fade_in = np.linspace(0, 1, min(fade_samples, segment_length))
                mask[:len(fade_in)] = fade_in

            # Apply fade out
            if fade_samples > 0 and segment_length > fade_samples:
                fade_out = np.linspace(1, 0, min(fade_samples, segment_length))
                mask[-len(fade_out):] = fade_out

            # Apply mask to audio
            separated[segment.speaker_id][start_idx:end_idx] += audio[start_idx:end_idx] * mask

        # Handle potential overlaps (where sum > original)
        for speaker in speakers:
            # Clip to prevent amplification
            separated[speaker] = np.clip(
                separated[speaker],
                -np.abs(audio).max(),
                np.abs(audio).max()
            )

        return separated

    def create_two_speaker_tracks(
        self,
        audio: np.ndarray,
        segments: List[SpeakerSegment],
        role_mapping: Dict[str, str],
        sr: int
    ) -> Dict[str, np.ndarray]:
        """
        Create two separate tracks for AI and user roles.

        Args:
            audio: Audio data as numpy array (mono)
            segments: List of speaker segments
            role_mapping: Mapping from speaker_id to role ('ai' or 'user')
            sr: Sample rate

        Returns:
            Dictionary with keys 'ai' and 'user', each containing an audio array
        """
        # First separate by speaker
        separated = self.separate_by_speaker(audio, segments, sr)

        # Combine speakers by role
        tracks = {
            'ai': np.zeros_like(audio),
            'user': np.zeros_like(audio)
        }

        for speaker_id, speaker_audio in separated.items():
            role = role_mapping.get(speaker_id)
            if role in ['ai', 'user']:
                tracks[role] += speaker_audio

        return tracks

    def detect_overlaps(
        self,
        segments: List[SpeakerSegment]
    ) -> List[tuple]:
        """
        Detect overlapping speech segments.

        Args:
            segments: List of speaker segments

        Returns:
            List of tuples (segment1, segment2) that overlap
        """
        overlaps = []

        for i, seg1 in enumerate(segments):
            for seg2 in segments[i+1:]:
                if seg1.overlaps_with(seg2) and seg1.speaker_id != seg2.speaker_id:
                    overlaps.append((seg1, seg2))

        return overlaps

    def get_speech_activity_mask(
        self,
        segments: List[SpeakerSegment],
        audio_length: int,
        sr: int,
        speaker_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Create a binary mask indicating speech activity.

        Args:
            segments: List of speaker segments
            audio_length: Total length of audio in samples
            sr: Sample rate
            speaker_id: If specified, only mask for this speaker

        Returns:
            Binary mask array of shape (audio_length,)
        """
        mask = np.zeros(audio_length, dtype=bool)

        for segment in segments:
            if speaker_id is not None and segment.speaker_id != speaker_id:
                continue

            start_idx = int(segment.start * sr)
            end_idx = int(segment.end * sr)

            start_idx = max(0, start_idx)
            end_idx = min(audio_length, end_idx)

            mask[start_idx:end_idx] = True

        return mask

    def apply_noise_reduction(
        self,
        audio: np.ndarray,
        mask: np.ndarray,
        noise_reduction_db: float = 40.0
    ) -> np.ndarray:
        """
        Apply noise reduction to non-speech regions.

        Args:
            audio: Audio data
            mask: Binary mask indicating speech regions (True = speech)
            noise_reduction_db: Amount of noise reduction in dB

        Returns:
            Audio with reduced noise in non-speech regions
        """
        output = audio.copy()

        # Apply attenuation to non-speech regions
        attenuation = 10 ** (-noise_reduction_db / 20.0)
        output[~mask] *= attenuation

        return output
