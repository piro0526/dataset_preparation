"""
Speaker diarization using pyannote.audio.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from pyannote.audio import Pipeline


@dataclass
class SpeakerSegment:
    """
    Represents a speaker segment with timing and identity.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        speaker_id: Speaker identifier (e.g., 'SPEAKER_00')
        confidence: Confidence score (0-1)
    """
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0

    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start

    def overlaps_with(self, other: 'SpeakerSegment') -> bool:
        """Check if this segment overlaps with another."""
        return not (self.end <= other.start or self.start >= other.end)


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio's pretrained models.

    Requires Hugging Face authentication token for model access.
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: Optional[str] = None
    ):
        """
        Initialize the speaker diarizer.

        Args:
            hf_token: Hugging Face authentication token.
                     If None, will try to read from HF_TOKEN environment variable.
            model_name: Pretrained model name from Hugging Face Hub
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)

        Raises:
            ValueError: If HF token is not provided and not in environment
        """
        # Get HF token
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "Hugging Face token is required. "
                "Provide it as argument or set HF_TOKEN environment variable."
            )

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load pipeline
        print(f"Loading speaker diarization model: {model_name}")
        print(f"Using device: {self.device}")

        try:
            self.pipeline = Pipeline.from_pretrained(
                model_name,
                token=hf_token
            )
            self.pipeline.to(torch.device(self.device))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load diarization model. "
                f"Make sure you have accepted the model's terms at: "
                f"https://huggingface.co/{model_name}\n"
                f"Error: {e}"
            )

    def diarize(
        self,
        audio: np.ndarray,
        sr: int,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Audio data as numpy array (mono)
            sr: Sample rate
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            List of SpeakerSegment objects ordered by start time
        """
        # Prepare audio dict for pyannote
        audio_dict = {
            'waveform': torch.from_numpy(audio).unsqueeze(0).float(),
            'sample_rate': sr
        }

        # Configure diarization parameters
        params = {}
        if num_speakers is not None:
            params['num_speakers'] = num_speakers
        if min_speakers is not None:
            params['min_speakers'] = min_speakers
        if max_speakers is not None:
            params['max_speakers'] = max_speakers

        # Run diarization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diarization = self.pipeline(audio_dict, **params)

        # Convert to SpeakerSegment list
        segments = []
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker_id=speaker,
                confidence=1.0  # pyannote doesn't provide confidence per segment
            )
            segments.append(segment)

        return sorted(segments, key=lambda s: s.start)

    def assign_roles(
        self,
        segments: List[SpeakerSegment],
        role_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Assign roles (AI/user) to speakers.

        Default strategy: Assign based on speaking time.
        - Most speaking time → AI (left channel)
        - Second most → User (right channel)

        Args:
            segments: List of speaker segments
            role_mapping: Optional manual mapping {speaker_id: role}
                         where role is 'ai' or 'user'

        Returns:
            Dictionary mapping speaker_id to role ('ai' or 'user')
        """
        if role_mapping is not None:
            return role_mapping

        # Calculate speaking time per speaker
        speaker_times = {}
        for segment in segments:
            speaker_id = segment.speaker_id
            duration = segment.duration()
            speaker_times[speaker_id] = speaker_times.get(speaker_id, 0.0) + duration

        # Sort speakers by speaking time (descending)
        sorted_speakers = sorted(
            speaker_times.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Assign roles
        roles = {}
        if len(sorted_speakers) >= 1:
            roles[sorted_speakers[0][0]] = 'ai'
        if len(sorted_speakers) >= 2:
            roles[sorted_speakers[1][0]] = 'user'

        # Assign remaining speakers to 'other' if any
        for speaker_id, _ in sorted_speakers[2:]:
            roles[speaker_id] = 'other'

        return roles

    def get_speaker_statistics(
        self,
        segments: List[SpeakerSegment]
    ) -> Dict[str, Dict]:
        """
        Get statistics about speakers.

        Args:
            segments: List of speaker segments

        Returns:
            Dictionary with speaker statistics:
            {
                'SPEAKER_00': {
                    'total_time': float,
                    'num_segments': int,
                    'avg_segment_duration': float,
                    'speaking_ratio': float
                },
                ...
            }
        """
        if not segments:
            return {}

        total_duration = max(s.end for s in segments)

        stats = {}
        speaker_segments = {}

        # Group segments by speaker
        for segment in segments:
            speaker_id = segment.speaker_id
            if speaker_id not in speaker_segments:
                speaker_segments[speaker_id] = []
            speaker_segments[speaker_id].append(segment)

        # Calculate statistics
        for speaker_id, spk_segments in speaker_segments.items():
            total_time = sum(s.duration() for s in spk_segments)
            num_segments = len(spk_segments)
            avg_duration = total_time / num_segments if num_segments > 0 else 0.0
            speaking_ratio = total_time / total_duration if total_duration > 0 else 0.0

            stats[speaker_id] = {
                'total_time': total_time,
                'num_segments': num_segments,
                'avg_segment_duration': avg_duration,
                'speaking_ratio': speaking_ratio
            }

        return stats

    def filter_segments(
        self,
        segments: List[SpeakerSegment],
        min_duration: float = 0.0,
        speaker_ids: Optional[List[str]] = None
    ) -> List[SpeakerSegment]:
        """
        Filter segments based on criteria.

        Args:
            segments: List of speaker segments
            min_duration: Minimum segment duration in seconds
            speaker_ids: Only keep segments from these speakers

        Returns:
            Filtered list of segments
        """
        filtered = segments

        if min_duration > 0:
            filtered = [s for s in filtered if s.duration() >= min_duration]

        if speaker_ids is not None:
            filtered = [s for s in filtered if s.speaker_id in speaker_ids]

        return filtered
