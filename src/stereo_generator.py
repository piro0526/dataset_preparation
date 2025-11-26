"""
Stereo audio generation for Moshi fine-tuning.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf

from .speaker_diarization import SpeakerSegment


class StereoAudioGenerator:
    """
    Generates stereo audio files suitable for Moshi fine-tuning.

    Channel assignment:
    - Left channel: AI responses
    - Right channel: User input
    """

    def __init__(
        self,
        target_sr: int = 24000,
        resample_method: str = "kaiser_fast"
    ):
        """
        Initialize stereo audio generator.

        Args:
            target_sr: Target sample rate for output (Moshi uses 24kHz)
            resample_method: Resampling method for librosa
                           ('kaiser_best', 'kaiser_fast', 'scipy', etc.)
        """
        self.target_sr = target_sr
        self.resample_method = resample_method

    def create_stereo(
        self,
        speaker_audios: Dict[str, np.ndarray],
        role_mapping: Dict[str, str],
        sr: int
    ) -> np.ndarray:
        """
        Create stereo audio from speaker tracks.

        Args:
            speaker_audios: Dictionary mapping speaker_id to mono audio
            role_mapping: Dictionary mapping speaker_id to role ('ai' or 'user')
            sr: Current sample rate

        Returns:
            Stereo audio array of shape (2, num_samples) at target sample rate
        """
        # Combine audio by role
        ai_audio = np.zeros_like(next(iter(speaker_audios.values())))
        user_audio = np.zeros_like(ai_audio)

        for speaker_id, audio in speaker_audios.items():
            role = role_mapping.get(speaker_id, 'other')
            if role == 'ai':
                ai_audio += audio
            elif role == 'user':
                user_audio += audio

        # Resample to target sample rate if needed
        if sr != self.target_sr:
            ai_audio = librosa.resample(
                ai_audio,
                orig_sr=sr,
                target_sr=self.target_sr,
                res_type=self.resample_method
            )
            user_audio = librosa.resample(
                user_audio,
                orig_sr=sr,
                target_sr=self.target_sr,
                res_type=self.resample_method
            )

        # Stack into stereo (left: AI, right: user)
        stereo = np.stack([ai_audio, user_audio], axis=0)

        # Normalize to prevent clipping
        max_val = np.abs(stereo).max()
        if max_val > 0.99:
            stereo = stereo * (0.99 / max_val)

        return stereo

    def create_stereo_from_roles(
        self,
        role_tracks: Dict[str, np.ndarray],
        sr: int
    ) -> np.ndarray:
        """
        Create stereo audio from already separated role tracks.

        Args:
            role_tracks: Dictionary with 'ai' and 'user' keys
            sr: Current sample rate

        Returns:
            Stereo audio array of shape (2, num_samples) at target sample rate
        """
        ai_audio = role_tracks.get('ai', np.zeros(1))
        user_audio = role_tracks.get('user', np.zeros(1))

        # Ensure same length
        max_len = max(len(ai_audio), len(user_audio))
        if len(ai_audio) < max_len:
            ai_audio = np.pad(ai_audio, (0, max_len - len(ai_audio)))
        if len(user_audio) < max_len:
            user_audio = np.pad(user_audio, (0, max_len - len(user_audio)))

        # Resample if needed
        if sr != self.target_sr:
            ai_audio = librosa.resample(
                ai_audio,
                orig_sr=sr,
                target_sr=self.target_sr,
                res_type=self.resample_method
            )
            user_audio = librosa.resample(
                user_audio,
                orig_sr=sr,
                target_sr=self.target_sr,
                res_type=self.resample_method
            )

        # Stack into stereo
        stereo = np.stack([ai_audio, user_audio], axis=0)

        # Normalize
        max_val = np.abs(stereo).max()
        if max_val > 0.99:
            stereo = stereo * (0.99 / max_val)

        return stereo

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Audio array (can be mono or stereo)
            orig_sr: Original sample rate
            target_sr: Target sample rate (uses self.target_sr if None)

        Returns:
            Resampled audio
        """
        if target_sr is None:
            target_sr = self.target_sr

        if orig_sr == target_sr:
            return audio

        # Handle stereo
        if audio.ndim == 2:
            resampled = np.stack([
                librosa.resample(
                    audio[i],
                    orig_sr=orig_sr,
                    target_sr=target_sr,
                    res_type=self.resample_method
                )
                for i in range(audio.shape[0])
            ])
            return resampled
        else:
            return librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type=self.resample_method
            )

    def export_for_moshi(
        self,
        stereo_audio: np.ndarray,
        output_path: str,
        metadata: Optional[Dict] = None,
        segments: Optional[List[SpeakerSegment]] = None,
        role_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Export stereo audio and metadata for Moshi fine-tuning.

        Args:
            stereo_audio: Stereo audio array of shape (2, num_samples)
            output_path: Path for output WAV file (without extension)
            metadata: Optional metadata dictionary
            segments: Optional list of speaker segments
            role_mapping: Optional speaker to role mapping

        Returns:
            Dictionary with paths to created files
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save WAV file (transpose for soundfile: (num_samples, 2))
        wav_path = output_path.with_suffix('.wav')
        sf.write(
            str(wav_path),
            stereo_audio.T,  # Transpose to (T, 2)
            self.target_sr,
            subtype='PCM_16'
        )

        # Create and save metadata
        meta = {
            'sample_rate': self.target_sr,
            'channels': 2,
            'duration_seconds': stereo_audio.shape[1] / self.target_sr,
            'channel_mapping': {
                'left': 'ai',
                'right': 'user'
            }
        }

        if metadata is not None:
            meta.update(metadata)

        if segments is not None:
            meta['segments'] = [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'speaker': seg.speaker_id,
                    'role': role_mapping.get(seg.speaker_id) if role_mapping else None,
                    'confidence': seg.confidence
                }
                for seg in segments
            ]

        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Create CSV for segments (if provided)
        csv_path = None
        if segments is not None:
            import pandas as pd
            csv_path = output_path.with_suffix('.csv')

            df = pd.DataFrame([
                {
                    'start': seg.start,
                    'end': seg.end,
                    'duration': seg.duration(),
                    'speaker_id': seg.speaker_id,
                    'role': role_mapping.get(seg.speaker_id) if role_mapping else None,
                    'confidence': seg.confidence
                }
                for seg in segments
            ])
            df.to_csv(csv_path, index=False)

        result = {
            'wav': str(wav_path),
            'metadata': str(metadata_path)
        }
        if csv_path:
            result['csv'] = str(csv_path)

        return result

    def validate_stereo(
        self,
        stereo_audio: np.ndarray
    ) -> Dict[str, any]:
        """
        Validate stereo audio format.

        Args:
            stereo_audio: Stereo audio array

        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # Check shape
        if stereo_audio.ndim != 2:
            validation['valid'] = False
            validation['errors'].append(
                f"Expected 2D array (stereo), got shape {stereo_audio.shape}"
            )
            return validation

        if stereo_audio.shape[0] != 2:
            validation['valid'] = False
            validation['errors'].append(
                f"Expected 2 channels, got {stereo_audio.shape[0]}"
            )

        # Check for silence
        left_rms = np.sqrt(np.mean(stereo_audio[0]**2))
        right_rms = np.sqrt(np.mean(stereo_audio[1]**2))

        if left_rms < 1e-6:
            validation['warnings'].append("Left channel (AI) appears to be silent")
        if right_rms < 1e-6:
            validation['warnings'].append("Right channel (user) appears to be silent")

        # Check for clipping
        if np.abs(stereo_audio).max() > 0.99:
            validation['warnings'].append("Audio may be clipping (max amplitude > 0.99)")

        # Check for DC offset
        left_mean = np.mean(stereo_audio[0])
        right_mean = np.mean(stereo_audio[1])

        if abs(left_mean) > 0.01:
            validation['warnings'].append(f"Left channel has DC offset: {left_mean:.4f}")
        if abs(right_mean) > 0.01:
            validation['warnings'].append(f"Right channel has DC offset: {right_mean:.4f}")

        return validation
