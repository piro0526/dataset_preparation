"""
Data loader for PodcastFillers dataset.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf


class PodcastFillersLoader:
    """
    Loader for PodcastFillers dataset files.

    Handles loading of:
    - Audio files (episode WAV files)
    - Speech transcripts (JSON files)
    - Annotations (CSV files)
    - VAD data (CSV files)
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Root directory of PodcastFillers dataset
        """
        self.data_root = Path(data_dir)
        self.audio_root = self.data_root / "audio"
        self.metadata_root = self.data_root / "metadata"

        # Verify paths exist
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_root}")
        if not self.audio_root.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_root}")
        if not self.metadata_root.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_root}")

    def get_episode_list(self, split: str = "train") -> List[str]:
        """
        Get list of episode names for a given data split.

        Args:
            split: Data split ('train', 'validation', or 'test')

        Returns:
            List of episode filenames (without extension)
        """
        audio_dir = self.audio_root / "episode_wav" / split
        if not audio_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {audio_dir}")

        episodes = []
        for wav_file in audio_dir.glob("*.wav"):
            episodes.append(wav_file.stem)

        return sorted(episodes)

    def load_episode_audio(
        self,
        episode_name: str,
        split: str = "train"
    ) -> Tuple[np.ndarray, int]:
        """
        Load episode audio file.

        Args:
            episode_name: Episode filename (without extension)
            split: Data split ('train', 'validation', or 'test')

        Returns:
            Tuple of (audio_data, sample_rate)
            audio_data shape: (num_samples,) for mono
        """
        audio_path = self.audio_root / "episode_wav" / split / f"{episode_name}.wav"

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = sf.read(str(audio_path))
        return audio, sr

    def load_episode_transcript(
        self,
        episode_name: str,
        split: str = "train"
    ) -> Dict:
        """
        Load episode speech transcript (JSON).

        Args:
            episode_name: Episode filename (without extension)
            split: Data split ('train', 'validation', or 'test')

        Returns:
            Dictionary containing transcript data with structure:
            {
                'duration': int (microseconds),
                'language': str,
                'segments': [
                    {
                        'speaker': int,
                        'duration': int,
                        'offset': int,
                        'nbest': [{'text': str, 'words': [...]}]
                    }
                ]
            }
        """
        transcript_path = (
            self.metadata_root / "episode_transcripts" / split / f"{episode_name}.json"
        )

        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        return transcript

    def load_episode_annotations(
        self,
        episode_name: str,
        split: str = "train"
    ) -> pd.DataFrame:
        """
        Load episode annotations (CSV).

        Args:
            episode_name: Episode filename (without extension)
            split: Data split ('train', 'validation', or 'test')

        Returns:
            DataFrame with columns:
            - clip_name, pfID, label_full_vocab, label_consolidated_vocab
            - podcast_filename, event_start_inepisode, event_end_inepisode
            - event_start_inclip, event_end_inclip
            - clip_start_inepisode, clip_end_inepisode
            - duration, confidence, n_annotators
            - episode_split_subset, clip_split_subset, pitch_cent
        """
        annotation_path = (
            self.metadata_root / "episode_annotations" / split / f"{episode_name}.csv"
        )

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        df = pd.read_csv(annotation_path)
        return df

    def load_vad_data(
        self,
        episode_name: str,
        split: str = "train"
    ) -> Optional[pd.DataFrame]:
        """
        Load Voice Activity Detection data (CSV).

        Args:
            episode_name: Episode filename (without extension)
            split: Data split ('train', 'validation', or 'test')

        Returns:
            DataFrame with columns: timestamp, activation
            Returns None if VAD file not found
        """
        vad_path = self.metadata_root / "episode_vad" / split / f"{episode_name}.csv"

        if not vad_path.exists():
            return None

        df = pd.read_csv(vad_path, header=None, names=['timestamp', 'activation'])
        return df

    def get_episode_info(
        self,
        episode_name: str,
        split: str = "train"
    ) -> Dict:
        """
        Get comprehensive information about an episode.

        Args:
            episode_name: Episode filename (without extension)
            split: Data split ('train', 'validation', or 'test')

        Returns:
            Dictionary containing:
            - episode_name: str
            - split: str
            - audio_duration: float (seconds)
            - sample_rate: int
            - num_annotations: int
            - has_transcript: bool
            - has_vad: bool
        """
        info = {
            'episode_name': episode_name,
            'split': split,
        }

        # Load audio to get duration and sample rate
        try:
            audio, sr = self.load_episode_audio(episode_name, split)
            info['audio_duration'] = len(audio) / sr
            info['sample_rate'] = sr
        except FileNotFoundError:
            info['audio_duration'] = None
            info['sample_rate'] = None

        # Check annotations
        try:
            annotations = self.load_episode_annotations(episode_name, split)
            info['num_annotations'] = len(annotations)
        except FileNotFoundError:
            info['num_annotations'] = 0

        # Check transcript
        transcript_path = (
            self.metadata_root / "episode_transcripts" / split / f"{episode_name}.json"
        )
        info['has_transcript'] = transcript_path.exists()

        # Check VAD
        vad_path = self.metadata_root / "episode_vad" / split / f"{episode_name}.csv"
        info['has_vad'] = vad_path.exists()

        return info
