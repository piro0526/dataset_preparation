"""
End-to-end pipeline for speaker diarization and stereo audio generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .audio_separator import AudioSeparator
from .data_loader import PodcastFillersLoader
from .speaker_diarization import SpeakerDiarizer
from .stereo_generator import StereoAudioGenerator


class DiarizationPipeline:
    """
    Complete pipeline for processing podcast episodes into Moshi-ready stereo audio.

    Pipeline stages:
    1. Load episode audio
    2. Perform speaker diarization
    3. Separate audio by speaker
    4. Assign speaker roles (AI/user)
    5. Generate stereo audio (24kHz)
    6. Export with metadata
    """

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "outputs",
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the diarization pipeline.

        Args:
            data_dir: Root directory of PodcastFillers dataset
            output_dir: Directory for output files
            hf_token: Hugging Face authentication token
            device: Device for diarization ('cuda' or 'cpu')
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.loader = PodcastFillersLoader(data_dir)
        self.diarizer = SpeakerDiarizer(hf_token=hf_token, device=device)

        # Get configuration
        self.config = self._get_default_config()
        if config is not None:
            self.config.update(config)

        # Initialize audio processor
        self.separator = AudioSeparator(
            fade_duration=self.config['fade_duration'],
            overlap_strategy=self.config['overlap_strategy']
        )

        self.stereo_generator = StereoAudioGenerator(
            target_sr=self.config['target_sample_rate'],
            resample_method=self.config['resample_method']
        )

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'target_sample_rate': 24000,
            'resample_method': 'kaiser_fast',
            'fade_duration': 0.01,
            'overlap_strategy': 'mix',
            'min_speakers': 2,
            'max_speakers': 4,
            'min_segment_duration': 0.0,
            'role_assignment_strategy': 'speaking_time'
        }

    def process_episode(
        self,
        episode_name: str,
        split: str = "train",
        output_subdir: Optional[str] = None,
        num_speakers: Optional[int] = None,
        role_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a single episode.

        Args:
            episode_name: Episode filename (without extension)
            split: Data split ('train', 'validation', or 'test')
            output_subdir: Optional subdirectory within output_dir
            num_speakers: Force specific number of speakers
            role_mapping: Manual speaker role mapping {speaker_id: role}

        Returns:
            Dictionary containing:
            - episode_name: str
            - output_paths: Dict[str, str]
            - segments: List[dict]
            - role_mapping: Dict[str, str]
            - statistics: Dict
            - duration: float
        """
        print(f"\n{'='*60}")
        print(f"Processing: {episode_name}")
        print(f"{'='*60}")

        # Setup output directory
        if output_subdir is None:
            output_subdir = split
        episode_output_dir = self.output_dir / output_subdir / episode_name
        episode_output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Load audio
        print("\n[1/6] Loading audio...")
        audio, sr = self.loader.load_episode_audio(episode_name, split)
        duration = len(audio) / sr
        print(f"  ✓ Duration: {duration:.2f}s, Sample rate: {sr}Hz")

        # Stage 2: Diarization
        print("\n[2/6] Performing speaker diarization...")
        diarization_params = {
            'num_speakers': num_speakers,
            'min_speakers': self.config['min_speakers'],
            'max_speakers': self.config['max_speakers']
        }
        if num_speakers is not None:
            diarization_params = {'num_speakers': num_speakers}

        segments = self.diarizer.diarize(audio, sr, **diarization_params)
        print(f"  ✓ Found {len(segments)} segments")

        # Get speaker statistics
        stats = self.diarizer.get_speaker_statistics(segments)
        for speaker_id, speaker_stats in stats.items():
            print(f"    {speaker_id}: {speaker_stats['total_time']:.1f}s "
                  f"({speaker_stats['speaking_ratio']*100:.1f}%), "
                  f"{speaker_stats['num_segments']} segments")

        # Stage 3: Assign roles
        print("\n[3/6] Assigning speaker roles...")
        if role_mapping is None:
            role_mapping = self.diarizer.assign_roles(segments)

        for speaker_id, role in role_mapping.items():
            print(f"    {speaker_id} → {role}")

        # Stage 4: Separate audio
        print("\n[4/6] Separating audio by speaker...")
        separated = self.separator.separate_by_speaker(audio, segments, sr)
        print(f"  ✓ Separated into {len(separated)} speaker tracks")

        # Create role-based tracks
        role_tracks = self.separator.create_two_speaker_tracks(
            audio, segments, role_mapping, sr
        )

        # Stage 5: Generate stereo audio
        print("\n[5/6] Generating stereo audio (24kHz)...")
        stereo = self.stereo_generator.create_stereo_from_roles(role_tracks, sr)
        print(f"  ✓ Stereo shape: {stereo.shape}")

        # Validate
        validation = self.stereo_generator.validate_stereo(stereo)
        if not validation['valid']:
            print(f"  ⚠ Validation errors: {validation['errors']}")
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"  ⚠ Warning: {warning}")

        # Stage 6: Export
        print("\n[6/6] Exporting files...")
        output_base = episode_output_dir / "stereo_24k"

        metadata = {
            'episode_name': episode_name,
            'split': split,
            'original_duration': duration,
            'original_sample_rate': sr,
            'num_speakers': len(stats),
            'speaker_statistics': stats,
            'role_mapping': role_mapping,
            'config': self.config
        }

        output_paths = self.stereo_generator.export_for_moshi(
            stereo,
            str(output_base),
            metadata=metadata,
            segments=segments,
            role_mapping=role_mapping
        )

        print(f"  ✓ Saved to: {episode_output_dir}")
        for file_type, path in output_paths.items():
            print(f"    - {file_type}: {Path(path).name}")

        # Prepare result
        result = {
            'episode_name': episode_name,
            'split': split,
            'output_paths': output_paths,
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'speaker_id': seg.speaker_id,
                    'role': role_mapping.get(seg.speaker_id),
                    'duration': seg.duration()
                }
                for seg in segments
            ],
            'role_mapping': role_mapping,
            'statistics': stats,
            'duration': duration,
            'validation': validation
        }

        print(f"\n✓ Processing complete!")
        return result

    def batch_process(
        self,
        episode_names: Optional[List[str]] = None,
        split: str = "train",
        num_episodes: Optional[int] = None,
        skip_existing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple episodes in batch.

        Args:
            episode_names: List of episode names to process.
                          If None, process all episodes in split.
            split: Data split to process
            num_episodes: Maximum number of episodes to process
            skip_existing: Skip episodes that already have output

        Returns:
            List of result dictionaries from process_episode
        """
        # Get episode list
        if episode_names is None:
            episode_names = self.loader.get_episode_list(split)

        if num_episodes is not None:
            episode_names = episode_names[:num_episodes]

        print(f"\n{'='*60}")
        print(f"Batch Processing: {len(episode_names)} episodes from {split}")
        print(f"{'='*60}")

        results = []
        errors = []

        for i, episode_name in enumerate(tqdm(episode_names, desc="Processing episodes")):
            # Check if already processed
            if skip_existing:
                output_dir = self.output_dir / split / episode_name
                if (output_dir / "stereo_24k.wav").exists():
                    print(f"\nSkipping {episode_name} (already processed)")
                    continue

            try:
                result = self.process_episode(episode_name, split)
                results.append(result)
            except Exception as e:
                error_info = {
                    'episode_name': episode_name,
                    'split': split,
                    'error': str(e)
                }
                errors.append(error_info)
                print(f"\n✗ Error processing {episode_name}: {e}")
                continue

        # Save summary
        summary = {
            'total_episodes': len(episode_names),
            'successful': len(results),
            'failed': len(errors),
            'split': split,
            'results': results,
            'errors': errors
        }

        summary_path = self.output_dir / split / "processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"Successful: {len(results)}/{len(episode_names)}")
        print(f"Failed: {len(errors)}/{len(episode_names)}")
        print(f"Summary saved to: {summary_path}")

        return results

    def get_processing_summary(self, split: str = "train") -> Dict:
        """
        Get summary of processed episodes.

        Args:
            split: Data split

        Returns:
            Summary dictionary
        """
        summary_path = self.output_dir / split / "processing_summary.json"

        if not summary_path.exists():
            return {
                'total_episodes': 0,
                'successful': 0,
                'failed': 0,
                'results': [],
                'errors': []
            }

        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        return summary
