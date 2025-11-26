"""
Utility functions for audio processing and visualization.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .speaker_diarization import SpeakerSegment


def format_time(seconds: float) -> str:
    """
    Format seconds as MM:SS.mmm

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def visualize_diarization(
    segments: List[SpeakerSegment],
    duration: float,
    role_mapping: Optional[dict] = None,
    figsize: Tuple[int, int] = (15, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize speaker diarization timeline.

    Args:
        segments: List of speaker segments
        duration: Total audio duration in seconds
        role_mapping: Optional mapping from speaker_id to role
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique speakers
    speakers = sorted(set(seg.speaker_id for seg in segments))
    speaker_colors = plt.cm.Set3(np.linspace(0, 1, len(speakers)))
    color_map = {speaker: color for speaker, color in zip(speakers, speaker_colors)}

    # Plot segments
    for segment in segments:
        speaker_idx = speakers.index(segment.speaker_id)
        color = color_map[segment.speaker_id]

        # Add role label if available
        label = segment.speaker_id
        if role_mapping and segment.speaker_id in role_mapping:
            role = role_mapping[segment.speaker_id]
            label = f"{segment.speaker_id} ({role})"

        ax.barh(
            speaker_idx,
            segment.duration(),
            left=segment.start,
            height=0.8,
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            label=label if segment == segments[0] or \
                  segment.speaker_id != segments[segments.index(segment)-1].speaker_id \
                  else ""
        )

    # Configure plot
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels([
        f"{spk} ({role_mapping.get(spk, 'unknown')})" if role_mapping else spk
        for spk in speakers
    ])
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Speaker', fontsize=12)
    ax.set_xlim(0, duration)
    ax.set_title('Speaker Diarization Timeline', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add legend with unique speakers
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_stereo_waveform(
    stereo_audio: np.ndarray,
    sr: int,
    segments: Optional[List[SpeakerSegment]] = None,
    role_mapping: Optional[dict] = None,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize stereo waveform with speaker segments.

    Args:
        stereo_audio: Stereo audio array of shape (2, num_samples)
        sr: Sample rate
        segments: Optional list of speaker segments
        role_mapping: Optional speaker to role mapping
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    duration = stereo_audio.shape[1] / sr
    time_axis = np.linspace(0, duration, stereo_audio.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Plot left channel (AI)
    axes[0].plot(time_axis, stereo_audio[0], linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Left Channel (AI)', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-1, 1)

    # Plot right channel (User)
    axes[1].plot(time_axis, stereo_audio[1], linewidth=0.5, alpha=0.7, color='orange')
    axes[1].set_ylabel('Amplitude', fontsize=11)
    axes[1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1].set_title('Right Channel (User)', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-1, 1)

    # Add speaker segments if provided
    if segments and role_mapping:
        for segment in segments:
            role = role_mapping.get(segment.speaker_id)
            if role == 'ai':
                axes[0].axvspan(segment.start, segment.end, alpha=0.2, color='blue')
            elif role == 'user':
                axes[1].axvspan(segment.start, segment.end, alpha=0.2, color='orange')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def calculate_audio_statistics(audio: np.ndarray, sr: int) -> dict:
    """
    Calculate basic audio statistics.

    Args:
        audio: Audio array (mono or stereo)
        sr: Sample rate

    Returns:
        Dictionary with statistics
    """
    if audio.ndim == 1:
        # Mono
        stats = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'num_samples': len(audio),
            'channels': 1,
            'rms': float(np.sqrt(np.mean(audio**2))),
            'peak': float(np.abs(audio).max()),
            'mean': float(np.mean(audio)),
            'std': float(np.std(audio))
        }
    else:
        # Stereo
        stats = {
            'duration': audio.shape[1] / sr,
            'sample_rate': sr,
            'num_samples': audio.shape[1],
            'channels': audio.shape[0],
            'channels_stats': {}
        }
        for i in range(audio.shape[0]):
            channel_name = ['left', 'right'][i] if audio.shape[0] == 2 else f'channel_{i}'
            stats['channels_stats'][channel_name] = {
                'rms': float(np.sqrt(np.mean(audio[i]**2))),
                'peak': float(np.abs(audio[i]).max()),
                'mean': float(np.mean(audio[i])),
                'std': float(np.std(audio[i]))
            }

    return stats


def validate_output_directory(output_dir: str) -> Path:
    """
    Validate and create output directory if needed.

    Args:
        output_dir: Output directory path

    Returns:
        Path object

    Raises:
        ValueError: If path exists but is not a directory
    """
    output_path = Path(output_dir)

    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Path exists but is not a directory: {output_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    path = Path(file_path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def print_separator(title: str = "", width: int = 60, char: str = "="):
    """
    Print a separator line with optional title.

    Args:
        title: Optional title to display
        width: Width of separator
        char: Character to use for separator
    """
    if title:
        print(f"\n{char*width}")
        print(f"{title:^{width}}")
        print(f"{char*width}")
    else:
        print(f"\n{char*width}")
