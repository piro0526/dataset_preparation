#!/usr/bin/env python3
"""
PodcastFillers Speaker Diarization CLI

Command-line interface for processing podcast episodes into Moshi-ready
stereo audio files with speaker diarization.
"""

import argparse
import os
import sys
from pathlib import Path

from src import DiarizationPipeline
from src.data_loader import PodcastFillersLoader


def process_single(args):
    """Process a single episode."""
    pipeline = DiarizationPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hf_token=args.hf_token or os.environ.get('HF_TOKEN'),
        device=args.device
    )

    result = pipeline.process_episode(
        episode_name=args.episode,
        split=args.split,
        num_speakers=args.num_speakers
    )

    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Episode: {result['episode_name']}")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Speakers: {len(result['statistics'])}")
    print(f"Segments: {len(result['segments'])}")
    print(f"\nOutput files:")
    for file_type, path in result['output_paths'].items():
        print(f"  {file_type}: {path}")


def process_batch(args):
    """Process multiple episodes in batch."""
    pipeline = DiarizationPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hf_token=args.hf_token or os.environ.get('HF_TOKEN'),
        device=args.device
    )

    results = pipeline.batch_process(
        split=args.split,
        num_episodes=args.num_episodes,
        skip_existing=not args.overwrite
    )

    print(f"\n✓ Processed {len(results)} episodes successfully")


def list_episodes(args):
    """List available episodes."""
    loader = PodcastFillersLoader(args.data_dir)

    for split in ['train', 'validation', 'test']:
        episodes = loader.get_episode_list(split)
        print(f"\n{split.upper()}: {len(episodes)} episodes")

        if args.verbose:
            for i, episode in enumerate(episodes[:args.limit], 1):
                info = loader.get_episode_info(episode, split)
                duration = info.get('audio_duration', 0)
                print(f"  {i:3d}. {episode[:60]:<60} ({duration:.1f}s)")


def show_info(args):
    """Show information about an episode."""
    loader = PodcastFillersLoader(args.data_dir)

    info = loader.get_episode_info(args.episode, args.split)

    print("\n" + "="*60)
    print("Episode Information")
    print("="*60)
    print(f"Name: {info['episode_name']}")
    print(f"Split: {info['split']}")
    print(f"Duration: {info['audio_duration']:.2f}s ({info['audio_duration']/60:.2f}min)")
    print(f"Sample rate: {info['sample_rate']}Hz")
    print(f"Annotations: {info['num_annotations']}")
    print(f"Has transcript: {'Yes' if info['has_transcript'] else 'No'}")
    print(f"Has VAD: {'Yes' if info['has_vad'] else 'No'}")


def show_summary(args):
    """Show processing summary."""
    pipeline = DiarizationPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    summary = pipeline.get_processing_summary(args.split)

    print("\n" + "="*60)
    print(f"Processing Summary: {args.split}")
    print("="*60)
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")

    if summary['errors'] and args.verbose:
        print("\nErrors:")
        for error in summary['errors']:
            print(f"  - {error['episode_name']}: {error['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="PodcastFillers Speaker Diarization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single episode
  python main.py process-single "a16z_a16z Podcast A Podcast about Podcasts" --split train

  # Process all training episodes
  python main.py process-batch --split train

  # List available episodes
  python main.py list --verbose

  # Show episode info
  python main.py info "a16z_a16z Podcast A Podcast about Podcasts" --split train

Environment Variables:
  HF_TOKEN    Hugging Face authentication token (required for diarization)
        """
    )

    # Global arguments
    parser.add_argument(
        '--data-dir',
        default='data/PodcastFillers',
        help='Path to PodcastFillers dataset (default: data/PodcastFillers)'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Output directory for processed files (default: outputs)'
    )
    parser.add_argument(
        '--hf-token',
        help='Hugging Face authentication token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device for processing (default: auto)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # process-single command
    single_parser = subparsers.add_parser(
        'process-single',
        help='Process a single episode'
    )
    single_parser.add_argument(
        'episode',
        help='Episode name (without extension)'
    )
    single_parser.add_argument(
        '--split',
        choices=['train', 'validation', 'test'],
        default='train',
        help='Data split (default: train)'
    )
    single_parser.add_argument(
        '--num-speakers',
        type=int,
        help='Force specific number of speakers'
    )
    single_parser.set_defaults(func=process_single)

    # process-batch command
    batch_parser = subparsers.add_parser(
        'process-batch',
        help='Process multiple episodes in batch'
    )
    batch_parser.add_argument(
        '--split',
        choices=['train', 'validation', 'test'],
        default='train',
        help='Data split (default: train)'
    )
    batch_parser.add_argument(
        '--num-episodes',
        type=int,
        help='Maximum number of episodes to process'
    )
    batch_parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    batch_parser.set_defaults(func=process_batch)

    # list command
    list_parser = subparsers.add_parser(
        'list',
        help='List available episodes'
    )
    list_parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum episodes to show per split (default: 10)'
    )
    list_parser.set_defaults(func=list_episodes)

    # info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about an episode'
    )
    info_parser.add_argument(
        'episode',
        help='Episode name (without extension)'
    )
    info_parser.add_argument(
        '--split',
        choices=['train', 'validation', 'test'],
        default='train',
        help='Data split (default: train)'
    )
    info_parser.set_defaults(func=show_info)

    # summary command
    summary_parser = subparsers.add_parser(
        'summary',
        help='Show processing summary'
    )
    summary_parser.add_argument(
        '--split',
        choices=['train', 'validation', 'test'],
        default='train',
        help='Data split (default: train)'
    )
    summary_parser.set_defaults(func=show_summary)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate device
    if args.device == 'auto':
        args.device = None

    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
