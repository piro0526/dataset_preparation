# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository works with the **PodcastFillers Dataset**, a research dataset for filler word detection and classification in podcast audio. The dataset contains 199 full-length podcast episodes (145 hours total) with 85,803 manually annotated audio events, including ~35,000 filler words ("uh", "um", etc.) and ~50,000 non-filler events (breaths, music, laughter, etc.).

**Citation**: Ge Zhu, Juan-Pablo Caceres and Justin Salamon, "Filler Word Detection and Classification: A Dataset and Benchmark", INTERSPEECH 2022.

**License**: Dataset metadata is under Adobe's non-commercial research license. Audio files are CC-licensed (CC-BY-3.0, CC-BY-SA 3.0, CC-BY-ND-3.0) from SoundCloud.

## Python Environment

- **Python Version**: 3.12 (specified in `.python-version`)
- **Package Manager**: uv (based on `pyproject.toml`)
- **Project Name**: podcastfillers

### Common Commands

```bash
# Run the main script
uv run python main.py

# Install dependencies (when added to pyproject.toml)
uv sync

# Add a new dependency
uv add <package-name>

# Run Python scripts
uv run python <script-name>.py
```

## Dataset Structure

The dataset is organized into three main areas under `data/`:

### 1. Audio Files
- `audio/episode_mp3/` - Original full-length episodes (MP3, stereo, 44.1kHz, 32-bit)
- `audio/episode_wav/` - Preprocessed episodes (WAV, mono, 16kHz, 32-bit)
- `audio/clip_wav/` - 1-second preprocessed clips centered on annotated events

### 2. Metadata Files

**Main annotation file**:
- `metadata/PodcastFillers.csv` - All 85,803 manually annotated events with key columns:
  - `clip_name` - Filename of the 1-second audio clip (format: `{pfID}.wav`)
  - `pfID` - Unique PodcastFillers ID (0-85802)
  - `label_full_vocab` - Full vocabulary (13 classes: Uh, Um, You know, Like, Other, Laughter, Breath, Agree, Words, Repetitions, Overlap, Music, Noise)
  - `label_consolidated_vocab` - Consolidated vocabulary (6 classes: Uh, Um, Words, Breath, Laughter, Music) or "None" for excluded events
  - `podcast_filename` - Episode filename without extension
  - `event_start_inepisode`, `event_end_inepisode` - Event timing in full episode (seconds)
  - `event_start_inclip`, `event_end_inclip` - Event timing within 1-second clip (seconds)
  - `clip_start_inepisode`, `clip_end_inepisode` - Clip timing in full episode (seconds)
  - `duration` - Event duration in seconds (can exceed 1.0 for long events)
  - `confidence` - Annotator confidence (0-1 range)
  - `n_annotators` - Number of crowd annotators
  - `episode_split_subset` - Episode-level split (train/validation/test)
  - `clip_split_subset` - Clip-level split (train/validation/test/extra)
  - `pitch_cent` - Pitch in cents

**Per-episode files** (in subdirectories with train/validation/test splits):
- `metadata/episode_transcripts/` - Speech-to-text JSON files from SpeechMatics STT
  - Each word has: `confidence`, `duration`, `offset` (in microseconds), `text`
- `metadata/episode_annotations/` - Per-episode CSV files (same format as main CSV)
- `metadata/episode_vad/` - Voice activity detection predictions (timestamps and activations)
- `metadata/episode_sed_eval_paper/` - Ground truth and AVC-FillerNet predictions in sed_eval format

### 3. Data Splits

**Critical**: The dataset uses predefined train/validation/test splits to prevent:
- **Speaker leakage**: Episodes from the same podcast show stay in the same subset
- **Gender imbalance**: Each subset maintains gender balance

Always use the provided splits (indicated by `clip_split_subset` or folder structure) for ML experiments to ensure results are comparable to the FillerNet paper.

## Dataset Label Vocabularies

### Full Vocabulary (85,803 events)
**Fillers**: Uh (17,907), Um (17,078), You know (668), Like (157), Other (315)
**Non-fillers**: Words (12,709), Repetitions (9,024), Breath (8,288), Laughter (6,623), Music (5,060), Agree (3,755), Noise (2,735), Overlap (1,484)

### Consolidated Vocabulary (76,689 events)
Used for FillerNet training: Words (21,733), Uh (17,907), Um (17,078), Breath (8,288), Laughter (6,623), Music (5,060)

## Working with the Data

### Speech Transcript Format
Each JSON file in `episode_transcripts/` contains a list of word dictionaries:
```json
{
  "confidence": 0.99,
  "duration": 1799998,
  "offset": 31300002,
  "text": "Hi"
}
```
- `duration` and `offset` are in microseconds (1e-6 seconds)

### Audio Clip Naming Convention
Clips in the CSV use format: `{pfID}.wav` (e.g., `00000.wav`, `00001.wav`)
Actual audio files follow: `{pfID}-{label_full_vocab}.wav` (e.g., `00000-Agree.wav`, `00001-Music.wav`)

### Episode File Naming Convention
All files use: `[show name]_[episode name].[extension]`
- Example: `a16z_a16z Podcast Things Come Together -- Truths about Tech in Africa.json`

## Architecture Notes

- **Current State**: The repository is in early stages with only a basic `main.py` placeholder
- **Expected Development**: Machine learning models for filler word detection/classification
- **Data Loading**: Large JSON transcript files (>700KB) require careful memory management
- **File Handling**: Episode names contain special characters and spaces - always properly escape paths
