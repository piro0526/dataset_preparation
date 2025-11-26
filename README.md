# PodcastFillers Speaker Diarization for Moshi Fine-tuning

PodcastFillersデータセットを使用して、Moshiファインチューニング用の話者分離ステレオ音声ファイルを生成するツール。

## 概要

このプロジェクトは、複数話者のポッドキャスト音声から以下を生成します：

- **ステレオWAVファイル** (24kHz, 2ch)
  - 左チャンネル: AI応答（Moshi出力想定）
  - 右チャンネル: ユーザー入力
- **メタデータJSON**: 話者セグメント情報、ロールマッピング
- **CSVファイル**: 詳細な話者分離結果

## 主な機能

- ✅ pyannote.audioによる高精度な話者分離
- ✅ 自動話者ロール割り当て（AI/ユーザー）
- ✅ 16kHz → 24kHzへの自動リサンプリング
- ✅ バッチ処理対応（199エピソード一括処理）
- ✅ GPU/CPU自動選択
- ✅ CLIとPython APIの両対応

## セットアップ

### 1. 依存関係のインストール

```bash
# uvを使用（推奨）
uv sync

# または pip
pip install -e .
```

### 2. Hugging Faceトークンの設定

pyannote.audioの事前学習モデルを使用するために必要：

1. https://huggingface.co/settings/tokens でトークンを生成
2. https://huggingface.co/pyannote/speaker-diarization-3.1 で利用規約に同意
3. 環境変数を設定：

```bash
export HF_TOKEN="your_hf_token_here"
```

## 使い方

### コマンドライン（CLI）

#### 単一エピソードの処理

```bash
python main.py process-single "a16z_a16z Podcast A Podcast about Podcasts" --split train
```

#### バッチ処理（全エピソード）

```bash
# trainセット全体を処理
python main.py process-batch --split train

# 最初の10エピソードのみ
python main.py process-batch --split train --num-episodes 10
```

#### エピソード一覧の表示

```bash
python main.py list --verbose
```

#### エピソード情報の確認

```bash
python main.py info "a16z_a16z Podcast A Podcast about Podcasts" --split train
```

#### 処理サマリーの表示

```bash
python main.py summary --split train
```

### Python API

```python
from src import DiarizationPipeline

# パイプラインの初期化
pipeline = DiarizationPipeline(
    data_dir='data',
    output_dir='outputs',
    hf_token='your_hf_token',
    device='auto'  # 'cuda' or 'cpu'
)

# 単一エピソードの処理
result = pipeline.process_episode(
    episode_name='a16z_a16z Podcast A Podcast about Podcasts',
    split='train'
)

# バッチ処理
results = pipeline.batch_process(
    split='train',
    num_episodes=10
)
```

### Jupyter Notebook

テストノートブックを使用：

```bash
uv run jupyter notebook notebooks/01_test_pipeline.ipynb
```

## プロジェクト構造

```
PodcastFillers/
├── src/
│   ├── data_loader.py          # データセット読み込み
│   ├── speaker_diarization.py  # pyannote.audio話者分離
│   ├── audio_separator.py      # 話者ごとの音声分離
│   ├── stereo_generator.py     # ステレオ音声生成（24kHz）
│   ├── pipeline.py             # 統合パイプライン
│   └── utils.py                # ユーティリティ関数
├── outputs/
│   ├── train/                  # 処理済み学習データ
│   │   ├── episode_name/
│   │   │   ├── stereo_24k.wav  # Moshi用ステレオ音声
│   │   │   ├── stereo_24k.json # メタデータ
│   │   │   └── stereo_24k.csv  # 話者分離結果
│   ├── validation/
│   └── test/
├── configs/
│   ├── default.yaml            # デフォルト設定
│   └── moshi_finetune.yaml     # Moshi用設定
├── notebooks/
│   └── 01_test_pipeline.ipynb  # テストノートブック
├── data/                       # PodcastFillersデータセット
├── main.py                     # CLIエントリーポイント
├── pyproject.toml
└── README.md
```

## 出力形式

### ステレオWAVファイル (`stereo_24k.wav`)

- **フォーマット**: WAV (PCM_16)
- **サンプリングレート**: 24,000 Hz（Moshi要件）
- **チャンネル**: 2（ステレオ）
  - **左チャンネル**: AI応答（最も発話時間の長い話者）
  - **右チャンネル**: ユーザー入力（2番目に長い話者）

### メタデータJSON (`stereo_24k.json`)

```json
{
  "episode_name": "...",
  "sample_rate": 24000,
  "channels": 2,
  "duration_seconds": 1993.0,
  "channel_mapping": {
    "left": "ai",
    "right": "user"
  },
  "speaker_statistics": {
    "SPEAKER_00": {
      "total_time": 1200.5,
      "num_segments": 145,
      "speaking_ratio": 0.602
    }
  },
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker": "SPEAKER_00",
      "role": "ai"
    }
  ]
}
```

### CSVファイル (`stereo_24k.csv`)

| start | end  | duration | speaker_id | role | confidence |
|-------|------|----------|------------|------|------------|
| 0.5   | 3.2  | 2.7      | SPEAKER_00 | ai   | 1.0        |
| 3.5   | 7.8  | 4.3      | SPEAKER_01 | user | 1.0        |

## 設定

設定ファイル（YAML）でパラメータをカスタマイズ可能：

### `configs/moshi_finetune.yaml`

```yaml
# Moshi最適化設定
target_sample_rate: 24000  # Moshi要件
min_speakers: 2
max_speakers: 2
fade_duration: 0.005
normalize_audio: true
```

## 処理時間の目安

| エピソード長 | GPU (CUDA) | CPU     |
|--------------|------------|---------|
| 10分         | 1-2分      | 3-5分   |
| 45分（平均） | 3-5分      | 7-13分  |
| 全199件      | 10-17時間  | 23-43時間 |

**推奨**: GPU環境での処理（CUDA対応）

## トラブルシューティング

### HF_TOKENエラー

```
ValueError: Hugging Face token is required
```

→ 環境変数 `HF_TOKEN` を設定してください

### メモリ不足

長いエピソード（1時間以上）でメモリ不足になる場合：

```python
# 設定でバッチサイズを調整
config = {'batch_size': 1}
pipeline = DiarizationPipeline(config=config)
```

### CUDA out of memory

GPU使用時にメモリ不足の場合、CPUに切り替え：

```bash
python main.py process-single "episode_name" --device cpu
```

## パフォーマンス最適化

### GPU使用

```bash
# CUDAが利用可能か確認
python -c "import torch; print(torch.cuda.is_available())"

# GPU強制指定
python main.py process-batch --device cuda
```

### 並列処理

```python
# config内でnum_workersを設定
config = {'num_workers': 4}
pipeline = DiarizationPipeline(config=config)
```

## 技術スタック

- **話者分離**: [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio)
- **音声処理**: librosa, soundfile, scipy
- **深層学習**: PyTorch, torchaudio
- **データ処理**: pandas, numpy

## 制限事項

- 話者が3人以上の場合、最も発話時間の長い2人のみを使用
- 話者の重複（オーバーラップ）は混合処理
- 既存の転写データに話者情報がないため、完全自動分離に依存

## ライセンス

- **このツール**: MIT License
- **PodcastFillersデータセット**: Adobe非商用研究ライセンス
- **音声ファイル**: CC-BY-3.0, CC-BY-SA 3.0, CC-BY-ND-3.0

## 引用

PodcastFillersデータセットを使用する場合：

```bibtex
@inproceedings{Zhu:FillerWords:INTERSPEECH:22,
  title = {Filler Word Detection and Classification: A Dataset and Benchmark},
  author = {Zhu, Ge and Caceres, Juan-Pablo and Salamon, Justin},
  booktitle = {23rd Annual Cong. of the Int. Speech Communication Association (INTERSPEECH)},
  address = {Incheon, Korea},
  month = {Sep.},
  year = {2022},
  url = {https://arxiv.org/abs/2203.15135}
}
```

## 参考資料

- [PodcastFillers Dataset](https://podcastfillers.github.io/)
- [pyannote.audio Documentation](https://github.com/pyannote/pyannote-audio)
- [Moshi: Real-time Speech-Text Foundation Model](https://kyutai.org/Moshi.pdf)
- [Moshi Fine-tuning Repository](https://github.com/kyutai-labs/moshi-finetune)

## サポート

問題が発生した場合は、GitHubのIssueを作成してください。
