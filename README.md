# DeepFake Emotion Detector

基于人脸表情、语音情感和文本情感跨模态一致性的深度伪造风险检测原型。输入一个 MP4 视频，系统会自动提取音频与视频帧、完成语音转写和三模态情感识别、按时间轴对齐结果，并生成 JSON 检测报告。

> 本项目输出的是基于情感不一致性的辅助风险指标，不是经过司法鉴定或生产环境校准的通用深伪检测器。低一致性也可能来自表演、反讽、噪声、遮挡或模型误判。

## 完整流程

```text
MP4 视频
├─ FFmpeg → 16 kHz 单声道 WAV → Whisper tiny → 分段转写
│                              ├─ wav2vec2 → 语音情感
│                              └─ mDeBERTa → 文本情感
└─ OpenCV → 2 FPS 视频帧 → 人脸检测 → HSEmotion → 面部情感
                                      ↓
                         0.5 秒时间轴对齐与一致性计算
                                      ↓
                           风险分数、real/fake、JSON 报告
```

当前风险分数定义为 `1 - 平均一致性`。三组模态两两比较，相同记为 1、不同记为 0、未知模态不参与该组比较；默认风险分数大于等于 `0.5` 时预测为 `fake`。阈值可通过命令行调整。

## 环境与安装

- Python 3.10（项目当前在 Python 3.10.9、Windows 11、CPU 环境验证）
- FFmpeg，且 `ffmpeg` 命令已加入 `PATH`
- 首次运行需要联网下载模型，模型体积较大，CPU 推理可能耗时较长

Windows PowerShell：

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ffmpeg -version
```

Linux/macOS 激活虚拟环境时使用 `source .venv/bin/activate`。FFmpeg 请通过系统包管理器安装。

## 一键运行

在项目根目录执行：

```powershell
python scripts/run_pipeline.py --input video.mp4
```

指定中文转写、报告位置和判定阈值：

```powershell
python scripts/run_pipeline.py `
  --input data/raw/demo.mp4 `
  --language zh `
  --output data/results/demo_report.json `
  --fake-threshold 0.5
```

默认中间产物写入 `data/processed/pipeline/<视频名>/`，最终报告写入 `data/results/<视频名>_report.json`。完整参数可运行：

```powershell
python scripts/run_pipeline.py --help
```

## 模型说明

| 阶段 | 默认实现/模型 | 输出标签或结果 |
|---|---|---|
| 音频提取 | FFmpeg | 16 kHz、单声道、PCM WAV |
| 视频帧 | OpenCV | 默认每秒 2 帧 JPEG |
| 语音转写 | OpenAI Whisper `tiny` | 语言、全文、带起止时间的分段 |
| 面部情感 | HSEmotion `enet_b0_8_best_afew` | 帧级情感与置信度 |
| 语音情感 | `superb/wav2vec2-base-superb-er` | 默认 3 秒窗的情感与分数 |
| 文本情感 | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | 分段级零样本情感与分数 |
| 对齐 | 最近邻时间对齐 | face/speech/text 与一致性 |

统一情感空间主要包含 `happy`、`neutral`、`sad`、`fear`、`anger` 和 `surprise`。模型由各自库在首次使用时下载；Whisper 缓存于 `models/whisper/`，Hugging Face/HSEmotion 权重使用其默认缓存位置。

## JSON 输出格式

报告使用 UTF-8 编码并保留中文，主要字段如下：

```json
{
  "schema_version": "1.0",
  "video": {
    "path": "D:/.../demo.mp4",
    "filename": "demo.mp4",
    "size_bytes": 123456,
    "duration_seconds": 12.4,
    "fps": 25.0,
    "frame_count": 310,
    "width": 1920,
    "height": 1080
  },
  "main_emotions": {
    "face": {"emotion": "neutral", "proportion": 0.72, "samples": 25},
    "speech": {"emotion": "sad", "proportion": 0.5, "samples": 5},
    "text": {"emotion": "neutral", "proportion": 0.67, "samples": 3}
  },
  "average_consistency": 0.64,
  "inconsistent_segments": [
    {"start": 3.0, "end": 5.5, "average_consistency": 0.333333, "samples": 5}
  ],
  "deepfake_risk_score": 0.36,
  "prediction": "real",
  "thresholds": {"fake": 0.5, "inconsistency": 1.0},
  "timings_seconds": {
    "audio_extraction": 0.8,
    "frame_extraction": 1.2,
    "speech_transcription": 8.4,
    "face_emotion": 4.1,
    "speech_emotion": 2.7,
    "text_emotion": 3.5,
    "alignment": 0.01,
    "total": 20.8
  },
  "artifacts": {
    "audio": ".../audio/demo.wav",
    "frames": ".../frames",
    "transcript": ".../transcript/demo.json",
    "face_emotion": ".../emotions/face.csv",
    "speech_emotion": ".../emotions/speech.csv",
    "text_emotion": ".../emotions/text.csv",
    "alignment": ".../alignment.csv"
  }
}
```

`inconsistent_segments` 会把连续低于 `--inconsistency-threshold`（默认 1.0）的对齐点合并为时间段。`artifacts` 提供各阶段产物路径，便于复核具体预测。

## 项目结构

```text
DeepFakeEmotionDetector/
├── data/
│   ├── raw/                         # 批处理脚本默认视频输入
│   ├── processed/                   # 音频、帧、转写、情感及对齐中间结果
│   └── results/                     # 一键流水线 JSON 报告
├── models/
│   ├── whisper/                     # Whisper 权重缓存（首次运行生成）
│   ├── download.py
│   └── README.md
├── scripts/
│   ├── run_pipeline.py              # 单视频一键入口
│   ├── extract_audio.py
│   ├── extract_frames.py
│   ├── speech_to_text.py
│   ├── face_emotion.py
│   ├── speech_emotion.py
│   ├── text_sentiment.py
│   └── alignment.py
├── tests/
├── web/                             # Web 界面预留目录
├── requirements.txt
└── README.md
```

## 分阶段运行

各脚本也支持目录级批处理，默认在 `data/raw` 和 `data/processed` 下读取与写入文件：

```powershell
python scripts/extract_audio.py --input data/raw --output data/processed/audio
python scripts/extract_frames.py --input data/raw --output data/processed/frames --fps 2
python scripts/speech_to_text.py --input data/processed/audio --language zh
python scripts/face_emotion.py
python scripts/speech_emotion.py
python scripts/text_sentiment.py
python scripts/alignment.py --interval 0.5 --face-fps 2
```

每个命令均可通过 `--help` 查看输入、输出和模型参数。中间 CSV 使用带 BOM 的 UTF-8，JSON 和 README 使用标准 UTF-8，以避免 Windows/Excel 环境中的中文乱码。

## 测试

```powershell
python -m pytest
```
