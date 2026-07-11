# DeepFake Emotion Detector

基于跨模态情感一致性的深度伪造诈骗视频检测系统。

## 项目简介

本项目面向 AI 换脸、AI 拟声等诈骗视频，计划分别提取人脸表情情感、语音情感和文本情感，并通过三种模态之间的一致性评估视频的深度伪造风险。

当前版本仅完成项目初始化和模块接口预留，尚未实现或集成任何检测模型。

## 环境要求

- Windows 11
- Python 3.10.9
- CPU 环境
- Python 虚拟环境 `.venv`
- 后续音视频处理可能需要单独安装 FFmpeg 并配置到 `PATH`

## 项目结构

```text
DeepFakeEmotionDetector/
├── data/
│   ├── videos/          # 待检测视频
│   ├── audio/           # 提取的音频
│   ├── frames/          # 提取的视频帧
│   └── results/         # 分析结果
├── models/              # 模型文件（不提交到 Git）
├── scripts/
│   ├── extract_audio.py
│   ├── extract_frames.py
│   ├── face_emotion.py
│   ├── speech_emotion.py
│   ├── speech_to_text.py
│   ├── text_sentiment.py
│   └── alignment.py
├── web/                 # 后续 Web 界面
├── tests/               # 自动化测试
├── requirements.txt
└── README.md
```

## 初始化环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 模块规划

- `extract_audio.py`：从视频中提取音频。
- `extract_frames.py`：按指定采样间隔提取视频帧。
- `face_emotion.py`：分析视频帧中的人脸表情情感。
- `speech_emotion.py`：分析音频中的语音情感。
- `speech_to_text.py`：将语音转写为文本。
- `text_sentiment.py`：分析转写文本的情感。
- `alignment.py`：计算多模态情感一致性及风险分数。

各脚本目前提供稳定、可扩展的函数接口和命令行入口；模型实现将在后续阶段逐步加入。

## 运行模板

每个模块均可通过 `--help` 查看预留的命令行接口，例如：

```powershell
python scripts\extract_audio.py --help
python scripts\alignment.py --help
```
