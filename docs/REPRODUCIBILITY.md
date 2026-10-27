# 运行和复现说明

基础流水线入口见项目README。研究脚本在本地Windows/Python环境开发，使用ffmpeg/ffprobe。部分脚本采用固定日期的实验目录，并依赖本地数据清单、先前结果及模型缓存；仅克隆本仓库不能直接重现全部历史数字。

## 代码入口

| 环节 | 脚本 |
|---|---|
| 数据整理/划分 | scripts/prepare_lavdf.py、prepare_lavdf_expansion.py |
| 情感与转写 | scripts/extract_lavdf_features.py |
| 情感差异 | scripts/interval_consistency.py |
| 同步提取 | scripts/run_syncnet_pilot.py、refine_syncnet_tracks.py、run_syncnet_batch400.py |
| 冻结与测试 | scripts/freeze_syncnet_protocol.py、run_syncnet_test80.py、evaluate_syncnet_test80.py |
| 模拟制作 | scripts/prepare_simulated40.py、prepare_simulated8.py、synthesize_simulated40.ps1、build_simulated40.py |
| 人工初审工具 | scripts/create_simulation_review.py、import_simulation_review.py |
| 模拟特征和评分 | scripts/extract_simulated48_features.py、extract_simulated48_sync.py、evaluate_simulated48.py |
| 语义与错误诊断 | scripts/run_simulated48_semantics.py、audit_simulated48_semantic_errors.py |

requirements.txt覆盖初始流水线，不是研究环境的完整锁文件。后续模块另使用scipy、soundfile、webrtcvad、FunASR/ModelScope和python_speech_features。建议分离基础环境与emotion2vec环境，先检查各脚本导入及其模型要求，不盲目覆盖已有环境。

复现需另行获得LAV-DF等数据，按脚本生成清单并检查哈希；历史固定选择和本地路径没有随仓库完整公开。模型下载与离线推理分开。HSEmotion、emotion2vec、Whisper、YuNet及零样本文本模型的权重均不入库。

SyncNet上游实现来自 https://github.com/joonson/syncnet_python ，仓库保留使用到的源码、MIT许可证与来源哈希（models/syncnet）。权重、官方示例不上传。相关模型与数据仍应分别遵守其条款。

模拟配音脚本使用Windows System.Speech及Microsoft Zira Desktop，须安装该声音。部分媒体处理使用Windows字体路径，其他系统需配置字体。浏览器冒烟检查依赖本机Edge和本地前端环境，未作为可移植测试上传。

## 测试

```powershell
python -m pytest tests -q
```

测试环境需具备相应模块依赖。单元测试验证数据处理和计算逻辑，不代表模型检测有效。模型训练与下载不应为了跑单元测试而自动启动。
