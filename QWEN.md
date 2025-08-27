# MoeChat 项目概述

## 项目简介

MoeChat 是一个强大的语音交互系统，专为自然对话和沉浸式角色扮演而设计。它利用 GPT-SoVITS 作为 TTS (Text-to-Speech) 模块，并集成了 ASR (Automatic Speech Recognition) 接口，后端使用 FunASR 进行语音识别。MoeChat 支持任何遵循 OpenAI 规范的 LLM API。

## 技术栈

- **主语言**: Python
- **主要框架/库**:
  - `FastAPI`: 用于构建 Web API。
  - `funasr`: 用于语音识别 (ASR)。
  - `torch`, `torchaudio`: 用于深度学习相关的音频处理。
  - `sounddevice`, `soundfile`: 用于音频播放和文件处理。
  - `numpy`, `scipy`: 用于科学计算。
  - `jionlp`: 用于自然语言处理相关任务。
  - `modelscope`: 用于模型加载和管理。
  - `faiss-cpu`: 用于向量相似度搜索。
  - `ruamel.yaml`: 用于 YAML 配置文件解析。
  - `uvicorn`: 用于运行 FastAPI 应用。
  - `pysilero`: 用于语音活动检测 (VAD)。
- **依赖管理**: `requirements.txt`
- **配置管理**: `config.yaml` (YAML 格式)

## 核心功能

- **语音合成 (TTS)**: 使用 GPT-SoVITS 通过 API 接口生成语音。
- **语音识别 (ASR)**: 集成本地 FunASR 模型或远程 ASR API 进行语音转文字。
- **大语言模型 (LLM) 集成**: 支持任何 OpenAI 兼容的 LLM API，用于生成对话内容。
- **角色模板系统**: 可以通过内置提示词模板创建和定制 AI 角色。
- **长期记忆 (日记系统)**: 记录完整的对话历史，并支持基于模糊时间表达式（如“昨天”、“上周”）的精确时间查询。
- **核心记忆**: 存储关于用户的关键信息、偏好和重要回忆，使用语义匹配进行检索。
- **知识库 (世界书)**: 为 LLM 提供额外的知识，如人物、物品、事件等，以增强角色扮演的一致性和能力。
- **情绪系统**: 根据对话内容影响 AI 的情绪状态，并可动态选择参考音频。
- **网络图片搜索与本地图片分享**。
- **表情包发送**。
- **财务系统 (复式记账)**。

## 项目架构

项目主要由以下几个部分组成：

1.  **`chat_server.py`**: 核心服务器，提供 FastAPI 接口。
    -   `/api/chat` 和 `/api/chat_v2`: 主要的聊天接口，接收用户消息，调用 LLM 生成回复，并通过 GSV TTS API 生成语音，以流式方式返回结果。
    -   `/api/asr`: 接收 Base64 编码的音频数据，使用 FunASR 进行语音识别，返回文本。
    -   `/api/asr_ws`: WebSocket 接口，用于实时 VAD 和 ASR。
    -   `/api/get_context`, `/api/get_config`, `/api/update_config`: 用于获取和更新聊天上下文及服务器配置。
2.  **`utilss/agent.py`**: Agent 模块，负责管理角色模板、上下文、长期记忆、核心记忆和知识库。`Agent` 类是核心，它整合了这些功能来构建发送给 LLM 的完整提示词 (Prompt)。
3.  **`utilss/long_mem.py`**: 实现长期记忆 (日记系统) 的存储和检索。
4.  **`utilss/core_mem.py`**: 实现核心记忆的存储和检索。
5.  **`utilss/data_base.py`**: 实现知识库 (世界书) 的管理和检索。
6.  **`utilss/config.py`**: 配置文件 (`config.yaml`) 的加载和管理。
7.  **`config.yaml`**: 主配置文件，包含了 LLM API、GSV TTS API、ASR 设置、Agent 角色设定、记忆和知识库开关及阈值等所有配置项。
8.  **`emotion_engine.py`**: 处理情绪系统的逻辑。
9.  **客户端**:
    -   `client_cli.py`: 一个简单的命令行客户端。
    -   `client-gui/`: 图形用户界面客户端 (未在当前目录列出，但 README 中提及)。
    -   `chat_server_with_web.py`: 可能是集成了 Web 客户端的服务器版本。

## 构建、运行与测试

### 环境准备

1.  **Python 版本**: 建议使用 Python 3.10 (如 README 中 Linux 部分所述)。
2.  **依赖安装**: 使用 `pip` 安装 `requirements.txt` 中列出的依赖包。
    ```bash
    pip install -r requirements.txt
    ```
3.  **GPT-SoVITS**: 需要单独设置 GPT-SoVITS 服务。可以将其与 MoeChat 目录并排放置，并按照其文档启动 API 服务 (`api_v2.py`)。
4.  **模型**: ASR 模型 (SenseVoiceSmall) 会在首次运行时自动下载到 `./utilss/models/SenseVoiceSmall`。

### 运行服务器

在 MoeChat 项目根目录下执行：

```bash
# 假设 GPT-SoVITS 的 Python 环境已正确设置
python chat_server.py
```
服务器默认会启动在 `0.0.0.0:8001`。

### 运行客户端

#### 命令行客户端

```bash
python client_cli.py
```

#### GUI 客户端

根据 README，对于 Windows:
```bash
GPT-SoVITS-version_name\runtime\python.exe client-gui\src\client_gui.py
```
对于 Linux:
```bash
python client-gui\src\client_gui.py
```

### 配置

所有配置项均在 `config.yaml` 文件中定义。关键配置包括：

- **LLM**: `api` (LLM API 地址), `key` (API 密钥), `model` (模型名称)。
- **GSV**: `api` (GPT-SoVITS TTS API 地址), `ref_audio_path` (默认参考音频路径), `prompt_text` (默认参考音频文本)。
- **Agent**: `is_up` (是否启用角色模板), `char` (角色名), `user` (用户名), `long_memory`, `is_core_mem`, `lore_books` (分别控制日记、核心记忆、知识库的开关)。
- **ASR**: `use_remote` (是否使用远程 ASR), `local_model_path` (本地 ASR 模型路径)。

## 开发规范

- **代码结构**: 核心逻辑在 `chat_server.py` 和 `utilss/` 目录下的模块中。
- **配置管理**: 使用 `config.yaml` 和 `utilss/config.py` 进行统一配置管理。
- **并发处理**: 使用 `threading` 和 `asyncio` 处理 ASR、LLM、TTS 等耗时操作，以避免阻塞主进程。
- **日志**: 使用 `utilss/log.py` 进行日志记录。