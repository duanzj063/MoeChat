# Qwen Code 上下文 (MoeChat 项目)

## 项目概览

这是一个名为 **MoeChat** 的 Python 项目。它是一个功能强大的**语音交互系统**，旨在实现与 AI 角色的自然对话和沉浸式角色扮演。

核心功能包括：
*   **语音合成 (TTS)**：使用 GPT-SoVITS 作为后端。
*   **语音识别 (ASR)**：集成 FunASR 作为本地 ASR 引擎，并支持远程 ASR API。
*   **大型语言模型 (LLM)**：兼容任何遵循 OpenAI 规范的 LLM API。
*   **高级记忆系统**：
    *   **短期记忆 (上下文)**：维持对话连贯性。
    *   **长期记忆 (日记系统)**：存储完整对话历史，支持基于时间（如“昨天”、“上周”）的精确查询。
    *   **核心记忆**：存储关于用户的关键事实和偏好，使用语义匹配。
    *   **世界书 (知识库)**：为角色扮演提供背景知识。
*   **情绪系统**：AI 角色具有可计算的情绪状态，影响对话和 TTS。
*   **角色模板**：使用预定义模板快速创建和定制 AI 角色。
*   **客户端**：提供 CLI 和 GUI (Flet) 客户端，以及一个 Web 客户端。
*   **附加功能**：表情包发送、本地图片分享、天气查询、简单的财务系统等。

## 技术栈

*   **后端/核心**: Python 3, FastAPI, Uvicorn
*   **AI/ML**: PyTorch, FunASR, SenseVoiceSmall (ASR), FAISS (向量检索)
*   **TTS**: GPT-SoVITS (外部 API)
*   **前端/客户端**: Python (CLI), Flet (GUI), JavaScript (Web 客户端)
*   **数据处理**: NumPy, SoundFile, jionlp
*   **配置**: YAML
*   **依赖管理**: pip, requirements.txt

## 项目结构

*   `chat_server.py`: 主服务器启动文件，提供 ASR、聊天等核心 API。
*   `chat_core.py`: 聊天核心逻辑，处理 LLM 交互、TTS 调用、记忆系统集成。
*   `chat_server_with_web.py`: 启动包含 Web 客户端的服务器。
*   `client_cli.py`: 命令行客户端。
*   `client-gui/`: Flet 图形用户界面客户端。
*   `web/`: Web 客户端前端代码。
*   `utilss/`: 核心工具库，包含配置、代理、记忆系统 (long_mem, core_mem)、ASR 封装、提示词模板 (prompt.py) 等。
*   `emotion/`: 情绪引擎实现。
*   `plugins/`: 可扩展插件目录 (如财务系统)。
*   `data/`: 存放角色数据、记忆文件、知识库等运行时数据。
*   `config.yaml`: 主配置文件。
*   `requirements.txt`: 服务端依赖。
*   `client-requirements.txt`: 客户端依赖。

## 配置 (`config.yaml`)

项目使用 `config.yaml` 进行配置。关键部分包括：
*   `LLM`: LLM API 端点、密钥、模型名称和额外参数。
*   `GSV`: GPT-SoVITS TTS API 端点和参数。
*   `ASR`: ASR 设置（本地或远程）。
*   `Agent`: 角色模板、记忆系统 (日记、核心记忆、世界书) 的开关和参数。
*   `Core`: 核心设置，如声纹识别 (SV)。
*   `extra_ref_audio`: 根据情绪标签动态选择 TTS 参考音频。

## 运行与构建

### 环境准备
*   安装 Python 3.10 (推荐)。
*   使用 `pip install -r requirements.txt` 安装服务端依赖。
*   使用 `pip install -r client-requirements.txt` 安装客户端依赖。
*   (可选，若使用本地 ASR) 脚本可能自动下载 `SenseVoiceSmall` 模型。
*   (可选，若使用声纹识别) 需要 `test.wav` 文件。

### 启动服务端
1.  确保 GPT-SoVITS 服务已启动。
2.  在项目根目录运行 `python chat_server.py` 启动 MoeChat 核心服务。
    *   或运行 `python chat_server_with_web.py` 启动包含 Web 客户端的服务。

### 启动客户端
*   **CLI**: 在项目根目录运行 `python client_cli.py`。
*   **GUI**: 在项目根目录运行 `python client-gui/src/client_gui.py`。
*   **Web**: 访问 `chat_server_with_web.py` 启动的服务对应的 Web 地址。

## 代码细节

*   **记忆系统**: `utilss/agent.py` (Agent 类) 是记忆系统的协调中心，管理短期上下文并调用 `utilss/long_mem.py` (日记)、`utilss/core_mem.py` (核心记忆)、`utilss/data_base.py` (世界书)。这些模块分别处理各自类型的记忆存储、检索和更新。
*   **情绪系统**: `emotion_engine.py` 实现了情绪计算逻辑，情绪状态存储在 `emotion_state.json` 中。
*   **提示词**: `utilss/prompt.py` 定义了各种系统提示词模板，结合 `config.yaml` 中的设定动态生成。
*   **API**: 主要后端 API 由 `chat_server.py` 中的 FastAPI 路由定义。

## 开发约定

*   Python 项目，遵循标准 Python 模块和包结构。
*   配置驱动，主要通过 `config.yaml` 控制行为。
*   模块化设计，核心逻辑分布在 `utilss` 和其他功能目录下。
*   使用 FastAPI 构建异步 Web 服务。
*   内存系统通过后台线程更新，以避免阻塞主对话流程。
