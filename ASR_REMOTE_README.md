# ASR远程配置使用说明

## 概述

MoeChat现在支持远程ASR（语音识别）服务，可以通过配置文件选择使用本地模型或远程API服务。

## 配置说明

### 1. 配置文件位置

编辑 `config.yaml` 文件，在 `ASR` 部分进行配置：

```yaml
ASR:
  use_remote: false          # 是否使用远程ASR服务
  remote_api: ""            # 远程ASR API地址
  api_key: ""               # 远程ASR API密钥
  model: ""                 # 远程ASR模型名称
  language: "zh"           # 语言设置
  local_model_path: "./utilss/models/SenseVoiceSmall"  # 本地模型路径
  device: "cuda:0"          # 设备设置
```

### 2. 配置项详解

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_remote` | 布尔值 | `false` | 是否使用远程ASR服务 |
| `remote_api` | 字符串 | `""` | 远程ASR服务的API地址 |
| `api_key` | 字符串 | `""` | 远程ASR服务的API密钥 |
| `model` | 字符串 | `""` | 远程ASR模型名称 |
| `language` | 字符串 | `"zh"` | 语音识别语言设置 |
| `local_model_path` | 字符串 | `"./utilss/models/SenseVoiceSmall"` | 本地模型路径 |
| `device` | 字符串 | `"cuda:0"` | 本地模型运行设备 |

## 使用方式

### 1. 使用本地ASR模型（默认）

```yaml
ASR:
  use_remote: false
  local_model_path: "./utilss/models/SenseVoiceSmall"
  device: "cuda:0"
```

### 2. 使用远程ASR服务

```yaml
ASR:
  use_remote: true
  remote_api: "http://your-asr-server.com/api/transcribe"
  api_key: "your-api-key"
  model: "whisper-1"
  language: "zh"
```

## 远程ASR API规范

### 请求格式

远程ASR服务需要支持以下HTTP POST请求格式：

```json
{
  "audio": "base64编码的音频数据",
  "model": "模型名称",
  "language": "语言代码"
}
```

### 请求头

```
Content-Type: application/json
Authorization: Bearer {api_key}
```

### 响应格式

```json
{
  "text": "识别结果文本"
}
```

## 支持的音频格式

- **格式**: WAV
- **采样率**: 16kHz
- **编码**: PCM
- **声道**: 单声道

## 故障排除

### 1. 远程ASR连接失败

检查以下几点：
- API地址是否正确
- API密钥是否有效
- 网络连接是否正常
- 远程服务是否正在运行

### 2. 本地ASR模型加载失败

- 检查模型文件是否存在
- 确保有足够的磁盘空间
- 检查CUDA是否可用（如果使用GPU）

### 3. 识别效果不佳

- 检查音频质量
- 确认语言设置正确
- 尝试调整模型参数

## 重新加载配置

修改配置文件后，可以通过以下方式重新加载ASR配置：

1. 重启整个服务
2. 调用 `reload_asr_config()` 函数（需要编程方式）

## 示例配置

### 完整配置示例

```yaml
# 使用OpenAI Whisper API
ASR:
  use_remote: true
  remote_api: "https://api.openai.com/v1/audio/transcriptions"
  api_key: "sk-your-openai-key"
  model: "whisper-1"
  language: "zh"

# 使用本地模型
ASR:
  use_remote: false
  local_model_path: "./utilss/models/SenseVoiceSmall"
  device: "cuda:0"
```

## 注意事项

1. **首次使用远程ASR**: 确保远程服务可用并且API密钥有效
2. **性能考虑**: 远程ASR可能会有网络延迟，本地模型响应更快
3. **成本考虑**: 某些远程ASR服务可能收费
4. **隐私考虑**: 远程服务会将音频数据发送到第三方服务器

## 技术支持

如果遇到问题，请检查：
1. 配置文件语法是否正确
2. 网络连接是否正常
3. 服务日志是否有错误信息
4. 远程API文档是否与实现匹配