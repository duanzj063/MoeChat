import requests
import json
import base64
import os
import io
from utilss import log as Log
from funasr import AutoModel
from modelscope import snapshot_download
import numpy as np
import soundfile as sf


class RemoteASR:
    """远程ASR服务调用类"""
    
    def __init__(self, api_url, api_key="", model="", language="zh"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.language = language
        
    def transcribe(self, audio_data):
        """
        调用远程ASR API进行语音识别
        
        Args:
            audio_data: 音频数据（numpy数组或文件路径）
            
        Returns:
            str: 识别结果文本
        """
        try:
            # 准备请求数据
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                # 如果是文件路径，读取文件
                with open(audio_data, 'rb') as f:
                    audio_bytes = f.read()
            elif isinstance(audio_data, np.ndarray):
                # 如果是numpy数组，转换为字节
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, 16000, format='WAV')
                audio_bytes = buffer.getvalue()
            else:
                # 假设是字节流
                audio_bytes = audio_data
            
            # 准备请求头
            headers = {
                'Content-Type': 'application/json',
            }
            
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            # 准备请求数据
            data = {
                'audio': base64.b64encode(audio_bytes).decode('utf-8'),
                'model': self.model,
                'language': self.language,
            }
            
            # 发送请求
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            else:
                Log.logger.error(f"远程ASR API调用失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            Log.logger.error(f"远程ASR调用异常: {str(e)}")
            return None


class LocalASR:
    """本地ASR模型类"""
    
    def __init__(self, model_path="./utilss/models/SenseVoiceSmall", device="cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """加载本地ASR模型"""
        try:
            self.model = AutoModel(
                model=self.model_path,
                disable_update=True,
                device=self.device,
            )
            Log.logger.info(f"本地ASR模型加载成功: {self.model_path}")
        except Exception as e:
            Log.logger.warning(f"本地ASR模型加载失败，尝试自动下载: {str(e)}")
            try:
                # 尝试自动下载模型
                model_dir = snapshot_download(
                    model_id="iic/SenseVoiceSmall",
                    local_dir=self.model_path,
                    revision="master"
                )
                self.model = AutoModel(
                    model=model_dir,
                    disable_update=True,
                    device="cpu",  # 回退到CPU
                )
                Log.logger.info("ASR模型自动下载并加载成功")
            except Exception as download_e:
                Log.logger.error(f"ASR模型下载失败: {str(download_e)}")
                raise
                
    def transcribe(self, audio_data):
        """
        使用本地模型进行语音识别
        
        Args:
            audio_data: 音频数据（numpy数组）
            
        Returns:
            str: 识别结果文本
        """
        try:
            if self.model is None:
                raise RuntimeError("ASR模型未加载")
                
            # 确保音频数据是numpy数组
            if isinstance(audio_data, str):
                # 如果是文件路径，读取文件
                import soundfile as sf
                audio_data, _ = sf.read(audio_data)
            
            # 进行语音识别
            rec_result = self.model.generate(
                input=audio_data,
                cache={},
                language="auto",  # 自动检测语言
                use_itn=True,
                batch_size_s=60,
            )
            
            # 提取识别结果
            if rec_result and len(rec_result) > 0:
                return rec_result[0]["text"]
            else:
                return ""
                
        except Exception as e:
            Log.logger.error(f"本地ASR识别异常: {str(e)}")
            return None


class ASRManager:
    """ASR管理器，统一管理本地和远程ASR"""
    
    def __init__(self, config):
        self.config = config
        self.asr = None
        self._init_asr()
        
    def _init_asr(self):
        """初始化ASR实例"""
        try:
            if self.config.get("use_remote", False):
                # 使用远程ASR
                self.asr = RemoteASR(
                    api_url=self.config.get("remote_api", ""),
                    api_key=self.config.get("api_key", ""),
                    model=self.config.get("model", ""),
                    language=self.config.get("language", "zh")
                )
                Log.logger.info("远程ASR服务初始化成功")
            else:
                # 使用本地ASR
                self.asr = LocalASR(
                    model_path=self.config.get("local_model_path", "./utilss/models/SenseVoiceSmall"),
                    device=self.config.get("device", "cuda:0")
                )
                Log.logger.info("本地ASR模型初始化成功")
        except Exception as e:
            Log.logger.error(f"ASR初始化失败: {str(e)}")
            # 尝试回退到本地模型
            try:
                self.asr = LocalASR(
                    model_path="./utilss/models/SenseVoiceSmall",
                    device="cpu"
                )
                Log.logger.info("ASR回退到本地CPU模式成功")
            except Exception as fallback_e:
                Log.logger.error(f"ASR回退也失败: {str(fallback_e)}")
                raise
                
    def transcribe(self, audio_data):
        """
        进行语音识别
        
        Args:
            audio_data: 音频数据（numpy数组或文件路径）
            
        Returns:
            str: 识别结果文本
        """
        if self.asr is None:
            Log.logger.error("ASR未初始化")
            return None
            
        return self.asr.transcribe(audio_data)
    
    def reload_config(self, config):
        """重新加载配置"""
        self.config = config
        self._init_asr()


# 全局ASR管理器实例
asr_manager = None


def init_asr(config):
    """初始化全局ASR管理器"""
    global asr_manager
    asr_manager = ASRManager(config)
    return asr_manager


def transcribe_audio(audio_data):
    """
    语音识别接口函数
    
    Args:
        audio_data: 音频数据
        
    Returns:
        str: 识别结果
    """
    global asr_manager
    if asr_manager is None:
        Log.logger.error("ASR管理器未初始化")
        return None
        
    return asr_manager.transcribe(audio_data)