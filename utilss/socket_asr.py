import socket
import threading
import struct
import json
from pysilero import VADIterator
import numpy as np
import base64
from scipy.signal import resample
from io import BytesIO
import soundfile as sf
from utilss import log as Log

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import time
from io import BytesIO

def asr(audio_data: bytes, asr_model: AutoModel):
    """
    语音识别函数，将音频数据转换为文本
    
    Args:
        audio_data (bytes): 音频数据，格式为WAV
        asr_model (AutoModel): 语音识别模型
        
    Returns:
        str: 识别出的文本，如果识别失败则返回None
    """
    # 记录开始时间
    tt = time.time()
    
    # 将音频数据转换为BytesIO对象，以便模型处理
    audio_data = BytesIO(audio_data)
    
    # 使用ASR模型进行语音识别
    res = asr_model.generate(
        input=audio_data,
        cache={},
        language="zh", # 设置识别语言为中文，可选："zh", "en", "yue", "ja", "ko", "nospeech"
        ban_emo_unk=True,  # 禁用情感和未知词
        use_itn=False,  # 不使用文本规范化
        disable_pbar=True,  # 禁用进度条
    )
    
    # 对识别结果进行后处理，去除空格
    text = str(rich_transcription_postprocess(res[0]["text"])).replace(" ", "")
    
    # 如果识别结果不为空，则返回结果
    if text:
        return text
    return None

def handle_client(client_socket: socket, asr_model: AutoModel):
    """
    处理客户端连接的函数，负责接收音频数据并进行语音识别
    
    Args:
        client_socket (socket): 客户端socket连接
        asr_model (AutoModel): 语音识别模型
    """
    # 创建语音活动检测(VAD)迭代器，设置语音填充时间为300ms
    vad_iterator = VADIterator(speech_pad_ms=300)
    current_speech = []  # 存储当前检测到的语音片段
    current_speech_tmp = []  # 临时存储音频样本
    status = False  # 标记是否正在接收语音
    
    # 循环接收客户端数据
    while True:
        # 接收完整消息
        data = rec(client_socket)
            
        if data is None:
            client_socket.close()
            Log.logger.info(f"客户端断开：{client_socket}")
            return
        
        # print(f"[客户端 {client_address}] {message}")
        data = json.loads(data)
        
        # 如果消息类型是语音识别(asr)
        if data["type"] == "asr":
            # 解码base64编码的音频数据
            audio_data = base64.urlsafe_b64decode(str(data["data"]).encode("utf-8"))
            # 将音频数据转换为numpy数组
            samples = np.frombuffer(audio_data, dtype=np.int16)
            # 将样本添加到临时列表
            current_speech_tmp.append(samples)
            
            # 如果临时样本数量少于4，继续接收
            if len(current_speech_tmp) < 4:
                continue
                
            # 合并临时样本
            resampled = np.concatenate(current_speech_tmp.copy())
            # 将样本值归一化到[-1, 1]范围并转换为float32类型
            resampled = (resampled / 32768.0).astype(np.float32)
            # 清空临时样本列表
            current_speech_tmp = []
            
            # 使用VAD迭代器处理音频数据
            for speech_dict, speech_samples in vad_iterator(resampled):
                # 如果检测到语音开始
                if "start" in speech_dict:
                    current_speech = []  # 清空当前语音片段
                    status = True  # 设置为正在接收语音状态
                    # print("开始说话")
                    pass
                    
                # 如果正在接收语音
                if status:
                    current_speech.append(speech_samples)
                else:
                    continue
                    
                # 检查是否是语音结束
                is_last = "end" in speech_dict
                if is_last:
                    # print("结束说话")
                    status = False  # 重置状态
                    # 合并所有语音片段
                    combined = np.concatenate(current_speech)
                    audio_bytes = b""
                    
                    # 将音频数据写入BytesIO对象
                    with BytesIO() as buffer:
                        sf.write(
                            buffer,
                            combined,
                            16000,  # 采样率
                            format="WAV",
                            subtype="PCM_16",  # 16位PCM格式
                        )
                        buffer.seek(0)
                        audio_bytes = buffer.read()  # 获取完整的WAV字节数据
                        
                        # 进行语音识别
                        res_text = asr(audio_bytes, asr_model)
                        if res_text:
                            # 将识别结果发送给客户端
                            try:
                                send(client_socket, res_text)
                            except:
                                # 如果发送失败，关闭客户端连接并返回
                                client_socket.close()
                                return
                    # 清空当前语音片段
                    current_speech = []

def send(sock, data):
    """
    发送消息函数，先发送消息长度，再发送消息内容
    
    Args:
        sock (socket): socket连接
        data (str): 要发送的数据
    """
    # 将数据编码为UTF-8字节
    data_bytes = data.encode('utf-8')
    # 计算数据长度
    length = len(data_bytes)
    
    # 发送长度信息（使用4字节无符号整数，网络字节序）
    sock.sendall(struct.pack('>I', length))
    # 发送实际数据
    sock.sendall(data_bytes)

def rec(sock):
    """
    接收消息函数，先读取消息长度，再读取对应长度的消息内容
    
    Args:
        sock (socket): socket连接
        
    Returns:
        str: 接收到的消息内容，如果连接关闭则返回None
    """
    # 先读取4字节的长度前缀
    length_bytes = sock.recv(4)
    if not length_bytes:
        return None  # 连接关闭
    
    # 解析长度（网络字节序转主机字节序）
    length = struct.unpack('>I', length_bytes)[0]
    
    # 循环读取直到获取完整数据
    data_bytes = b''
    while len(data_bytes) < length:
        # 每次最多读取剩余长度的数据
        remaining = length - len(data_bytes)
        chunk = sock.recv(min(remaining, 4096))  # 缓冲区设为4096字节
        if not chunk:
            return None  # 连接中断
        data_bytes += chunk
    
    # 将接收到的字节解码为UTF-8字符串
    return data_bytes.decode('utf-8')

def start_server(host: str, port: int, asr_model: AutoModel):
    """
    启动服务器函数，创建socket服务器并监听客户端连接
    
    Args:
        host (str): 服务器主机地址
        port (int): 服务器端口号
        asr_model (AutoModel): 语音识别模型
    """
    # 创建TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定地址和端口
    server_socket.bind((host, port))
    # 开始监听，最大连接数为5
    server_socket.listen(5)
    Log.logger.info(f"socket_asr服务启动，监听 {host}:{port}...")
    
    try:
        # 循环接受客户端连接
        while True:
            # 接受客户端连接
            client_socket, addr = server_socket.accept()
            Log.logger.info(f"新连接：{addr}")
            # 启动新线程处理客户端连接
            client_thread = threading.Thread(target=handle_client, args=(client_socket, asr_model, ))
            client_thread.start()
    except KeyboardInterrupt:
        # 如果用户按下Ctrl+C，关闭服务器
        Log.logger.info("服务器正在关闭...")
    finally:
        # 关闭服务器socket
        server_socket.close()

# if __name__ == "__main__":
#     start_server("0.0.0.0", 8002, asr_model)

# class ImprovedFullDuplexServer:
#     def __init__(self, host: str, port: int, asr_model: AutoModel, ):
#         self.host = host
#         self.port = port
#         self.server_socket = None
#         self.running = False
#         self.asr_model = asr_model
    
#     # asr功能
#     def asr(self, audio_data: bytes):
#         # global is_sv
#         # global sv_pipeline

#         tt = time.time()
#         # if is_sv:
#         #     if not sv_pipeline.check_speaker(audio_data):
#         #         return None

#         # with open(f"./tmp/{tt}.wav", "wb") as file:
#         #     file.write(audio_data)
#         audio_data = BytesIO(audio_data)
#         res = self.asr_model.generate(
#             input=audio_data,
#             # input=f"{model.model_path}/example/zh.mp3",
#             cache={},
#             language="zh", # "zh", "en", "yue", "ja", "ko", "nospeech"
#             ban_emo_unk=True,
#             use_itn=False,
#             # batch_size=200,
#         )
#         # print(f"{model.model_path}/example/zh.mp3",)
#         text = str(rich_transcription_postprocess(res[0]["text"])).replace(" ", "")
#         # text = res[0]["text"]
#         print()
#         print(f"[{time.time() - tt}]{text}\n\n")
#         if text:
#             return text
#         return None
        
#     def start_server(self):
#         try:
#             self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             self.server_socket.bind((self.host, self.port))
#             self.server_socket.listen(5)
#             self.running = True
            
#             print(f"服务器启动成功，监听 {self.host}:{self.port}")
            
#             while self.running:
#                 try:
#                     client_socket, client_address = self.server_socket.accept()
#                     print(f"客户端 {client_address} 已连接")
                    
#                     # 为每个客户端创建处理线程
#                     client_thread = threading.Thread(
#                         target=self.handle_client, 
#                         args=(client_socket, client_address)
#                     )
#                     client_thread.daemon = True
#                     client_thread.start()
                    
#                 except Exception as e:
#                     if self.running:
#                         print(f"接受连接时出错: {e}")
                        
#         except Exception as e:
#             print(f"服务器启动失败: {e}")
#         finally:
#             self.stop_server()
    
#     def send_message(self, client_socket, message):
#         """发送带长度前缀的消息"""
#         try:
#             # 将消息编码为字节
#             if isinstance(message, dict):
#                 message_bytes = json.dumps(message).encode('utf-8')
#             else:
#                 message_bytes = str(message).encode('utf-8')
            
#             # 计算消息长度
#             message_length = len(message_bytes)
            
#             # 发送长度信息（使用4字节整数）
#             length_prefix = struct.pack('!I', message_length)
#             client_socket.send(length_prefix)
            
#             # 发送实际消息内容
#             client_socket.send(message_bytes)
            
#             # print(f"发送消息: {message} (长度: {message_length} 字节)")
            
#         except Exception as e:
#             pass
#             # print(f"发送消息时出错: {e}")
    
#     def receive_message(self, client_socket):
#         """接收带长度前缀的消息"""
#         try:
#             # 首先接收4字节的长度信息
#             length_data = self._recv_exact(client_socket, 4)
#             if not length_data:
#                 return None
                
#             # 解析长度
#             message_length = struct.unpack('!I', length_data)[0]
#             # print(f"准备接收长度为 {message_length} 字节的消息")
            
#             # 根据长度接收完整消息
#             message_data = self._recv_exact(client_socket, message_length)
#             if not message_data:
#                 return None
                
#             # 解码消息
#             message = message_data.decode('utf-8')
#             data = json.loads(message)
#             return data
            
#         except Exception as e:
#             # print(f"接收消息时出错: {e}")
#             return None
    
#     def _recv_exact(self, sock, length):
#         """确保接收指定长度的数据"""
#         data = b''
#         while len(data) < length:
#             chunk = sock.recv(length - len(data))
#             if not chunk:
#                 return None
#             data += chunk
#         return data
    
#     def handle_client(self, client_socket, client_address):
#         """处理客户端连接"""
#         vad_iterator = VADIterator(speech_pad_ms=270)
#         current_speech = []
#         current_speech_tmp = []
#         status = False
#     # try:
#         while self.running:
#             # 接收客户端消息
#             data = self.receive_message(client_socket)
#             if data is None:
#                 break
#             # print(f"[客户端 {client_address}] {message}")
#             if data["type"] == "asr":
#                 audio_data = base64.urlsafe_b64decode(str(data["data"]).encode("utf-8"))
#                 samples = np.frombuffer(audio_data, dtype=np.float32)
#                 current_speech_tmp.append(samples)
#                 if len(current_speech_tmp) < 9:
#                     continue
#                 resampled = np.concatenate(current_speech_tmp.copy())
#                 current_speech_tmp = []
#                 # resampled = resample(samples, 1600)
#                 resampled = resample(resampled, 1600)
#                 resampled = resampled.astype(np.float32)
                
#                 for speech_dict, speech_samples in vad_iterator(resampled):
#                     if "start" in speech_dict:
#                         current_speech = []
#                         status = True
#                         pass
#                     if status:
#                         current_speech.append(speech_samples)
#                     else:
#                         continue
#                     is_last = "end" in speech_dict
#                     if is_last:
#                         status = False
#                         combined = np.concatenate(current_speech)
#                         audio_bytes = b""
#                         with BytesIO() as buffer:
#                             sf.write(
#                                 buffer,
#                                 combined,
#                                 16000,
#                                 format="WAV",
#                                 subtype="PCM_16",
#                             )
#                             buffer.seek(0)
#                             audio_bytes = buffer.read()  # 完整的 WAV bytes
#                             res_text = self.asr(audio_bytes)
#                             if res_text:
#                                 # await c_websocket.send_text(res_text)
#                                 self.send_message(client_socket, res_text)
#                         current_speech = []  # 清空当前段落
#                 # 发送回复消息
#                 # reply = f"服务器收到: {message}"
#                 # self.send_message(client_socket, reply)
                
#                 # if message.lower() == 'quit':
#                 #     break
                    
#         # except Exception as e:
#         #     print(f"处理客户端 {client_address} 时出错: {e}")
#         # finally:
#         #     client_socket.close()
#         #     print(f"客户端 {client_address} 连接已关闭")
    
#     def stop_server(self):
#         self.running = False
#         if self.server_socket:
#             self.server_socket.close()

# if __name__ == "__main__":
#     server = ImprovedFullDuplexServer()
#     try:
#         server.start_server()
#     except KeyboardInterrupt:
#         print("\n正在关闭服务器...")
