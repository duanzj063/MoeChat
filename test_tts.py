#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS API测试脚本
用于测试GPT-SoVITS TTS API的连通性和功能
"""

import requests
import json
import time
import os
import numpy as np
import wave
from pathlib import Path

def create_test_audio_file():
    """创建一个简单的测试音频文件"""
    try:
        # 创建5秒的音频（符合3-10秒要求）
        sample_rate = 22050
        duration = 5.0  # 5秒
        samples = int(sample_rate * duration)
        
        # 创建一个简单的正弦波音频而不是静音
        frequency = 440  # A4音符
        t = np.linspace(0, duration, samples, False)
        audio_data = (np.sin(2 * np.pi * frequency * t) * 0.3 * 32767).astype(np.int16)
        
        # 使用绝对路径，放在GPT-SoVITS目录中
        api_dir = r"d:\code\pythonWork\MoeChat\GPT-SoVITS-v2pro-20250604-nvidia50"
        test_audio_path = os.path.join(api_dir, "test_ref_audio.wav")
        
        with wave.open(test_audio_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"✅ 测试音频文件已创建: {test_audio_path} (时长: {duration}秒)")
        return test_audio_path
    except ImportError:
        print("⚠️ numpy或wave模块不可用，将使用空路径测试")
        return ""
    except Exception as e:
        print(f"❌ 音频文件创建失败: {e}")
        return ""

def test_tts_api():
    """测试TTS API功能"""
    api_url = "http://127.0.0.1:9880"
    
    print("=== TTS API 测试开始 ===")
    
    # 1. 测试API连通性
    print("\n1. 测试API连通性...")
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        print(f"✅ API连通性测试成功，状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ API连通性测试失败: {e}")
        return False
    
    # 2. 创建测试音频文件
    print("\n2. 创建测试音频文件...")
    test_audio_path = create_test_audio_file()
    
    # 3. 测试TTS合成功能
    print("\n3. 测试TTS合成功能...")
    
    # 准备测试数据
    test_data = {
        "text": "你好，这是一个TTS测试。",
        "text_lang": "zh",
        "ref_audio_path": test_audio_path,  # 使用创建的测试音频
        "prompt_text": "我是测试语音。",
        "prompt_lang": "zh",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut0",
        "batch_size": 20,
        "speed_factor": 1.0,
        "seed": -1
    }
    
    try:
        # 发送TTS请求
        response = requests.post(
            f"{api_url}/tts",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ TTS合成请求成功")
            
            # 检查响应内容类型
            content_type = response.headers.get('content-type', '')
            print(f"响应内容类型: {content_type}")
            
            if 'audio' in content_type:
                # 保存音频文件
                output_dir = Path("test_output")
                output_dir.mkdir(exist_ok=True)
                
                audio_file = output_dir / f"test_tts_{int(time.time())}.wav"
                with open(audio_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ 音频文件已保存: {audio_file}")
                print(f"文件大小: {len(response.content)} 字节")
                
                return True
            else:
                print(f"❌ 响应不是音频格式: {content_type}")
                print(f"响应内容: {response.text[:200]}...")
                return False
        else:
            print(f"❌ TTS合成失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            print(f"请求数据: {json.dumps(test_data, ensure_ascii=False, indent=2)}")
            return False
            
    except Exception as e:
        print(f"❌ TTS合成测试失败: {e}")
        return False

def test_config_integration():
    """测试与MoeChat配置的集成"""
    print("\n4. 测试配置集成...")
    
    try:
        # 读取配置文件
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        gsv_config = config.get('GSV', {})
        api_url = gsv_config.get('api', '')
        
        if api_url == "http://127.0.0.1:9880":
            print("✅ 配置文件中的API地址正确")
            return True
        else:
            print(f"❌ 配置文件中的API地址不正确: {api_url}")
            return False
            
    except Exception as e:
        print(f"❌ 配置集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("GPT-SoVITS TTS API 测试工具")
    print("=" * 50)
    
    # 执行测试
    tests = [
        test_tts_api,
        test_config_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
    
    print("\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！TTS API工作正常")
        return True
    else:
        print("⚠️  部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)