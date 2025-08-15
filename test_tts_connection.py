#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS服务连接测试脚本
用于诊断TTS服务连接问题
"""

import requests
import json
import sys
from utilss import config as CConfig

def test_tts_connection():
    """测试TTS服务连接"""
    print("=== TTS服务连接测试 ===")
    
    # 读取配置
    tts_api = CConfig.config["GSV"]["api"]
    print(f"TTS API地址: {tts_api}")
    
    # 测试基本连接
    print("\n1. 测试基本连接...")
    try:
        # 尝试GET请求测试连接
        response = requests.get(tts_api, timeout=5)
        print(f"✓ 服务器响应状态码: {response.status_code}")
        if response.status_code != 200:
            print(f"⚠ 响应内容: {response.text[:200]}")
    except requests.exceptions.ConnectionError as e:
        print(f"✗ 连接失败: 无法连接到服务器")
        print(f"  错误详情: {str(e)}")
        print(f"  请检查:")
        print(f"    - TTS服务是否正在运行")
        print(f"    - IP地址是否正确: {tts_api.split('//')[1].split(':')[0]}")
        print(f"    - 端口是否正确: {tts_api.split(':')[-1].split('/')[0]}")
        print(f"    - 防火墙是否阻止连接")
        return False
    except requests.exceptions.Timeout as e:
        print(f"✗ 连接超时: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ 未知错误: {str(e)}")
        return False
    
    # 测试TTS API
    print("\n2. 测试TTS API...")
    test_data = {
        "text": "你好，这是一个测试",
        "text_lang": CConfig.config["GSV"]["text_lang"],
        "ref_audio_path": CConfig.config["GSV"]["ref_audio_path"],
        "prompt_text": CConfig.config["GSV"]["prompt_text"],
        "prompt_lang": CConfig.config["GSV"]["prompt_lang"],
        "seed": CConfig.config["GSV"]["seed"],
        "top_k": CConfig.config["GSV"]["top_k"],
        "batch_size": CConfig.config["GSV"]["batch_size"],
    }
    
    try:
        response = requests.post(tts_api, json=test_data, timeout=30)
        if response.status_code == 200:
            print(f"✓ TTS API测试成功")
            print(f"  响应大小: {len(response.content)} 字节")
            print(f"  Content-Type: {response.headers.get('content-type', 'unknown')}")
            return True
        else:
            print(f"✗ TTS API返回错误状态码: {response.status_code}")
            print(f"  响应内容: {response.text[:500]}")
            return False
    except requests.exceptions.ConnectionError as e:
        print(f"✗ TTS API连接失败: {str(e)}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"✗ TTS API请求超时: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ TTS API测试失败: {str(e)}")
        return False

def print_config_info():
    """打印配置信息"""
    print("\n=== 当前TTS配置信息 ===")
    gsv_config = CConfig.config["GSV"]
    for key, value in gsv_config.items():
        if key != "ex_config":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value if value else '无额外配置'}")

def main():
    """主函数"""
    try:
        print_config_info()
        success = test_tts_connection()
        
        print("\n=== 测试结果 ===")
        if success:
            print("✓ TTS服务连接正常")
            print("如果仍然遇到问题，请检查:")
            print("  - 服务器负载是否过高")
            print("  - 网络是否稳定")
            print("  - 配置参数是否正确")
        else:
            print("✗ TTS服务连接失败")
            print("建议解决方案:")
            print("  1. 确认TTS服务器正在运行")
            print("  2. 检查IP地址和端口配置")
            print("  3. 检查防火墙设置")
            print("  4. 尝试在服务器上直接访问TTS API")
            print("  5. 查看TTS服务器日志")
            
    except Exception as e:
        print(f"测试脚本执行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()