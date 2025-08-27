import requests
import json
import numpy as np
import time

# 远程Embedding模型配置
EMBEDDING_CONFIG = {
    'url': 'http://27.159.93.61:9997',
    'api_key': '1',
    'model_name': 'Qwen3-Embedding-0.6B'
}

def test_remote_embedding_connection():
    """测试远程Embedding模型连接和功能"""
    print("正在测试远程Embedding模型连接...")
    
    try:
        # 1. 测试模型列表接口
        print("\n1. 获取可用模型列表...")
        start_time = time.time()
        
        list_url = f"{EMBEDDING_CONFIG['url']}/v1/models"
        headers = {
            'Authorization': f"Bearer {EMBEDDING_CONFIG['api_key']}",
            'Content-Type': 'application/json'
        }
        
        response = requests.get(list_url, headers=headers, timeout=10)
        end_time = time.time()
        
        print(f"  -> 模型列表请求耗时: {(end_time - start_time) * 1000:.2f} ms")
        
        if response.status_code == 200:
            models = response.json()
            print(f"  -> 成功获取模型列表，共 {len(models.get('data', []))} 个模型")
            
            # 检查目标模型是否存在
            target_model_found = False
            for model in models.get('data', []):
                if EMBEDDING_CONFIG['model_name'] in model.get('id', ''):
                    target_model_found = True
                    print(f"  -> 找到目标模型: {model.get('id')}")
                    break
            
            if not target_model_found:
                print(f"  -> 警告: 未找到目标模型 {EMBEDDING_CONFIG['model_name']}")
                print("  -> 可用模型列表:")
                for model in models.get('data', []):
                    print(f"     - {model.get('id')}")
        else:
            print(f"  -> 获取模型列表失败: {response.status_code} - {response.text}")
            return False
            
        # 2. 测试Embedding接口
        print("\n2. 测试Embedding接口...")
        
        test_texts = [
            "这是一个测试文本",
            "用户喜欢看电影，特别是科幻片",
            "今天天气很好，适合出门散步"
        ]
        
        embedding_url = f"{EMBEDDING_CONFIG['url']}/v1/embeddings"
        
        payload = {
            "model": EMBEDDING_CONFIG['model_name'],
            "input": test_texts
        }
        
        start_time = time.time()
        response = requests.post(embedding_url, headers=headers, json=payload, timeout=30)
        end_time = time.time()
        
        print(f"  -> Embedding请求耗时: {(end_time - start_time) * 1000:.2f} ms")
        
        if response.status_code == 200:
            result = response.json()
            embeddings = result.get('data', [])
            
            print(f"  -> 成功获取 {len(embeddings)} 个文本的向量")
            
            if embeddings:
                first_embedding = embeddings[0].get('embedding', [])
                print(f"  -> 向量维度: {len(first_embedding)}")
                print(f"  -> 向量前5个值: {first_embedding[:5]}")
                
                # 3. 测试向量相似度计算
                print("\n3. 测试向量相似度计算...")
                if len(embeddings) >= 2:
                    vec1 = np.array(embeddings[0]['embedding'])
                    vec2 = np.array(embeddings[1]['embedding'])
                    
                    # 计算余弦相似度
                    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    print(f"  -> 文本1和文本2的余弦相似度: {cosine_sim:.4f}")
                    
                    # 计算L2距离
                    l2_distance = np.linalg.norm(vec1 - vec2)
                    print(f"  -> 文本1和文本2的L2距离: {l2_distance:.4f}")
                
                print("\n结论: 远程Embedding模型连接成功，功能正常！")
                return True
            else:
                print("  -> 错误: 未获取到向量数据")
                return False
        else:
            print(f"  -> Embedding请求失败: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("  -> 错误: 请求超时")
        return False
    except requests.exceptions.ConnectionError:
        print("  -> 错误: 连接失败，请检查网络和服务器状态")
        return False
    except Exception as e:
        print(f"  -> 发生未知错误: {e}")
        return False

def test_embedding_performance():
    """测试Embedding模型性能"""
    print("\n\n=== Embedding性能测试 ===")
    
    # 测试不同批次大小的性能
    batch_sizes = [1, 5, 10, 20]
    test_text = "这是一个用于性能测试的示例文本"
    
    embedding_url = f"{EMBEDDING_CONFIG['url']}/v1/embeddings"
    headers = {
        'Authorization': f"Bearer {EMBEDDING_CONFIG['api_key']}",
        'Content-Type': 'application/json'
    }
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        test_texts = [f"{test_text} {i}" for i in range(batch_size)]
        
        payload = {
            "model": EMBEDDING_CONFIG['model_name'],
            "input": test_texts
        }
        
        try:
            start_time = time.time()
            response = requests.post(embedding_url, headers=headers, json=payload, timeout=60)
            end_time = time.time()
            
            if response.status_code == 200:
                elapsed_time = (end_time - start_time) * 1000
                avg_time_per_text = elapsed_time / batch_size
                print(f"  -> 总耗时: {elapsed_time:.2f} ms")
                print(f"  -> 平均每个文本: {avg_time_per_text:.2f} ms")
            else:
                print(f"  -> 请求失败: {response.status_code}")
                
        except Exception as e:
            print(f"  -> 测试失败: {e}")

if __name__ == "__main__":
    # 安装依赖提示
    try:
        import requests
        import numpy as np
    except ImportError as e:
        print(f"错误: 缺少依赖库 {e}")
        print("请安装: pip install requests numpy")
        exit(1)
    
    print("=== 远程Embedding模型测试 ===")
    print(f"模型URL: {EMBEDDING_CONFIG['url']}")
    print(f"模型名称: {EMBEDDING_CONFIG['model_name']}")
    
    success = test_remote_embedding_connection()
    
    if success:
        test_embedding_performance()
        print("\n\n✅ 所有测试完成！远程Embedding模型可以正常使用。")
    else:
        print("\n\n❌ 测试失败！请检查配置和网络连接。")
        exit(1)