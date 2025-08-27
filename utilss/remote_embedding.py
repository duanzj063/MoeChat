"""远程Embedding模型实现

该模块实现了基于Xinference API的远程Embedding模型调用。
支持Qwen3-Embedding-0.6B等多种远程Embedding模型。
"""

import requests
import numpy as np
from typing import List, Union, Dict, Any, Optional
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .vector_store import EmbeddingModel
from utilss import log as Log


class RemoteEmbeddingModel(EmbeddingModel):
    """远程Embedding模型实现
    
    通过HTTP API调用远程部署的Embedding模型（如Qwen3-Embedding-0.6B）。
    支持批量处理、重试机制和性能优化。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化远程Embedding模型
        
        Args:
            config: 配置字典，包含API地址、密钥等信息
        """
        self.config = config
        self.base_url = config['base_url'].rstrip('/')
        self.api_key = config.get('api_key', '')
        self.model_name = config.get('model_name', 'Qwen3-Embedding-0.6B')
        
        # 性能配置
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        self.batch_size = config.get('batch_size', 32)
        self.max_workers = config.get('max_workers', 4)
        
        # 缓存配置
        self.enable_cache = config.get('enable_cache', False)
        self._cache = {} if self.enable_cache else None
        
        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else ''
        })
        
        # 获取模型信息
        self._model_info = self._get_model_info()
        self.dimension = self._model_info.get('dimension', 1024)
        
        Log.logger.info(f"远程Embedding模型初始化完成: {self.model_name}, 维度: {self.dimension}")
    
    def _get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            response = self.session.get(
                f"{self.base_url}/v1/models",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            models_data = response.json()
            
            # 查找目标模型
            for model in models_data.get('data', []):
                if model.get('id') == self.model_name:
                    Log.logger.info(f"找到模型: {self.model_name}")
                    return {
                        'id': model.get('id'),
                        'dimension': 1024,  # Qwen3-Embedding-0.6B的维度
                        'max_tokens': model.get('max_tokens', 8192)
                    }
            
            # 如果没找到具体模型，返回默认配置
            Log.logger.warning(f"未找到模型 {self.model_name}，使用默认配置")
            return {
                'id': self.model_name,
                'dimension': 1024,
                'max_tokens': 8192
            }
            
        except Exception as e:
            Log.logger.error(f"获取模型信息失败: {e}")
            return {
                'id': self.model_name,
                'dimension': 1024,
                'max_tokens': 8192
            }
    
    def _make_request(self, texts: List[str], retry_count: int = 0) -> List[List[float]]:
        """发送API请求"""
        try:
            payload = {
                'model': self.model_name,
                'input': texts,
                'encoding_format': 'float'
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取embedding向量
            embeddings = []
            for item in result.get('data', []):
                embedding = item.get('embedding', [])
                if not embedding:
                    raise ValueError(f"获取到空的embedding向量")
                embeddings.append(embedding)
            
            if len(embeddings) != len(texts):
                raise ValueError(f"返回的embedding数量({len(embeddings)})与输入文本数量({len(texts)})不匹配")
            
            return embeddings
            
        except Exception as e:
            if retry_count < self.max_retries:
                Log.logger.warning(f"API请求失败，第{retry_count + 1}次重试: {e}")
                time.sleep(self.retry_delay * (retry_count + 1))
                return self._make_request(texts, retry_count + 1)
            else:
                Log.logger.error(f"API请求最终失败: {e}")
                raise
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量"""
        # 统一处理为列表
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        if not texts:
            return np.array([])
        
        # 检查缓存
        if self.enable_cache:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._cache:
                    cached_results.append((i, self._cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_results = []
        
        # 处理未缓存的文本
        all_embeddings = [None] * len(texts)
        
        # 填充缓存结果
        for idx, embedding in cached_results:
            all_embeddings[idx] = embedding
        
        if uncached_texts:
            # 批量处理
            if len(uncached_texts) <= self.batch_size:
                # 单批处理
                embeddings = self._make_request(uncached_texts)
                for i, embedding in enumerate(embeddings):
                    original_idx = uncached_indices[i]
                    all_embeddings[original_idx] = embedding
                    
                    # 更新缓存
                    if self.enable_cache:
                        self._cache[uncached_texts[i]] = embedding
            else:
                # 多批并行处理
                embeddings = self._batch_encode(uncached_texts)
                for i, embedding in enumerate(embeddings):
                    original_idx = uncached_indices[i]
                    all_embeddings[original_idx] = embedding
                    
                    # 更新缓存
                    if self.enable_cache:
                        self._cache[uncached_texts[i]] = embedding
        
        # 转换为numpy数组
        result = np.array(all_embeddings, dtype=np.float32)
        
        # 如果输入是单个文本，返回一维数组
        if single_text:
            result = result[0]
        
        Log.logger.debug(f"编码完成，输入文本数: {len(texts)}, 输出向量形状: {result.shape}")
        return result
    
    def _batch_encode(self, texts: List[str]) -> List[List[float]]:
        """批量并行编码"""
        all_embeddings = []
        
        # 分批处理
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        if len(batches) == 1:
            # 单批处理，不需要并行
            return self._make_request(batches[0])
        
        # 多批并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {executor.submit(self._make_request, batch): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                try:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    Log.logger.error(f"批量处理失败: {e}")
                    raise
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 测试编码一个简单文本
            test_text = "健康检查测试"
            embedding = self.encode(test_text)
            
            if embedding is not None and len(embedding) == self.dimension:
                Log.logger.info("远程Embedding模型健康检查通过")
                return True
            else:
                Log.logger.error("健康检查失败：返回的向量维度不正确")
                return False
                
        except Exception as e:
            Log.logger.error(f"健康检查失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        stats = {
            'model_name': self.model_name,
            'base_url': self.base_url,
            'dimension': self.dimension,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers,
            'cache_enabled': self.enable_cache
        }
        
        if self.enable_cache and self._cache:
            stats['cache_size'] = len(self._cache)
        
        return stats
    
    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache and self._cache:
            cache_size = len(self._cache)
            self._cache.clear()
            Log.logger.info(f"已清空缓存，原缓存大小: {cache_size}")
    
    def benchmark(self, test_texts: List[str] = None, iterations: int = 3) -> Dict[str, float]:
        """性能基准测试"""
        if test_texts is None:
            test_texts = [
                "这是一个测试文本",
                "性能基准测试正在进行",
                "远程Embedding模型响应时间测试",
                "批量处理性能评估",
                "向量化处理速度测试"
            ]
        
        results = {
            'single_text_avg_time': 0.0,
            'batch_text_avg_time': 0.0,
            'throughput_texts_per_second': 0.0
        }
        
        try:
            # 单文本处理测试
            single_times = []
            for _ in range(iterations):
                start_time = time.time()
                self.encode(test_texts[0])
                single_times.append(time.time() - start_time)
            
            results['single_text_avg_time'] = sum(single_times) / len(single_times)
            
            # 批量处理测试
            batch_times = []
            for _ in range(iterations):
                start_time = time.time()
                self.encode(test_texts)
                batch_times.append(time.time() - start_time)
            
            results['batch_text_avg_time'] = sum(batch_times) / len(batch_times)
            results['throughput_texts_per_second'] = len(test_texts) / results['batch_text_avg_time']
            
            Log.logger.info(f"性能基准测试完成: {results}")
            
        except Exception as e:
            Log.logger.error(f"性能基准测试失败: {e}")
        
        return results