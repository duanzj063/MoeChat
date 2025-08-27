#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地Embedding模型实现

提供基于ModelScope的本地embedding模型功能。
"""

import os
import numpy as np
from typing import List, Dict, Any, Union
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from .vector_store import EmbeddingModel

class LocalEmbeddingModel(EmbeddingModel):
    """本地Embedding模型实现
    
    使用ModelScope库实现的本地embedding模型。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化本地Embedding模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_name = config.get('model_name', 'nlp_gte_sentence-embedding_chinese-base')
        self.cache_folder = config.get('cache_folder', './models')
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self):
        """加载ModelScope模型"""
        try:
            # 尝试从本地路径加载
            local_path = f"./utilss/models/{self.model_name.split('/')[-1]}"
            if os.path.exists(local_path):
                self.pipeline = pipeline(
                    Tasks.sentence_embedding,
                    model=local_path,
                    sequence_length=100
                )
            else:
                # 从ModelScope加载
                self.pipeline = pipeline(
                    Tasks.sentence_embedding,
                    model=self.model_name,
                    sequence_length=100
                )
            
            # 测试模型并获取维度
            test_result = self.pipeline(input={"source_sentence": ["测试文本"]})
            if isinstance(test_result, dict) and 'text_embedding' in test_result:
                test_vector = test_result['text_embedding'][0]
                self.dimension = len(test_vector)
            else:
                # 回退方案
                self.dimension = 768  # GTE模型的默认维度
                
            print(f"本地Embedding模型加载成功: {self.model_name}, 维度: {self.dimension}")
            
        except Exception as e:
            raise RuntimeError(f"加载本地Embedding模型失败: {e}")
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量
        
        Args:
            text: 输入文本或文本列表
            
        Returns:
            np.ndarray: 编码后的向量或向量矩阵
        """
        try:
            if isinstance(text, str):
                # 单个文本
                result = self.pipeline(input={"source_sentence": [text]})
                if isinstance(result, dict) and 'text_embedding' in result:
                    vector = np.array(result['text_embedding'][0], dtype=np.float32)
                else:
                    raise ValueError(f"未知的模型输出格式: {type(result)}")
                return vector
            else:
                # 文本列表
                result = self.pipeline(input={"source_sentence": text})
                if isinstance(result, dict) and 'text_embedding' in result:
                    vectors = np.array(result['text_embedding'], dtype=np.float32)
                else:
                    raise ValueError(f"未知的模型输出格式: {type(result)}")
                return vectors
                
        except Exception as e:
            raise RuntimeError(f"文本编码失败: {e}")
    
    def get_dimension(self) -> int:
        """获取向量维度
        
        Returns:
            int: 向量维度
        """
        return self.dimension
    
    def health_check(self) -> bool:
        """健康检查
        
        Returns:
            bool: 模型是否正常工作
        """
        try:
            # 测试编码功能
            test_vector = self.encode("健康检查测试文本")
            return isinstance(test_vector, np.ndarray) and test_vector.shape[0] == self.dimension
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'cache_folder': self.cache_folder,
            'model_type': 'local_modelscope'
        }
    
    def clear_cache(self) -> bool:
        """清空缓存（本地模型无需实现）
        
        Returns:
            bool: 总是返回True
        """
        return True
    
    def benchmark(self, texts: List[str]) -> Dict[str, float]:
        """性能基准测试
        
        Args:
            texts: 测试文本列表
            
        Returns:
            Dict[str, float]: 性能指标
        """
        import time
        
        try:
            # 单个文本编码测试
            start_time = time.time()
            for text in texts[:10]:  # 测试前10个
                self.encode(text)
            single_time = (time.time() - start_time) / min(10, len(texts))
            
            # 批量编码测试
            start_time = time.time()
            if len(texts) > 1:
                self.encode(texts[:10])
            batch_time = (time.time() - start_time) / min(10, len(texts))
            
            return {
                'avg_single_encode_time': single_time,
                'avg_batch_encode_time': batch_time,
                'dimension': self.dimension
            }
        except Exception as e:
            return {
                'error': str(e),
                'avg_single_encode_time': -1,
                'avg_batch_encode_time': -1,
                'dimension': self.dimension
            }