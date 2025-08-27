#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地FAISS向量存储实现

提供基于FAISS的本地向量存储功能，支持向量的添加、搜索、删除等操作。
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import faiss

from .vector_store import VectorStore
from . import log as Log

class LocalFAISSVectorStore(VectorStore):
    """本地FAISS向量存储实现
    
    使用FAISS库实现的本地向量存储，支持高效的向量相似度搜索。
    """
    
    def __init__(self, config: Dict[str, Any], table_name: str):
        """初始化本地FAISS向量存储
        
        Args:
            config: 配置字典
            table_name: 存储标识名称
        """
        self.config = config
        self.table_name = table_name
        self.index_type = config.get('index_type', 'flat')
        self.save_path = config.get('save_path', f'./faiss_indexes/{table_name}')
        self.default_dimension = config.get('dimension', 1024)  # 从配置读取默认维度
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # 初始化存储
        self.vectors = []
        self.texts = []
        self.metadatas = []
        self.index = None
        self.dimension = None
        
        Log.logger.info(f"[LocalFAISS] 初始化向量存储: {table_name}, 默认维度: {self.default_dimension}")
        
        # 尝试加载已有的索引
        self._load_index()
    
    def _load_index(self):
        """加载已有的FAISS索引"""
        try:
            if os.path.exists(f"{self.save_path}.index"):
                # 加载FAISS索引
                self.index = faiss.read_index(f"{self.save_path}.index")
                self.dimension = self.index.d
                
                # 加载文本和元数据
                if os.path.exists(f"{self.save_path}.data"):
                    with open(f"{self.save_path}.data", 'rb') as f:
                        data = pickle.load(f)
                        self.vectors = data.get('vectors', [])
                        self.texts = data.get('texts', [])
                        self.metadatas = data.get('metadatas', [])
                        
                print(f"加载FAISS索引成功，包含{len(self.texts)}个向量")
        except Exception as e:
            print(f"加载FAISS索引失败: {e}")
            self.index = None
    
    def _save_index(self):
        """保存FAISS索引"""
        try:
            if self.index is not None:
                # 保存FAISS索引
                faiss.write_index(self.index, f"{self.save_path}.index")
                
                # 保存文本和元数据
                data = {
                    'vectors': self.vectors,
                    'texts': self.texts,
                    'metadatas': self.metadatas
                }
                with open(f"{self.save_path}.data", 'wb') as f:
                    pickle.dump(data, f)
                    
        except Exception as e:
            print(f"保存FAISS索引失败: {e}")
    
    def _create_index(self, dimension: int):
        """创建FAISS索引"""
        Log.logger.info(f"[LocalFAISS] 创建FAISS索引，维度: {dimension}, 索引类型: {self.index_type}")
        if self.index_type == 'flat':
            # 使用内积相似度的平面索引
            self.index = faiss.IndexFlatIP(dimension)
        elif self.index_type == 'ivf':
            # 使用IVF索引
            nlist = self.config.get('nlist', 100)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            # 默认使用平面索引
            self.index = faiss.IndexFlatIP(dimension)
        
        self.dimension = dimension
        Log.logger.info(f"[LocalFAISS] FAISS索引创建完成，维度: {self.dimension}")
    
    def add_vectors(self, vectors: List[np.ndarray], texts: List[str], 
                   metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """添加向量到存储中"""
        Log.logger.info(f"[LocalFAISS] 开始添加向量: {len(vectors)}个向量, {len(texts)}个文本")
        if not vectors or not texts:
            Log.logger.warning(f"[LocalFAISS] 空向量或文本列表，跳过添加")
            return []
        
        if len(vectors) != len(texts):
            raise ValueError("向量和文本数量不匹配")
        
        # 转换向量格式
        Log.logger.info(f"[LocalFAISS] 向量类型: {type(vectors[0])}, 向量形状: {getattr(vectors[0], 'shape', 'N/A')}")
        if isinstance(vectors[0], list):
            vectors = [np.array(v, dtype=np.float32) for v in vectors]
            Log.logger.info(f"[LocalFAISS] 已转换list向量为numpy数组")
        else:
            vectors = [v.astype(np.float32) for v in vectors]
            Log.logger.info(f"[LocalFAISS] 已转换向量为float32格式")
        
        # 检查向量维度
        vector_dim = vectors[0].shape[0]
        Log.logger.info(f"[LocalFAISS] 向量维度: {vector_dim}, 当前索引维度: {self.dimension}, 默认维度: {self.default_dimension}")
        if self.dimension is None:
            # 使用配置中的默认维度或向量的实际维度
            target_dim = self.default_dimension if self.default_dimension else vector_dim
            if vector_dim != target_dim:
                Log.logger.error(f"[LocalFAISS] 向量维度不匹配，期望{target_dim}，实际{vector_dim}")
                raise ValueError(f"向量维度不匹配，期望{target_dim}，实际{vector_dim}")
            Log.logger.info(f"[LocalFAISS] 创建新索引，维度: {target_dim}")
            self._create_index(target_dim)
        elif self.dimension != vector_dim:
            Log.logger.error(f"[LocalFAISS] 向量维度不匹配，期望{self.dimension}，实际{vector_dim}")
            raise ValueError(f"向量维度不匹配，期望{self.dimension}，实际{vector_dim}")
        
        # 准备元数据
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("元数据和文本数量不匹配")
        
        # 生成ID
        start_id = len(self.texts)
        ids = [str(start_id + i) for i in range(len(texts))]
        
        # 添加到存储
        self.vectors.extend(vectors)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        # 添加到FAISS索引
        vectors_array = np.vstack(vectors)
        self.index.add(vectors_array)
        
        # 如果是IVF索引且还未训练，进行训练
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(self.vectors) >= 100:  # 需要足够的数据进行训练
                all_vectors = np.vstack(self.vectors)
                self.index.train(all_vectors)
        
        # 保存索引
        self._save_index()
        
        Log.logger.info(f"[LocalFAISS] 成功添加{len(vectors)}个向量，返回ID: {ids[:3]}...")
        return ids
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
              threshold: float = 0.0) -> List[Tuple[str, float, str]]:
        """搜索相似向量"""
        Log.logger.info(f"[LocalFAISS] 开始搜索，top_k={top_k}, threshold={threshold}, 当前向量数={len(self.texts)}")
        if self.index is None or len(self.texts) == 0:
            Log.logger.warning(f"[LocalFAISS] 索引为空或无向量数据，返回空结果")
            return []
        
        # 确保查询向量格式正确
        Log.logger.info(f"[LocalFAISS] 查询向量类型: {type(query_vector)}, 形状: {getattr(query_vector, 'shape', 'N/A')}")
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
            Log.logger.info(f"[LocalFAISS] 已转换list查询向量为numpy数组")
        else:
            query_vector = query_vector.astype(np.float32)
            Log.logger.info(f"[LocalFAISS] 已转换查询向量为float32格式")
        
        # 检查维度
        if query_vector.shape[0] != self.dimension:
            Log.logger.error(f"[LocalFAISS] 查询向量维度不匹配，期望{self.dimension}，实际{query_vector.shape[0]}")
            raise ValueError(f"查询向量维度不匹配，期望{self.dimension}，实际{query_vector.shape[0]}")
        
        # 执行搜索
        query_vector = query_vector.reshape(1, -1)
        Log.logger.info(f"[LocalFAISS] 执行FAISS搜索，查询向量形状: {query_vector.shape}")
        similarities, indices = self.index.search(query_vector, min(top_k, len(self.texts)))
        Log.logger.info(f"[LocalFAISS] 搜索完成，相似度: {similarities[0][:3]}, 索引: {indices[0][:3]}")
        
        # 处理结果
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= 0 and similarity >= threshold:
                results.append((str(idx), float(similarity), self.texts[idx]))
        
        Log.logger.info(f"[LocalFAISS] 返回{len(results)}个搜索结果")
        return results
    
    def delete_vector(self, vector_id: str) -> bool:
        """删除向量（FAISS不支持直接删除，需要重建索引）"""
        try:
            idx = int(vector_id)
            if 0 <= idx < len(self.texts):
                # 从列表中移除
                del self.vectors[idx]
                del self.texts[idx]
                del self.metadatas[idx]
                
                # 重建索引
                if self.vectors:
                    self._create_index(self.dimension)
                    vectors_array = np.vstack(self.vectors)
                    self.index.add(vectors_array)
                    
                    # 训练IVF索引
                    if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                        if len(self.vectors) >= 100:
                            self.index.train(vectors_array)
                else:
                    self.index = None
                    self.dimension = None
                
                # 保存索引
                self._save_index()
                return True
        except (ValueError, IndexError):
            pass
        return False
    
    def update_vector(self, vector_id: str, new_vector: np.ndarray, 
                     text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新向量（需要重建索引）"""
        try:
            idx = int(vector_id)
            if 0 <= idx < len(self.texts):
                # 更新数据
                if isinstance(new_vector, list):
                    new_vector = np.array(new_vector, dtype=np.float32)
                else:
                    new_vector = new_vector.astype(np.float32)
                
                if new_vector.shape[0] != self.dimension:
                    raise ValueError(f"向量维度不匹配，期望{self.dimension}，实际{new_vector.shape[0]}")
                
                self.vectors[idx] = new_vector
                self.texts[idx] = text
                if metadata is not None:
                    self.metadatas[idx] = metadata
                
                # 重建索引
                self._create_index(self.dimension)
                vectors_array = np.vstack(self.vectors)
                self.index.add(vectors_array)
                
                # 训练IVF索引
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    if len(self.vectors) >= 100:
                        self.index.train(vectors_array)
                
                # 保存索引
                self._save_index()
                return True
        except (ValueError, IndexError):
            pass
        return False
    
    def get_vector_count(self) -> int:
        """获取向量总数"""
        return len(self.texts)
    
    def delete_by_ids(self, vector_ids: List[str]) -> bool:
        """批量删除向量"""
        try:
            # 转换ID为索引
            indices_to_remove = []
            for vector_id in vector_ids:
                try:
                    idx = int(vector_id)
                    if 0 <= idx < len(self.texts):
                        indices_to_remove.append(idx)
                except ValueError:
                    continue
            
            if not indices_to_remove:
                return True
            
            # 按降序排序，从后往前删除
            indices_to_remove.sort(reverse=True)
            
            # 删除数据
            for idx in indices_to_remove:
                del self.vectors[idx]
                del self.texts[idx]
                del self.metadatas[idx]
            
            # 重建索引
            if self.vectors:
                self._create_index(self.dimension)
                vectors_array = np.vstack(self.vectors)
                self.index.add(vectors_array)
                
                # 训练IVF索引
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    if len(self.vectors) >= 100:
                        self.index.train(vectors_array)
            else:
                self.index = None
                self.dimension = None
            
            # 保存索引
            self._save_index()
            return True
            
        except Exception as e:
            print(f"批量删除向量失败: {e}")
            return False
    
    def clear_vectors(self) -> bool:
        """清空所有向量"""
        return self.clear_all()
    
    def clear_all(self) -> bool:
        """清空所有向量"""
        try:
            self.vectors = []
            self.texts = []
            self.metadatas = []
            self.index = None
            self.dimension = None
            
            # 删除保存的文件
            for ext in ['.index', '.data']:
                file_path = f"{self.save_path}{ext}"
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return True
        except Exception as e:
            print(f"清空向量存储失败: {e}")
            return False
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension or 0
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查基本状态
            if self.index is None and len(self.texts) > 0:
                return False
            
            # 检查数据一致性
            if len(self.vectors) != len(self.texts) or len(self.texts) != len(self.metadatas):
                return False
            
            return True
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            'vector_count': len(self.texts),
            'dimension': self.dimension or 0,
            'index_type': self.index_type,
            'save_path': self.save_path,
            'index_trained': hasattr(self.index, 'is_trained') and self.index.is_trained if self.index else False
        }