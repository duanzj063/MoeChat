"""向量存储抽象接口模块

该模块定义了向量存储和Embedding模型的抽象基类，
为不同的实现提供统一的接口规范。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class VectorStore(ABC):
    """向量存储抽象基类
    
    定义了向量存储系统的基本接口，包括向量的增删改查操作。
    不同的实现（如FAISS、PostgreSQL等）需要继承此类并实现所有抽象方法。
    """
    
    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, texts: List[str], 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """添加向量和对应文本
        
        Args:
            vectors: 向量数组，shape为(n, dimension)
            texts: 对应的文本列表
            metadata: 可选的元数据列表
            
        Returns:
            List[str]: 返回添加的向量ID列表
            
        Raises:
            ValueError: 当向量和文本数量不匹配时
            Exception: 存储操作失败时
        """
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5, 
              threshold: float = 0.5) -> List[Tuple[str, float, str]]:
        """向量相似度搜索
        
        Args:
            query_vector: 查询向量，shape为(dimension,)
            top_k: 返回最相似的前k个结果
            threshold: 相似度阈值，低于此值的结果将被过滤
            
        Returns:
            List[Tuple[str, float, str]]: 返回(向量ID, 相似度分数, 文本)的列表
            
        Raises:
            ValueError: 当查询向量维度不正确时
            Exception: 搜索操作失败时
        """
        pass
    
    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> bool:
        """根据ID删除向量
        
        Args:
            ids: 要删除的向量ID列表
            
        Returns:
            bool: 删除是否成功
            
        Raises:
            Exception: 删除操作失败时
        """
        pass
    
    @abstractmethod
    def update_vector(self, vector_id: str, vector: np.ndarray, 
                     text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新向量
        
        Args:
            vector_id: 要更新的向量ID
            vector: 新的向量，shape为(dimension,)
            text: 新的文本
            metadata: 可选的新元数据
            
        Returns:
            bool: 更新是否成功
            
        Raises:
            ValueError: 当向量维度不正确时
            Exception: 更新操作失败时
        """
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """获取向量总数
        
        Returns:
            int: 向量总数
        """
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """清空所有向量
        
        Returns:
            bool: 清空是否成功
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度
        
        Returns:
            int: 向量维度
        """
        pass


class EmbeddingModel(ABC):
    """Embedding模型抽象基类
    
    定义了文本向量化模型的基本接口。
    不同的实现（如本地模型、远程API等）需要继承此类并实现所有抽象方法。
    """
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """文本转向量
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            np.ndarray: 向量数组，shape为(len(texts), dimension)
            
        Raises:
            ValueError: 当输入文本为空时
            Exception: 编码操作失败时
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度
        
        Returns:
            int: 向量维度
        """
        pass
    
    def encode_single(self, text: str) -> np.ndarray:
        """单个文本转向量（便捷方法）
        
        Args:
            text: 要编码的文本
            
        Returns:
            np.ndarray: 向量，shape为(dimension,)
        """
        result = self.encode([text])
        return result[0] if len(result) > 0 else np.array([])


class VectorStoreFactory:
    """向量存储工厂类
    
    根据配置创建相应的向量存储实例。
    """
    
    @staticmethod
    def create_vector_store(config: Dict[str, Any], table_name: str) -> VectorStore:
        """创建向量存储实例
        
        Args:
            config: 配置字典
            table_name: 表名
            
        Returns:
            VectorStore: 向量存储实例
            
        Raises:
            ValueError: 当配置不正确或模式不支持时
        """
        mode = config.get('mode', 'local')
        
        if mode == 'local':
            from .local_faiss_store import LocalFAISSVectorStore
            return LocalFAISSVectorStore(config.get('local', {}), table_name)
        elif mode == 'remote':
            from .pg_vector_store import PostgreSQLVectorStore
            return PostgreSQLVectorStore(config.get('remote', {}), table_name)
        else:
            raise ValueError(f"不支持的向量存储模式: {mode}")


class EmbeddingFactory:
    """Embedding模型工厂类
    
    根据配置创建相应的Embedding模型实例。
    """
    
    @staticmethod
    def create_embedding_model(config: Dict[str, Any]) -> EmbeddingModel:
        """创建Embedding模型实例
        
        Args:
            config: 配置字典
            
        Returns:
            EmbeddingModel: Embedding模型实例
            
        Raises:
            ValueError: 当配置不正确或模式不支持时
        """
        mode = config.get('mode', 'local')
        
        if mode == 'local':
            from .local_embedding import LocalEmbeddingModel
            return LocalEmbeddingModel(config.get('local', {}))
        elif mode == 'remote':
            from .remote_embedding import RemoteEmbeddingModel
            return RemoteEmbeddingModel(config.get('remote', {}))
        else:
            raise ValueError(f"不支持的Embedding模式: {mode}")


def get_vector_store(table_name: str) -> VectorStore:
    """获取向量存储实例的便捷函数
    
    Args:
        table_name: 表名
        
    Returns:
        VectorStore: 向量存储实例
    """
    from . import config as CConfig
    
    # 获取向量存储配置
    vector_config = CConfig.config.get('VectorStore', {})
    
    # 使用工厂创建向量存储实例
    return VectorStoreFactory.create_vector_store(vector_config, table_name)