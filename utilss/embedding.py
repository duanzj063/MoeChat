"""Embedding模块 - 支持本地和远程模型

该模块提供统一的embedding接口，支持:
1. 本地ModelScope模型
2. 远程Xinference API模型

根据配置文件自动选择合适的模型。
"""

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download
import numpy as np
import yaml
import os
from typing import List, Union
from utilss import log as Log
from utilss.vector_store import EmbeddingFactory

# 全局embedding模型实例
embedding_model = None

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        Log.logger.error(f"加载配置文件失败: {e}")
        return None

def init_embedding_model():
    """初始化embedding模型"""
    global embedding_model
    
    if embedding_model is not None:
        return embedding_model
    
    config = load_config()
    if not config:
        Log.logger.error("配置文件加载失败，使用默认本地模型")
        embedding_model = _load_local_model()
        return embedding_model
    
    try:
        # 使用工厂模式创建embedding模型
        embedding_config = config.get('Embedding', {})
        embedding_model = EmbeddingFactory.create_embedding_model(embedding_config)
        Log.logger.info(f"Embedding模型初始化成功: {embedding_config.get('mode', 'local')}模式")
        return embedding_model
        
    except Exception as e:
        Log.logger.error(f"Embedding模型初始化失败: {e}，回退到本地模型")
        embedding_model = _load_local_model()
        return embedding_model

def _load_local_model():
    """加载本地ModelScope模型（回退方案）"""
    try:
        model = pipeline(
            Tasks.sentence_embedding,
            model="./utilss/models/nlp_gte_sentence-embedding_chinese-base",
            sequence_length=100
        )
        Log.logger.info("本地ModelScope模型加载成功")
        return LocalModelWrapper(model)
    except Exception as e:
        Log.logger.warning(f"本地模型加载失败，尝试下载: {e}")
        try:
            model_id = "iic/nlp_gte_sentence-embedding_chinese-base"
            local_dir = "./utilss/models/nlp_gte_sentence-embedding_chinese-base"
            snapshot_download(model_id=model_id, local_dir=local_dir)
            
            model = pipeline(
                Tasks.sentence_embedding,
                model=local_dir,
                sequence_length=100
            )
            Log.logger.info("本地ModelScope模型下载并加载成功")
            return LocalModelWrapper(model)
        except Exception as download_error:
            Log.logger.error(f"本地模型下载失败: {download_error}")
            raise

class LocalModelWrapper:
    """本地ModelScope模型包装器，提供统一接口"""
    
    def __init__(self, model):
        self.model = model
        self.dimension = 768  # GTE模型的维度
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本为向量"""
        if isinstance(texts, str):
            texts = [texts]
        
        result = self.model(input={"source_sentence": texts})["text_embedding"]
        return np.array(result, dtype=np.float32)
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
    
    def similarity_search(self, query: str, candidates: List[str]) -> List[float]:
        """计算相似度分数"""
        input_data = {
            "source_sentence": [query],
            "sentences_to_compare": candidates
        }
        scores = self.model(input=input_data)["scores"]
        return scores

def t2vect(text: Union[str, List[str]]) -> np.ndarray:
    """文本转向量的统一接口
    
    Args:
        text: 单个文本字符串或文本列表
        
    Returns:
        numpy数组，包含文本的向量表示
    """
    model = init_embedding_model()
    return model.encode(text)

def test(msg: str, memorys: List[str], thresholds: float) -> str:
    """测试函数，用于记忆检索
    
    Args:
        msg: 查询消息
        memorys: 记忆列表
        thresholds: 相似度阈值
        
    Returns:
        匹配的记忆内容
    """
    if not memorys:
        return ""
    
    model = init_embedding_model()
    
    try:
        # 如果是本地模型包装器，使用原有的相似度计算方法
        if isinstance(model, LocalModelWrapper):
            scores = model.similarity_search(msg, memorys)
        else:
            # 对于远程模型，需要手动计算相似度
            query_vector = model.encode(msg)
            memory_vectors = model.encode(memorys)
            
            # 计算余弦相似度
            scores = []
            for memory_vector in memory_vectors:
                # 归一化向量
                query_norm = query_vector / np.linalg.norm(query_vector)
                memory_norm = memory_vector / np.linalg.norm(memory_vector)
                
                # 计算余弦相似度
                similarity = np.dot(query_norm, memory_norm)
                scores.append(float(similarity))
        
        # 筛选超过阈值的记忆
        res_msg = ""
        for i, score in enumerate(scores):
            if score > thresholds:
                res_msg += str(memorys[i]) + "\n\n"
        
        return res_msg if res_msg else None
        
    except Exception as e:
        Log.logger.error(f"记忆检索失败: {e}")
        return ""

def get_embedding_model():
    """获取当前的embedding模型实例"""
    return init_embedding_model()

def get_dimension() -> int:
    """获取当前模型的向量维度"""
    model = init_embedding_model()
    return model.get_dimension()

# 初始化模型（延迟加载）
embedding_model = None
