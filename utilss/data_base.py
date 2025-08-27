"""
数据库模块

该模块实现了基于FAISS向量数据库的知识库系统，用于存储和检索AI角色的背景知识（Lore Books）。
主要功能包括：
1. 加载和向量化世界书（YAML格式）内容
2. 构建向量索引以实现快速相似度搜索
3. 支持增量更新和缓存机制
4. 提供高效的文本查询接口
"""

import os
import yaml
import hashlib
import pickle
import numpy as np
import faiss
from utilss import embedding
from utilss import config as CConfig, log as Log
from utilss.vector_store import VectorStoreFactory

# 设置环境变量，避免KMP库重复加载警告
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

def sum_md5(file_path: str):
    """
    计算文件的MD5哈希值
    
    Args:
        file_path (str): 文件路径
        
    Returns:
        str: 文件的MD5哈希值
        
    功能说明：
        - 用于检测文件是否发生变化
        - 通过比较新旧文件的MD5值判断是否需要重新向量化
    """
    with open(file_path, 'rb') as file:
        md5_obj = hashlib.md5()
        while True:
            data = file.read(4096)  # 每次读取4096字节
            if not data:
                break
            md5_obj.update(data)
 
    md5_value = md5_obj.hexdigest()
    return md5_value

class DataBase:
    """
    知识库数据库类
    
    该类实现了基于FAISS向量数据库的知识库系统，用于存储和检索AI角色的背景知识（Lore Books）。
    主要功能包括：
    1. 加载和向量化世界书（YAML格式）内容
    2. 构建向量索引以实现快速相似度搜索
    3. 支持增量更新和缓存机制
    4. 提供高效的文本查询接口
    
    属性：
        thresholds (float): 相似度阈值，用于过滤查询结果
        top_k (int): 每次查询返回的最大结果数量
        path (str): 数据库存储路径
        databases (list): 存储所有文本内容的列表
        vects (np.ndarray): 存储所有向量的numpy数组
        index (faiss.Index): FAISS向量索引对象
    """
    
    def update_config(self):
        """
        更新配置信息
        
        从配置文件中读取以下参数：
        - books_thresholds: 相似度阈值
        - books_top_k: 查询深度（返回结果数量）
        - char: 角色名称，用于确定数据库路径
        
        功能说明：
            - 每次初始化时从配置文件读取最新设置
            - 根据角色名称确定数据库存储路径
        """
        # 优先从根级别配置读取，如果没有则从Agent配置读取
        self.thresholds = float(CConfig.config.get("books_thresholds", 
                                                  CConfig.config.get("Agent", {}).get("books_thresholds", 0.3)))
        self.top_k = int(CConfig.config.get("books_top_k", 
                                           CConfig.config.get("Agent", {}).get("scan_depth", 5)))
        char = CConfig.config["Agent"]["char"]
        self.path = f"./data/agents/{char}/data_base"

    def __init__(self): 
        """
        初始化数据库
        
        初始化流程：
        1. 更新配置信息
        2. 初始化向量存储（本地FAISS或远程PostgreSQL）
        3. 初始化embedding模型
        4. 加载已处理的文件列表（基于MD5值）
        5. 检测新文件或已修改的文件
        6. 向量化新文件内容并存储
        7. 构建或更新向量索引
        
        功能说明：
            - 支持本地FAISS和远程PostgreSQL两种存储模式
            - 支持增量更新，只处理新增或修改的文件
            - 使用MD5哈希值检测文件变化
            - 自动处理空数据库情况
        """
        self.update_config()
        self.databases = []
        
        # 初始化向量存储
        self._init_vector_store()
        
        # 初始化embedding模型
        self._init_embedding_model()
        
        # 创建临时标签目录，用于存储向量化后的缓存文件
        os.makedirs(self.path+"/tmp/labels", exist_ok=True)

        # 加载已处理的文件列表及其MD5值
        base_list = {}
        try:
            with open(self.path+"/tmp/label.yaml", "r", encoding="utf-8") as f:
                base_list = yaml.safe_load(f)
                if base_list == None:
                    base_list = {}
        except:
            base_list = {}
            
        # 扫描数据库目录，找出新增或修改的文件
        file_list = os.listdir(self.path)
        books = []  # 新增或修改的文件名列表
        books_path = []  # 新增或修改的文件路径列表
        for file in file_list:
            file_path = os.path.join(self.path, file)
            if os.path.isfile(file_path):
                f_md5 = sum_md5(file_path)
                # 如果文件不在列表中或MD5值不匹配，则需要重新处理
                if file not in  base_list or base_list[file] != f_md5:
                    books.append(file)
                    books_path.append(file_path)
        
        # 向量化新增或修改的世界书内容
        for index in range(len(books)):
            with open(books_path[index], "r", encoding="utf-8") as f:
                datas = yaml.safe_load(f)
                try:
                    tmp1 = []  # 存储键（标题）
                    tmp2 = []  # 存储值（内容）
                    for data in datas:
                        tmp1.append(data)
                        tmp2.append(datas[data])
                    
                    # 将文本转换为向量
                    if self.embedding_model:
                        vect_list = self.embedding_model.encode(tmp1)
                    else:
                        vect_list = embedding.t2vect(tmp1)
                    
                    # 根据存储模式处理数据
                    if self.use_remote_store:
                        # 远程存储：直接添加到向量库
                        ids = self.vector_store.add_vectors(
                            vect_list, tmp2, 
                            [{"source": books[index], "type": "lore_book"} for _ in tmp2]
                        )
                        Log.logger.info(f"成功向量化并存储【{books[index]}】世界书到远程数据库，共加载{len(tmp1)}条数据。")
                    else:
                        # 本地存储：保存缓存文件
                        res_data = {"vect": vect_list, "text": tmp2}
                        pick_data = pickle.dumps(res_data)
                        with open(f"{self.path}/tmp/labels/{books[index]}.pkl", "wb") as f2:
                            f2.write(pick_data)
                        Log.logger.info(f"成功向量化【{books[index]}】世界书，共加载{len(tmp1)}条数据。")
                    
                    # 更新文件MD5值
                    base_list[books[index]] = sum_md5(books_path[index])
                except Exception as e:
                    Log.logger.error(f"[错误]世界书[{ books[index]}]加载错误: {str(e)}")
                    # 即使出错也更新MD5值，避免重复尝试
                    base_list[books[index]] = sum_md5(books_path[index])
        
        # 根据存储模式初始化索引
        if self.use_remote_store:
            # 远程存储：不需要本地索引
            Log.logger.info(f"使用远程PostgreSQL向量存储，当前向量数量: {self.vector_store.get_vector_count()}")
        else:
            # 本地存储：加载所有向量数据并建立FAISS索引
            self._load_local_data()
            
        # 保存更新后的文件列表
        with open( f"{self.path}/tmp/label.yaml", "w") as f:
             yaml.safe_dump(base_list, f)
    
    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            # 获取向量存储配置
            vector_config = CConfig.config.get('VectorStore', {})
            
            # 创建向量存储实例
            table_name = f"lore_books_{CConfig.config['Agent']['char']}"
            self.vector_store = VectorStoreFactory.create_vector_store(vector_config, table_name)
            
            # 判断是否使用远程存储
            self.use_remote_store = vector_config.get('mode', 'local') == 'remote'
            
            Log.logger.info(f"向量存储初始化成功: {vector_config.get('mode', 'local')}模式")
            
        except Exception as e:
            Log.logger.error(f"向量存储初始化失败: {e}，回退到本地FAISS模式")
            self.use_remote_store = False
            self.vector_store = None
    
    def _init_embedding_model(self):
        """初始化embedding模型"""
        try:
            from utilss.vector_store import EmbeddingFactory
            
            # 获取embedding配置
            embedding_config = CConfig.config.get('Embedding', {})
            
            # 创建embedding模型实例
            self.embedding_model = EmbeddingFactory.create_embedding_model(embedding_config)
            
            Log.logger.info(f"Embedding模型初始化成功: {embedding_config.get('mode', 'local')}模式")
            
        except Exception as e:
            Log.logger.error(f"Embedding模型初始化失败: {e}，使用默认模型")
            self.embedding_model = None
    
    def _load_local_data(self):
        """加载本地FAISS数据"""
        file_list = os.listdir(f"{self.path}/tmp/labels")
        tmp_list = []
        
        for file in file_list:
            if not os.path.isfile( f"{self.path}/tmp/labels/{file}"):
                continue
            with open(f"{self.path}/tmp/labels/{file}", "rb") as f:
                tmp_data =  pickle.load(f)
            tmp_list.append(tmp_data["vect"])
            self.databases += tmp_data["text"]
            Log.logger.info(f"成功加载【{file}】世界书，共加载{len(tmp_data['vect'])}条数据。")
            
        # 合并所有向量数据
        if len(tmp_list) > 0:
            self.vects = np.concatenate(tmp_list)
        else:
            # 如果没有数据，创建一个填充向量
            v = embedding.t2vect(["填充用废物"])
            tmp_list.append(v)
            self.vects = np.concatenate(tmp_list)
            self.databases.append("这是一条没有用的知识")
            
        # 建立FAISS索引（使用内积相似度）
        self.index = faiss.IndexFlatIP(len(self.vects[0]))
        self.index.add(self.vects)

    def search(self, text: list[str]) -> str:
        """
        查询接口
        
        根据输入的文本查询最相关的知识库内容
        
        Args:
            text (list[str]): 查询文本列表
            
        Returns:
            str: 查询结果，格式为"内容1\n\n内容2\n\n..."
            
        功能说明：
            - 将查询文本转换为向量
            - 根据存储模式使用不同的搜索方法
            - 根据相似度阈值过滤结果
            - 返回符合条件的内容，每条内容之间用两个换行符分隔
        """
        try:
            if self.use_remote_store and self.vector_store:
                # 远程存储：使用向量库的搜索接口
                msg = ""
                for query_text in text:
                    # 向量化查询内容
                    if self.embedding_model:
                        query_vector = self.embedding_model.encode(query_text)
                    else:
                        query_vector = embedding.t2vect(query_text)
                    
                    # 执行向量搜索
                    results = self.vector_store.search(
                        query_vector, 
                        top_k=self.top_k, 
                        threshold=self.thresholds
                    )
                    
                    # 处理搜索结果
                    for _, similarity, content in results:
                        msg += content + "\n\n"
                
                return msg
            else:
                # 本地存储：使用FAISS索引
                msg = ""
                # 检查是否有有效的索引和数据
                if not hasattr(self, 'index') or self.index is None:
                    Log.logger.error("FAISS索引未初始化")
                    return ""
                
                if not hasattr(self, 'databases') or len(self.databases) == 0:
                    Log.logger.error("数据库内容为空")
                    return ""
                
                # 向量化查询内容
                if self.embedding_model:
                    vect = self.embedding_model.encode(text)
                else:
                    vect = embedding.t2vect(text)
                    
                if vect is None or len(vect) == 0:
                    Log.logger.error("查询文本向量化失败")
                    return ""
                
                # 查询：返回top_k个最相似的结果
                # D: 相似度分数数组, I: 对应的索引数组
                D, I = self.index.search(vect, self.top_k)
                
                # 处理查询结果
                for index in range(len(D)):
                    for i2 in range(len(D[index])):
                        # 如果相似度分数大于等于阈值，则添加到结果中
                        if D[index][i2] >= self.thresholds:
                            msg += self.databases[I[index][i2]] + "\n\n"
                
                if not msg.strip():
                    Log.logger.warning(f"未找到满足阈值({self.thresholds})的搜索结果")
                
                return msg
                
        except Exception as e:
            Log.logger.error(f"知识库搜索失败: {e}")
            import traceback
            Log.logger.error(f"详细错误信息: {traceback.format_exc()}")
            return ""
