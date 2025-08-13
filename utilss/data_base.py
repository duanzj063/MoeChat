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
        - scan_depth: 查询深度（返回结果数量）
        - char: 角色名称，用于确定数据库路径
        
        功能说明：
            - 每次初始化时从配置文件读取最新设置
            - 根据角色名称确定数据库存储路径
        """
        self.thresholds = float(CConfig.config["Agent"]["books_thresholds"])
        self.top_k = int(CConfig.config["Agent"]["scan_depth"])
        char = CConfig.config["Agent"]["char"]
        self.path = f"./data/agents/{char}/data_base"

    def __init__(self): 
        """
        初始化数据库
        
        初始化流程：
        1. 更新配置信息
        2. 创建必要的目录结构
        3. 加载已处理的文件列表（基于MD5值）
        4. 检测新文件或已修改的文件
        5. 向化新文件内容并保存缓存
        6. 加载所有向量数据
        7. 构建FAISS索引
        
        功能说明：
            - 支持增量更新，只处理新增或修改的文件
            - 使用MD5哈希值检测文件变化
            - 向量化后的数据以pickle格式缓存，提高性能
            - 自动处理空数据库情况，提供默认填充数据
        """
        self.update_config()
        # self.vects = np.array([])
        self.databases = []
        # self.thresholds = thresholds
        # self.top_k = top_k

        # self.path = f"./data/agents/{char}/data_base"
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
        
        # 向量化新增或修改的世界书内容，并保存缓存文件
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
                    vect_list = embedding.t2vect(tmp1)
                    # 准备要缓存的数据
                    res_data = {"vect": vect_list, "text": tmp2}
                    pick_data = pickle.dumps(res_data)
                    # 保存缓存文件
                    with open(f"{self.path}/tmp/labels/{books[index]}.pkl", "wb") as f2:
                        f2.write(pick_data)
                    Log.logger.info(f"成功向量化【{books[index]}】世界书，共加载{len(tmp1)}条数据。")
                    # 更新文件MD5值
                    base_list[books[index]] = sum_md5(books_path[index])
                except Exception as e:
                    Log.logger.error(f"[错误]世界书[{ books[index]}]加载错误: {str(e)}")
                    # 即使出错也更新MD5值，避免重复尝试
                    base_list[books[index]] = sum_md5(books_path[index])
        
        # 加载所有向量数据，建立数据库、索引，并保存更新后的总表内容
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
            
        # 保存更新后的文件列表
        with open( f"{self.path}/tmp/label.yaml", "w") as f:
             yaml.safe_dump(base_list, f)
             
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
            - 使用FAISS索引进行相似度搜索
            - 根据相似度阈值过滤结果
            - 返回符合条件的内容，每条内容之间用两个换行符分隔
        """
        # 储存返回结果
        msg = ""
        # 向量化查询内容
        vect = embedding.t2vect(text)
        # 查询：返回top_k个最相似的结果
        # D: 相似度分数数组, I: 对应的索引数组
        D, I = self.index.search(vect, self.top_k)
        # 处理查询结果
        for index in range(len(D)):
            for i2 in range(len(D[index])):
                # 如果相似度分数大于等于阈值，则添加到结果中
                if D[index][i2] >= self.thresholds:
                    msg += self.databases[I[index][i2]] + "\n\n"
        return msg
