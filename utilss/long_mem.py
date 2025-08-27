import os
import yaml
import jionlp as jio
import time
from utilss import embedding, prompt
from utilss.vector_store import get_vector_store
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString
import numpy as np
import pickle
import requests
import jionlp
from bisect import bisect_left, bisect_right
from utilss import config as CConfig, log as Log

class Memorys:
    def update_config(self):
        """
        更新配置信息，从全局配置中加载角色、用户、记忆阈值等参数
        """
        self.char = CConfig.config["Agent"]["char"]  # 角色名称
        self.user = CConfig.config["Agent"]["user"]  # 用户名称
        self.thresholds = CConfig.config["Agent"]["mem_thresholds"]  # 记忆相似度阈值
        self.is_check_memorys = CConfig.config["Agent"]["is_check_memorys"]  # 是否启用记忆检查
    
    def __init__(self):
        """
        初始化记忆系统
        1. 加载配置
        2. 初始化记忆存储结构
        3. 加载已有记忆数据
        4. 建立记忆索引
        """
        # 更新配置信息
        self.update_config()

        # 初始化记忆存储结构
        self.memorys_key = []       # 记录所有记忆的key，秒级整形时间戳
        self.memorys_data = {}      # 记录所有记忆的文本数据，key为时间戳
        self.vectors = []           # 记录文本tag向量，与memorys_key一一对应
        
        # 初始化向量库
        try:
            self.vector_store = get_vector_store("long_memory")
            Log.logger.info("长期记忆向量库初始化成功")
        except Exception as e:
            Log.logger.error(f"长期记忆向量库初始化失败: {str(e)}")
            self.vector_store = None

        # 加载记忆数据
        msg_vectors = []
        path = f"./data/agents/{self.char}/memorys"  # 记忆文件存储路径
        for root, dirs, files in os.walk(path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    # 只处理.yaml文件
                    if file_path.find(".yaml") == -1:
                        continue
                    
                    msgs = []
                    tag = []
                    # 读取记忆文件
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        for key in data:
                            # 替换消息中的占位符
                            self.memorys_data[key] = str(data[key]["msg"]).replace("{{user}}", self.user).replace("{{char}}", self.char)
                            tag.append(data[key]["text_tag"])
                            m_list = data[key]["msg"].split("\n")
                            self.memorys_key.append(key)
                            msgs.append(f"{m_list[1]}{m_list[2]}")
                    
                    # 检查是否存在对应的向量文件
                    if os.path.exists(file_path.replace(".yaml", ".pkl")):
                        # 加载已有向量
                        with open(file_path.replace(".yaml", ".pkl"), "rb") as f:
                            tmp_data = pickle.load(f)
                            self.vectors += tmp_data
                        Log.logger.info(f"加载记忆【{file}】")
                    else:
                        # 生成并保存向量
                        Log.logger.info(f"向量化记忆【{file}】")
                        t_v = embedding.t2vect(tag)
                        msg_vectors.append(t_v)
                        with open(file_path.replace(".yaml", ".pkl"), "wb") as f:
                            pickle.dump(t_v, f)
                        Log.logger.info(f"向量化完成【{file}】")
                except Exception as e:
                    Log.logger.error(f"【{file_path}】记忆加载失败...{str(e)}")
                    continue
        
        # 记录加载结果
        Log.logger.info(f"共加载{len(self.memorys_key)}条记忆...{len(self.vectors)}条记忆向量")
        
        # 将现有记忆添加到向量库
        if self.vector_store and self.memorys_key:
            try:
                texts = []
                metadata = []
                vectors = []
                
                for i, key in enumerate(self.memorys_key):
                    if i < len(self.vectors):
                        texts.append(self.memorys_data[key])
                        metadata.append({"timestamp": key, "type": "long_memory"})
                        vectors.append(self.vectors[i])
                
                if vectors:
                    self.vector_store.add_vectors(vectors, texts, metadata)
                    Log.logger.info(f"成功将{len(vectors)}条长期记忆添加到向量库")
            except Exception as e:
                Log.logger.error(f"添加长期记忆到向量库失败: {str(e)}")

    def find_range_indices(self, low, high) -> list:
        """
        使用二分查找找到在时间范围内的记忆索引
        
        Args:
            low (int): 时间范围下限（时间戳）
            high (int): 时间范围上限（时间戳）
            
        Returns:
            list: 包含起始和结束索引的列表，如果没有找到则返回None
        """
        # 找到第一个 >= low 的索引
        start_idx = bisect_left(self.memorys_key, low)
        # 找到最后一个 <= high 的索引
        end_idx = bisect_right(self.memorys_key, high)
        
        # 如果没有找到匹配的元素
        if end_idx == 0 or start_idx >= len(self.memorys_key):
            return None
        return [start_idx, end_idx-1]
    
    def get_memorys(self, msg: str, res_msg: list, t_n: str):
        """
        获取与输入消息相关的记忆
        
        Args:
            msg (str): 输入的消息文本
            res_msg (list): 用于存储相关记忆的结果列表
            t_n (str): 消息的时间戳字符串
        """
        # 如果没有记忆数据，直接返回
        if not len(self.memorys_key) > 0:
            return
        
        time_span_list = []

        # 提取文本中的时间实体
        res = jio.ner.extract_time(f"[{t_n}]{msg}", time_base=time.time(), with_parsing=False)

        # 获取与文本关联的时间范围信息
        if len(res) > 1:
            for t in res[1:]:
                try:
                    # 解析时间表达式
                    res_t = jio.parse_time(t["text"], time_base=res[0]["text"])
                    # 转换为时间戳
                    time_st1 = int(time.mktime(time.strptime(res_t["time"][0], "%Y-%m-%d %H:%M:%S")))
                    time_st2 = int(time.mktime(time.strptime(res_t["time"][1], "%Y-%m-%d %H:%M:%S")))
                    time_span_list.append(time_st1)
                    time_span_list.append(time_st2)
                except Exception as e:
                    Log.logger.error(f"获取时间区间失败{res_t}, 错误: {str(e)}")
        
        # 如果没有提取到时间信息，直接返回
        if not time_span_list:
            return
        
        # 查找时间范围内的记忆
        res_index = self.find_range_indices(time_span_list[0], time_span_list[1])
        if not res_index:
            return
        
        # 根据配置决定是否进行深度检索
        if self.is_check_memorys:
            Log.logger.info(f"深度检索记忆，检索阈值{self.thresholds}")
            
            # 使用向量库进行搜索
            if self.vector_store:
                try:
                    # 在时间范围内搜索相关记忆
                    results = self.vector_store.search_vectors(msg, top_k=10, threshold=self.thresholds)
                    tmp_msg = ""
                    
                    for result in results:
                        timestamp = result['metadata'].get('timestamp')
                        # 检查是否在时间范围内
                        if timestamp and time_span_list[0] <= timestamp <= time_span_list[1]:
                            if timestamp in self.memorys_data:
                                tmp_msg += str(self.memorys_data[timestamp]) + "\n"
                    
                    if tmp_msg:
                        res_msg.append(tmp_msg)
                except Exception as e:
                    Log.logger.error(f"向量库搜索失败，使用本地搜索: {str(e)}")
                    # 回退到原始方法
                    q_v = embedding.t2vect([msg])[0]
                    tmp_msg = ""
                    for index in range(res_index[0]+1, res_index[1]+1):
                        rr = np.dot(self.vectors[index], q_v)
                        if rr >= self.thresholds:
                            tmp_msg += str(self.memorys_data[self.memorys_key[index]]) + "\n"
                    if tmp_msg:
                        res_msg.append(tmp_msg)
            else:
                # 使用原始方法
                q_v = embedding.t2vect([msg])[0]
                tmp_msg = ""
                for index in range(res_index[0]+1, res_index[1]+1):
                    rr = np.dot(self.vectors[index], q_v)
                    if rr >= self.thresholds:
                        tmp_msg += str(self.memorys_data[self.memorys_key[index]]) + "\n"
                if tmp_msg:
                    res_msg.append(tmp_msg)
        else:
            # 直接添加时间范围内的所有记忆
            tmp_mem = ""
            for index in range(res_index[0]+1, res_index[1]+1):
                tmp_mem += str(self.memorys_data[self.memorys_key[index]])
                tmp_mem += "\n"
            if len(tmp_mem) > 0:
                res_msg.append(tmp_mem)
    
    def add_memory(self, m_data: dict):
        """
        添加新的记忆到系统中
        
        Args:
            m_data (dict): 包含记忆数据的字典，格式为：
                {
                    "t_n": int,  # 时间戳
                    "msg": str,  # 记忆内容
                    "text_tag": str  # 记忆标签
                }
        """
        # 提取时间戳
        t_n = int(m_data["t_n"])
        # 添加到记忆列表
        self.memorys_key.append(t_n)
        self.memorys_data[t_n] = m_data["msg"]
        # 生成并保存向量
        tag_vector = embedding.t2vect([m_data["text_tag"]])[0]
        self.vectors.append(tag_vector)
        
        # 生成文件名
        time_st = time.localtime(t_n)
        file_name = f"{time_st.tm_year}-{time_st.tm_mon}-{time_st.tm_mday}.yaml"
        file_pkl = f"{time_st.tm_year}-{time_st.tm_mon}-{time_st.tm_mday}.pkl"
        
        # 准备要保存的数据
        data = {
            t_n: {
                "text_tag": m_data["text_tag"],
                "msg": LiteralScalarString(m_data["msg"])
            }
        }
        
        # 配置YAML写入器
        Yaml = YAML()
        Yaml.preserve_quotes = True
        Yaml.width = 4096

        # 追加写入记忆文件
        with open(f"./data/agents/{self.char}/memorys/{file_name}", 'a', encoding='utf-8') as f:
            Yaml.dump(data, f)
        
        # 计算当天的起始时间戳
        day_time = t_n - (t_n- time.timezone)%86400
        # 找到当天记忆的起始索引
        index = bisect_left(self.memorys_key, day_time)
        # 保存当天的向量数据
        v_list = self.vectors[index:]
        with open(f"./data/agents/{self.char}/memorys/{file_pkl}", "wb") as f:
            pickle.dump(v_list, f)
        
        # 添加到向量库
        if self.vector_store:
            try:
                metadata = {"timestamp": t_n, "type": "long_memory"}
                self.vector_store.add_vectors(
                    [tag_vector], 
                    [m_data["msg"]], 
                    [metadata]
                )
                Log.logger.info(f"成功添加长期记忆到向量库: {t_n}")
            except Exception as e:
                Log.logger.error(f"添加长期记忆到向量库失败: {str(e)}")
    
    def add_memory1(self, data: list, t_n: int, llm_config: dict):
        """
        提取记忆摘要，记录长期记忆
        通过LLM分析对话内容，生成记忆标签并保存
        
        Args:
            data (list): 对话数据列表，包含用户和角色的消息
            t_n (int): 时间戳
            llm_config (dict): LLM配置信息，包含API地址、密钥、模型等
        """
        # 获取记忆标签提示词
        mmsg = prompt.get_mem_tag_prompt
        # 构建用户消息
        res_msg = "用户：" + data[-2]["content"]
        
        # 构建请求体
        res_body = {
            "model": llm_config["model"],
            "messages": [
                {"role": "system", "content": mmsg},
                {"role": "user", "content": res_msg}
            ]
        }
        
        # 设置请求头
        key = llm_config["key"]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        
        res_tag = ""
        try:
            # 发送请求到LLM
            res = requests.post(llm_config["api"], json=res_body, headers=headers, timeout=15)
            res = res.json()["choices"][0]["message"]["content"]
            # 清理返回结果
            res = jionlp.remove_html_tag(res).replace(" ", "").replace("\n", "")
            Log.logger.info(f"记录日记结果【{res}】")
            
            # 判断是否为日常闲聊
            if res.find("日常闲聊") == -1:
                res_tag = res
            else:
                res_tag = "日常闲聊"
        except Exception as e:
            Log.logger.error(f"错误获取聊天信息！错误: {str(e)}")
            res_tag = "日常闲聊"
        
        # 格式化时间字符串
        t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_n))
        # 提取对话内容
        m1 = data[-2]["content"]  # 用户消息
        m2 = data[-1]["content"]  # 角色消息
        # 设置角色占位符
        c1 = "{{user}}"
        c2 = "{{char}}"
        
        # 构建记忆数据
        m_data = {
            "t_n": t_n,
            "text_tag": res_tag,
            "msg": f"时间：{t_str}\n{c1}：{m1}\n{c2}：{m2}"
        }
        
        # 添加记忆到系统
        self.add_memory(m_data)
