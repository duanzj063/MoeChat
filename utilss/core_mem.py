# import 
from utilss import embedding
from utilss import config as CConfig, log as Log
from utilss.vector_store import get_vector_store
import yaml
import time
import os
import shortuuid
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"

class Core_Mem:
    def update_config(self):
        self.char = CConfig.config["Agent"]["char"]
        self.user = CConfig.config["Agent"]["user"]
        self.thresholds = 0.5
        self.file_path = f"./data/agents/{self.char}/core_mem.yml"
        
    def __init__(self):
        # self.char = config["char"]
        # self.user = config["user"]
        # self.thresholds = 0.5
        # self.file_path = f"./data/agents/{self.char}/core_mem.yml"
        self.update_config()
        self.msgs = []
        self.mems = []
        self.uuid = []

        # 初始化向量库
        self.vector_store = get_vector_store("core_memory")
        
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:  # 确保数据不为空
                    for key in data:
                        if key and isinstance(data[key], dict):  # 跳过注释行
                            self.mems.append(data[key]["text"])
                            self.msgs.append(f"记忆获取时间：{data[key]['time']}\n{data[key]['text']}")
                            self.uuid.append(key)
        else:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n\n# 核心记忆文件，请勿自行修改！否侧会丢失索引！\n\n")
            t_n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            uuid = shortuuid.ShortUUID().random(length=10)
            while  uuid in self.uuid:
                uuid = shortuuid.ShortUUID().random(length=10)
            text = {uuid: {"time": t_n, "text": f"第一次相遇"}}
            with open(self.file_path, "a", encoding="utf-8") as f:
                yaml.safe_dump(text, f, allow_unicode=True)
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    for key in data:
                        if key and isinstance(data[key], dict):
                            self.mems.append(data[key]["text"])
                            self.msgs.append(f"记忆获取时间：{data[key]['time']}\n{data[key]['text']}")
                            self.uuid.append(key)
        
        # 如果有记忆数据，将其添加到向量库中
        if self.mems:
            try:
                vects = embedding.t2vect(self.mems)
                metadata = [{"uuid": uuid, "time": self.msgs[i].split('\n')[0]} for i, uuid in enumerate(self.uuid)]
                self.vector_store.add_vectors(vects, self.mems, metadata)
                Log.logger.info(f"已将{len(self.mems)}条核心记忆加载到向量库中")
            except Exception as e:
                Log.logger.error(f"核心记忆向量化失败: {str(e)}")

    def find_mem(self, msg: str, res_msg: list):
        try:
            Log.logger.info(f"[CoreMem] 开始搜索核心记忆，查询: {msg[:50]}...")
            # 使用向量库搜索
            Log.logger.info(f"[CoreMem] 生成查询向量")
            query_vector = embedding.t2vect([msg])[0]
            Log.logger.info(f"[CoreMem] 查询向量形状: {query_vector.shape}, 阈值: {self.thresholds}")
            results = self.vector_store.search(query_vector, top_k=5, threshold=self.thresholds)
            Log.logger.info(f"[CoreMem] 搜索返回{len(results)}个结果")
            if results:
                msg_content = ""
                for vector_id, similarity, text in results:
                    Log.logger.info(f"[CoreMem] 处理结果: ID={vector_id}, 相似度={similarity:.4f}, 文本={text[:30]}...")
                    # 根据向量ID找到对应的格式化消息
                    try:
                        idx = int(vector_id)
                        if 0 <= idx < len(self.msgs):
                            msg_content += self.msgs[idx] + "\n"
                            Log.logger.info(f"[CoreMem] 添加记忆内容，索引={idx}")
                        else:
                            Log.logger.warning(f"[CoreMem] 索引超出范围: {idx}, 总数: {len(self.msgs)}")
                    except (ValueError, IndexError) as e:
                        Log.logger.warning(f"[CoreMem] 处理向量ID失败: {vector_id}, 错误: {e}")
                        continue
                if msg_content:
                    Log.logger.info(f"[CoreMem] 找到相关记忆，长度: {len(msg_content)}")
                    res_msg.append(msg_content)
                else:
                    Log.logger.info(f"[CoreMem] 未找到有效的记忆内容")
            else:
                Log.logger.info(f"[CoreMem] 未找到相关记忆")
        except Exception as e:
            Log.logger.error(f"核心记忆搜索失败: {str(e)}")
        
    def add_memory(self, msg: list):
        m_list = {}
        new_texts = []
        new_metadata = []
        
        for m in msg:
            uuid = shortuuid.ShortUUID().random(length=10)
            while uuid in self.uuid:
                uuid = shortuuid.ShortUUID().random(length=10)
            t_n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            m_list[uuid] = {"time": t_n, "text": m}
            
            # 添加到内存列表
            self.mems.append(m)
            self.msgs.append(f"记忆获取时间：{t_n}\n{m}")
            self.uuid.append(uuid)
            
            # 准备向量库数据
            new_texts.append(m)
            new_metadata.append({"uuid": uuid, "time": f"记忆获取时间：{t_n}"})
        
        # 保存到YAML文件
        with open(self.file_path, "a", encoding="utf-8") as f:
            yaml.safe_dump(m_list, f, allow_unicode=True)
        
        # 添加到向量库
        try:
            vects = embedding.t2vect(new_texts)
            self.vector_store.add_vectors(vects, new_texts, new_metadata)
            Log.logger.info(f"[提示]添加核心记忆到向量库: {msg}")
        except Exception as e:
            Log.logger.error(f"添加核心记忆到向量库失败: {str(e)}")
            Log.logger.info(f"[提示]添加核心记忆到本地文件: {msg}")
