#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量库改造效果测试脚本

测试内容：
1. 配置文件加载和解析
2. 本地FAISS模式测试
3. 远程PostgreSQL模式测试
4. Embedding模型切换测试
5. 数据存储和检索功能测试
6. 性能对比测试
"""

import os
import sys
import time
import yaml
import numpy as np
from typing import List, Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utilss.vector_store import VectorStoreFactory, EmbeddingFactory
from utilss import embedding
from utilss import config as CConfig
from utilss.data_base import DataBase

class VectorMigrationTester:
    """向量库改造测试类"""
    
    def __init__(self):
        self.test_results = []
        self.test_data = [
            "这是一个测试文本，用于验证向量化功能",
            "人工智能是计算机科学的一个分支",
            "机器学习是实现人工智能的重要方法",
            "深度学习是机器学习的一个子领域",
            "自然语言处理是AI的重要应用领域"
        ]
    
    def log_test(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """记录测试结果"""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "duration": duration
        }
        self.test_results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message} ({duration:.3f}s)")
    
    def test_config_loading(self):
        """测试配置文件加载"""
        try:
            start_time = time.time()
            
            # 加载配置文件
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查VectorStore配置
            assert 'VectorStore' in config, "配置文件缺少VectorStore配置"
            vector_config = config['VectorStore']
            assert 'mode' in vector_config, "VectorStore配置缺少mode字段"
            
            # 检查Embedding配置
            assert 'Embedding' in config, "配置文件缺少Embedding配置"
            embedding_config = config['Embedding']
            assert 'mode' in embedding_config, "Embedding配置缺少mode字段"
            
            duration = time.time() - start_time
            self.log_test("配置文件加载", True, "配置文件结构正确", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("配置文件加载", False, f"配置文件加载失败: {e}", duration)
            return False
    
    def test_local_embedding(self):
        """测试本地Embedding模型"""
        try:
            start_time = time.time()
            
            # 创建本地embedding配置
            local_config = {
                'mode': 'local',
                'local': {
                    'model_name': 'damo/nlp_gte_sentence-embedding_chinese-base',
                    'cache_folder': './models'
                }
            }
            
            # 创建本地embedding模型
            embedding_model = EmbeddingFactory.create_embedding_model(local_config)
            
            # 测试编码功能
            test_text = "这是一个测试文本"
            vector = embedding_model.encode(test_text)
            
            assert isinstance(vector, np.ndarray), "返回的向量类型不正确"
            assert len(vector.shape) == 1, "向量维度不正确"
            assert vector.shape[0] > 0, "向量长度为0"
            
            # 测试维度获取
            dimension = embedding_model.get_dimension()
            assert dimension == vector.shape[0], "维度信息不一致"
            
            duration = time.time() - start_time
            self.log_test("本地Embedding模型", True, f"向量维度: {dimension}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("本地Embedding模型", False, f"测试失败: {e}", duration)
            return False
    
    def test_remote_embedding(self):
        """测试远程Embedding模型"""
        try:
            start_time = time.time()
            
            # 创建远程embedding配置
            remote_config = {
                'mode': 'remote',
                'remote': {
                    'base_url': 'http://27.159.93.61:9997',
                    'api_key': '1',
                    'model_name': 'Qwen3-Embedding-0.6B',
                    'timeout': 30,
                    'max_retries': 3
                }
            }
            
            # 创建远程embedding模型
            embedding_model = EmbeddingFactory.create_embedding_model(remote_config)
            
            # 测试健康检查
            health_ok = embedding_model.health_check()
            assert health_ok, "远程模型健康检查失败"
            
            # 测试编码功能
            test_text = "这是一个测试文本"
            vector = embedding_model.encode(test_text)
            
            assert isinstance(vector, np.ndarray), "返回的向量类型不正确"
            assert len(vector.shape) == 1, "向量维度不正确"
            assert vector.shape[0] > 0, "向量长度为0"
            
            # 测试维度获取
            dimension = embedding_model.get_dimension()
            assert dimension == vector.shape[0], "维度信息不一致"
            
            duration = time.time() - start_time
            self.log_test("远程Embedding模型", True, f"向量维度: {dimension}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("远程Embedding模型", False, f"测试失败: {e}", duration)
            return False
    
    def test_local_vector_store(self):
        """测试本地FAISS向量存储"""
        try:
            start_time = time.time()
            
            # 创建本地向量存储配置
            local_config = {
                'mode': 'local',
                'local': {
                    'index_type': 'flat',
                    'save_path': './test_faiss_index'
                }
            }
            
            # 创建本地向量存储
            vector_store = VectorStoreFactory.create_vector_store(local_config, "test_local")
            
            # 清空之前的数据
            vector_store.clear_vectors()
            
            # 准备测试数据
            vectors = []
            texts = []
            metadatas = []
            
            for i, text in enumerate(self.test_data):
                # 使用当前embedding模块生成向量
                vector = embedding.t2vect(text)
                vectors.append(vector)
                texts.append(text)
                metadatas.append({"id": i, "type": "test"})
            
            # 测试添加向量
            ids = vector_store.add_vectors(vectors, texts, metadatas)
            assert len(ids) == len(self.test_data), "添加的向量数量不正确"
            
            # 测试搜索功能
            query_vector = embedding.t2vect("人工智能相关内容")
            results = vector_store.search(query_vector, top_k=3, threshold=0.1)
            assert len(results) > 0, "搜索结果为空"
            
            # 测试向量数量
            count = vector_store.get_vector_count()
            assert count == len(self.test_data), f"向量数量统计不正确: 期望{len(self.test_data)}, 实际{count}"
            
            duration = time.time() - start_time
            self.log_test("本地FAISS向量存储", True, f"存储{count}个向量，搜索返回{len(results)}个结果", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("本地FAISS向量存储", False, f"测试失败: {e}", duration)
            return False
    
    def test_remote_vector_store(self):
        """测试远程PostgreSQL向量存储"""
        try:
            start_time = time.time()
            
            # 创建远程向量存储配置
            remote_config = {
                'mode': 'remote',
                'remote': {
                    'db_config': {
                        'host': '27.159.93.61',
                        'port': 5432,
                        'database': 'postgres',
                        'user': 'root',
                        'password': 'Fsti<#>2025'
                    },
                    'vector_config': {
                        'dimension': 1024,
                        'index_type': 'ivfflat',
                        'lists': 100
                    }
                }
            }
            
            # 创建远程向量存储
            vector_store = VectorStoreFactory.create_vector_store(remote_config, "test_remote")
            
            # 测试健康检查
            health_ok = vector_store.health_check()
            assert health_ok, "远程向量存储健康检查失败"
            
            # 清空测试表
            vector_store.clear_vectors()
            
            # 准备测试数据（使用远程embedding）
            remote_embedding_config = {
                'mode': 'remote',
                'remote': {
                    'base_url': 'http://27.159.93.61:9997',
                    'api_key': '1',
                    'model_name': 'Qwen3-Embedding-0.6B'
                }
            }
            
            embedding_model = EmbeddingFactory.create_embedding_model(remote_embedding_config)
            
            vectors = []
            texts = []
            metadatas = []
            
            for i, text in enumerate(self.test_data):
                vector = embedding_model.encode(text)
                vectors.append(vector)
                texts.append(text)
                metadatas.append({"id": i, "type": "test"})
            
            # 测试添加向量
            ids = vector_store.add_vectors(vectors, texts, metadatas)
            assert len(ids) == len(self.test_data), "添加的向量数量不正确"
            
            # 测试搜索功能
            query_vector = embedding_model.encode("人工智能相关内容")
            results = vector_store.search(query_vector, top_k=3, threshold=0.1)
            assert len(results) > 0, "搜索结果为空"
            
            # 测试向量数量
            count = vector_store.get_vector_count()
            assert count == len(self.test_data), "向量数量统计不正确"
            
            duration = time.time() - start_time
            self.log_test("远程PostgreSQL向量存储", True, f"存储{count}个向量，搜索返回{len(results)}个结果", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("远程PostgreSQL向量存储", False, f"测试失败: {e}", duration)
            return False
    
    def test_database_integration(self):
        """测试DataBase类集成"""
        try:
            start_time = time.time()
            
            # 备份原配置
            original_config = CConfig.config.copy() if hasattr(CConfig, 'config') else {}
            
            # 设置测试配置（本地模式，确保测试能正常运行）
            test_config = {
                'VectorStore': {
                    'mode': 'local',
                    'local': {
                        'save_path': './data/agents/test_char/vector_store/faiss_index'
                    }
                },
                'Embedding': {
                    'mode': 'local',
                    'local': {
                        'model_name': 'damo/nlp_gte_sentence-embedding_chinese-base',
                        'cache_folder': './models'
                    }
                },
                'Agent': {
                    'char': 'test_char'
                },
                'books_thresholds': 0.3,
                'books_top_k': 5
            }
            
            # 更新配置
            CConfig.config = test_config
            
            # 创建测试目录
            test_db_path = "./data/agents/test_char/data_base"
            os.makedirs(test_db_path, exist_ok=True)
            
            # 创建测试世界书文件
            test_book = {
                "AI基础": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "机器学习": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                "深度学习": "深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的工作方式。"
            }
            
            with open(f"{test_db_path}/test_book.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(test_book, f, allow_unicode=True)
            
            # 初始化DataBase
            db = DataBase()
            
            # 测试搜索功能
            search_results = db.search(["什么是人工智能"])
            assert isinstance(search_results, str), "搜索结果类型不正确"
            assert len(search_results.strip()) > 0, "搜索结果为空"
            
            # 恢复原配置
            CConfig.config = original_config
            
            duration = time.time() - start_time
            self.log_test("DataBase类集成测试", True, "搜索功能正常", duration)
            return True
            
        except Exception as e:
            # 恢复原配置
            if 'original_config' in locals():
                CConfig.config = original_config
            
            duration = time.time() - start_time
            self.log_test("DataBase类集成测试", False, f"测试失败: {e}", duration)
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("向量库改造效果测试")
        print("="*60)
        
        tests = [
            self.test_config_loading,
            self.test_local_embedding,
            self.test_remote_embedding,
            self.test_local_vector_store,
            self.test_remote_vector_store,
            self.test_database_integration
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_test(test.__name__, False, f"测试异常: {e}", 0)
            print()
        
        # 输出测试总结
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n失败的测试:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test_name']}: {result['message']}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    tester = VectorMigrationTester()
    tester.run_all_tests()