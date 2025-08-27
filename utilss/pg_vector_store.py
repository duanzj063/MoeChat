"""PostgreSQL + pgvector 向量存储实现

该模块实现了基于PostgreSQL和pgvector扩展的向量存储系统。
提供高性能的向量相似度搜索和完整的CRUD操作。
"""

import psycopg2
import psycopg2.extras
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import uuid
import json
import time
from contextlib import contextmanager
from .vector_store import VectorStore
from utilss import log as Log


class PostgreSQLVectorStore(VectorStore):
    """PostgreSQL + pgvector 向量存储实现
    
    使用PostgreSQL数据库和pgvector扩展实现向量存储和检索。
    支持多种向量索引类型（IVFFlat、HNSW）和相似度计算方法。
    """
    
    def __init__(self, config: Dict[str, Any], table_name: str):
        """初始化PostgreSQL向量存储
        
        Args:
            config: 配置字典，包含数据库连接和向量配置信息
            table_name: 表名
        """
        self.config = config
        self.table_name = table_name
        self.db_config = config['db_config']
        self.vector_config = config.get('vector_config', {})
        self.dimension = self.vector_config.get('dimension', 1024)
        self.index_type = self.vector_config.get('index_type', 'ivfflat')
        
        # 连接池配置
        self.max_connections = config.get('max_connections', 10)
        self.connection_timeout = config.get('connection_timeout', 30)
        
        # 初始化数据库连接和表结构
        self._init_database()
        
        Log.logger.info(f"PostgreSQL向量存储初始化完成: {table_name}, 维度: {self.dimension}")
    
    def _init_database(self):
        """初始化数据库连接和表结构"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 确保pgvector扩展已启用
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # 创建表
                    self._create_table(cursor)
                    
                    # 创建索引
                    self._create_index(cursor)
                    
                conn.commit()
                Log.logger.info(f"数据库表 {self.table_name} 初始化完成")
                
        except Exception as e:
            Log.logger.error(f"数据库初始化失败: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            conn = psycopg2.connect(
                **self.db_config,
                connect_timeout=self.connection_timeout
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def _create_table(self, cursor):
        """创建向量表"""
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id VARCHAR(50) PRIMARY KEY,
                vector vector({self.dimension}) NOT NULL,
                text TEXT NOT NULL,
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        cursor.execute(create_table_sql)
        
        # 创建更新时间触发器
        trigger_sql = f"""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
            
            DROP TRIGGER IF EXISTS update_{self.table_name}_updated_at ON {self.table_name};
            CREATE TRIGGER update_{self.table_name}_updated_at
                BEFORE UPDATE ON {self.table_name}
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        cursor.execute(trigger_sql)
    
    def _create_index(self, cursor):
        """创建向量索引"""
        index_name = f"{self.table_name}_vector_idx"
        
        # 删除已存在的索引
        cursor.execute(f"DROP INDEX IF EXISTS {index_name};")
        
        if self.index_type == 'ivfflat':
            lists = self.vector_config.get('lists', 100)
            index_sql = f"""
                CREATE INDEX {index_name} ON {self.table_name} 
                USING ivfflat (vector vector_cosine_ops) 
                WITH (lists = {lists});
            """
        elif self.index_type == 'hnsw':
            ef_construction = self.vector_config.get('ef_construction', 64)
            index_sql = f"""
                CREATE INDEX {index_name} ON {self.table_name} 
                USING hnsw (vector vector_cosine_ops) 
                WITH (ef_construction = {ef_construction});
            """
        else:
            # 默认使用IVFFlat
            index_sql = f"""
                CREATE INDEX {index_name} ON {self.table_name} 
                USING ivfflat (vector vector_cosine_ops) 
                WITH (lists = 100);
            """
        
        cursor.execute(index_sql)
        Log.logger.info(f"创建向量索引: {index_name}, 类型: {self.index_type}")
    
    def add_vectors(self, vectors, texts: List[str], 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """添加向量和对应文本"""
        if len(vectors) != len(texts):
            raise ValueError("向量数量与文本数量不匹配")
        
        # 转换向量格式
        if isinstance(vectors, list):
            if len(vectors) > 0 and isinstance(vectors[0], (list, np.ndarray)):
                # 列表格式的向量
                vectors = [np.array(v, dtype=np.float32) if isinstance(v, list) else v for v in vectors]
                vector_dim = vectors[0].shape[0]
            else:
                raise ValueError("向量格式不正确")
        elif isinstance(vectors, np.ndarray):
            vector_dim = vectors.shape[1]
        else:
            raise ValueError("不支持的向量格式")
        
        if vector_dim != self.dimension:
            raise ValueError(f"向量维度不匹配，期望 {self.dimension}，实际 {vector_dim}")
        
        # 生成ID列表
        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # 准备元数据
        if metadata is None:
            metadata = [{}] * len(vectors)
        elif len(metadata) != len(vectors):
            raise ValueError("元数据数量与向量数量不匹配")
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 批量插入
                    insert_sql = f"""
                        INSERT INTO {self.table_name} (id, vector, text, metadata)
                        VALUES %s
                    """
                    
                    # 准备数据
                    data_tuples = []
                    for i, (vector, text, meta) in enumerate(zip(vectors, texts, metadata)):
                        vector_str = '[' + ','.join(map(str, vector)) + ']'
                        data_tuples.append((ids[i], vector_str, text, json.dumps(meta)))
                    
                    # 执行批量插入
                    psycopg2.extras.execute_values(
                        cursor, insert_sql, data_tuples, template=None, page_size=100
                    )
                    
                conn.commit()
                Log.logger.info(f"成功添加 {len(vectors)} 个向量到 {self.table_name}")
                return ids
                
        except Exception as e:
            Log.logger.error(f"添加向量失败: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, top_k: int = 5, 
              threshold: float = 0.5) -> List[Tuple[str, float, str]]:
        """向量相似度搜索"""
        if query_vector.shape[0] != self.dimension:
            raise ValueError(f"查询向量维度不匹配，期望 {self.dimension}，实际 {query_vector.shape[0]}")
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 设置HNSW搜索参数（如果使用HNSW索引）
                    if self.index_type == 'hnsw':
                        ef_search = self.vector_config.get('ef_search', 40)
                        cursor.execute(f"SET hnsw.ef_search = {ef_search};")
                    
                    # 构建查询向量字符串
                    query_vector_str = '[' + ','.join(map(str, query_vector)) + ']'
                    
                    # 执行相似度搜索（使用余弦相似度）
                    search_sql = f"""
                        SELECT id, 1 - (vector <=> %s::vector) as similarity, text
                        FROM {self.table_name}
                        WHERE 1 - (vector <=> %s::vector) >= %s
                        ORDER BY vector <=> %s::vector
                        LIMIT %s;
                    """
                    
                    cursor.execute(search_sql, (
                        query_vector_str, query_vector_str, threshold, query_vector_str, top_k
                    ))
                    
                    results = cursor.fetchall()
                    
                    # 转换结果格式
                    formatted_results = [
                        (row[0], float(row[1]), row[2]) for row in results
                    ]
                    
                    Log.logger.debug(f"搜索完成，返回 {len(formatted_results)} 个结果")
                    return formatted_results
                    
        except Exception as e:
            Log.logger.error(f"向量搜索失败: {e}")
            raise
    
    def delete_by_ids(self, ids: List[str]) -> bool:
        """根据ID删除向量"""
        if not ids:
            return True
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    delete_sql = f"DELETE FROM {self.table_name} WHERE id = ANY(%s);"
                    cursor.execute(delete_sql, (ids,))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    Log.logger.info(f"成功删除 {deleted_count} 个向量")
                    return True
                    
        except Exception as e:
            Log.logger.error(f"删除向量失败: {e}")
            return False
    
    def update_vector(self, vector_id: str, vector: np.ndarray, 
                     text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """更新向量"""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"向量维度不匹配，期望 {self.dimension}，实际 {vector.shape[0]}")
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    vector_str = '[' + ','.join(map(str, vector)) + ']'
                    metadata_json = json.dumps(metadata or {})
                    
                    update_sql = f"""
                        UPDATE {self.table_name} 
                        SET vector = %s::vector, text = %s, metadata = %s::jsonb
                        WHERE id = %s;
                    """
                    
                    cursor.execute(update_sql, (vector_str, text, metadata_json, vector_id))
                    
                    updated_count = cursor.rowcount
                    conn.commit()
                    
                    if updated_count > 0:
                        Log.logger.info(f"成功更新向量: {vector_id}")
                        return True
                    else:
                        Log.logger.warning(f"未找到要更新的向量: {vector_id}")
                        return False
                        
        except Exception as e:
            Log.logger.error(f"更新向量失败: {e}")
            return False
    
    def get_vector_count(self) -> int:
        """获取向量总数"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                    count = cursor.fetchone()[0]
                    return count
                    
        except Exception as e:
            Log.logger.error(f"获取向量数量失败: {e}")
            return 0
    
    def clear_vectors(self) -> bool:
        """清空所有向量"""
        return self.clear_all()
    
    def clear_all(self) -> bool:
        """清空所有向量"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"DELETE FROM {self.table_name};")
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    Log.logger.info(f"成功清空表 {self.table_name}，删除 {deleted_count} 个向量")
                    return True
                    
        except Exception as e:
            Log.logger.error(f"清空向量表失败: {e}")
            return False
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 获取基本统计信息
                    stats_sql = f"""
                        SELECT 
                            COUNT(*) as total_vectors,
                            pg_size_pretty(pg_total_relation_size('{self.table_name}')) as table_size,
                            pg_size_pretty(pg_indexes_size('{self.table_name}')) as index_size
                        FROM {self.table_name};
                    """
                    
                    cursor.execute(stats_sql)
                    result = cursor.fetchone()
                    
                    return {
                        'total_vectors': result[0],
                        'table_size': result[1],
                        'index_size': result[2],
                        'dimension': self.dimension,
                        'index_type': self.index_type
                    }
                    
        except Exception as e:
            Log.logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1;")
                    return True
        except Exception as e:
            Log.logger.error(f"健康检查失败: {e}")
            return False