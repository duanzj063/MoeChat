#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL向量库查询工具
用于检查向量数据库中的表和数据情况
"""

import psycopg2
import yaml
import sys
import os
from typing import List, Dict, Any

class PostgreSQLVectorQuery:
    def __init__(self, config_path: str = "config.yaml"):
        """初始化PostgreSQL连接"""
        self.config = self._load_config(config_path)
        self.db_config = self.config['VectorStore']['remote']['db_config']
        self.connection = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            sys.exit(1)
    
    def connect(self):
        """连接到PostgreSQL数据库"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            print(f"✅ 成功连接到PostgreSQL数据库: {self.db_config['host']}:{self.db_config['port']}")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            sys.exit(1)
    
    def disconnect(self):
        """断开数据库连接"""
        if self.connection:
            self.connection.close()
            print("🔌 数据库连接已断开")
    
    def execute_query(self, query: str) -> List[tuple]:
        """执行SQL查询"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f"❌ 查询执行失败: {e}")
            return []
    
    def list_vector_tables(self):
        """列出所有向量表"""
        print("\n📋 查询向量表列表...")
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name LIKE '%vector%' OR table_name LIKE '%memory%' OR table_name LIKE '%embedding%'
        ORDER BY table_name;
        """
        
        results = self.execute_query(query)
        if results:
            print(f"📊 找到 {len(results)} 个相关表:")
            for (table_name,) in results:
                print(f"  - {table_name}")
        else:
            print("⚠️  未找到相关的向量表")
        
        return [table[0] for table in results]
    
    def list_all_tables(self):
        """列出所有表"""
        print("\n📋 查询所有表...")
        query = """
        SELECT table_name, table_type
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
        
        results = self.execute_query(query)
        if results:
            print(f"📊 找到 {len(results)} 个表:")
            for table_name, table_type in results:
                print(f"  - {table_name} ({table_type})")
        else:
            print("⚠️  未找到任何表")
        
        return results
    
    def check_table_data(self, table_name: str):
        """检查指定表的数据情况"""
        print(f"\n🔍 检查表 '{table_name}' 的数据...")
        
        # 检查表结构
        structure_query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' AND table_schema = 'public'
        ORDER BY ordinal_position;
        """
        
        columns = self.execute_query(structure_query)
        if columns:
            print(f"📋 表结构 ({len(columns)} 列):")
            for col_name, data_type, nullable in columns:
                print(f"  - {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
        
        # 检查数据行数
        count_query = f"SELECT COUNT(*) FROM {table_name};"
        count_result = self.execute_query(count_query)
        if count_result:
            row_count = count_result[0][0]
            print(f"📊 数据行数: {row_count}")
            
            if row_count > 0:
                # 显示前几行数据
                sample_query = f"SELECT * FROM {table_name} LIMIT 3;"
                sample_data = self.execute_query(sample_query)
                if sample_data:
                    print(f"📄 前3行数据示例:")
                    for i, row in enumerate(sample_data, 1):
                        print(f"  行{i}: {row}")
            else:
                print("⚠️  表中没有数据")
        
        return count_result[0][0] if count_result else 0
    
    def check_vector_extension(self):
        """检查向量扩展是否安装"""
        print("\n🔧 检查向量扩展...")
        query = "SELECT * FROM pg_extension WHERE extname = 'vector';"
        results = self.execute_query(query)
        
        if results:
            print("✅ pgvector扩展已安装")
            for ext_info in results:
                print(f"  扩展信息: {ext_info}")
        else:
            print("❌ pgvector扩展未安装")
        
        return bool(results)
    
    def run_full_check(self):
        """运行完整的数据库检查"""
        print("🚀 开始PostgreSQL向量库数据检查...")
        print("=" * 60)
        
        # 连接数据库
        self.connect()
        
        try:
            # 检查向量扩展
            self.check_vector_extension()
            
            # 列出所有表
            all_tables = self.list_all_tables()
            
            # 列出向量相关表
            vector_tables = self.list_vector_tables()
            
            # 检查每个表的数据
            total_rows = 0
            if all_tables:
                print("\n" + "=" * 60)
                print("📊 详细数据检查")
                for table_name, _ in all_tables:
                    row_count = self.check_table_data(table_name)
                    total_rows += row_count
            
            # 总结
            print("\n" + "=" * 60)
            print("📈 检查总结:")
            print(f"  - 总表数: {len(all_tables)}")
            print(f"  - 向量相关表数: {len(vector_tables)}")
            print(f"  - 总数据行数: {total_rows}")
            
            if total_rows == 0:
                print("\n⚠️  数据库中没有任何数据，可能的原因:")
                print("  1. 应用程序还没有写入任何向量数据")
                print("  2. 向量存储配置可能有问题")
                print("  3. 需要先进行一些对话来生成向量数据")
            else:
                print(f"\n✅ 数据库中共有 {total_rows} 行数据")
                
        finally:
            # 断开连接
            self.disconnect()

def main():
    """主函数"""
    try:
        query_tool = PostgreSQLVectorQuery()
        query_tool.run_full_check()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()