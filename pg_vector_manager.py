#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL向量库管理工具
提供查询、清理、测试等功能
"""

import psycopg2
import yaml
import json
import sys
import argparse
from typing import Dict, Any, List
from datetime import datetime

class PostgreSQLVectorManager:
    def __init__(self, config_path: str = "config.yaml"):
        """初始化管理器"""
        self.config = self._load_config(config_path)
        self.db_config = self.config['VectorStore']['remote']['db_config']
        self.connection = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def connect(self):
        """连接数据库"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.connection:
            self.connection.close()
    
    def execute_query(self, query: str, params=None) -> List[tuple]:
        """执行查询"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
            else:
                self.connection.commit()
                results = []
            cursor.close()
            return results
        except Exception as e:
            print(f"❌ 查询执行失败: {e}")
            return []
    
    def status(self):
        """显示数据库状态"""
        print("🔍 PostgreSQL向量库状态")
        print("=" * 50)
        
        if not self.connect():
            return
        
        try:
            # 检查pgvector扩展
            results = self.execute_query("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
            if results:
                print(f"✅ pgvector扩展: {results[0][0]} v{results[0][1]}")
            else:
                print("❌ pgvector扩展未安装")
            
            # 查询所有表
            tables = self.execute_query("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            
            print(f"\n📊 数据库表统计 (共{len(tables)}个表):")
            total_rows = 0
            
            for (table_name,) in tables:
                row_count = self.execute_query(f"SELECT COUNT(*) FROM {table_name};")[0][0]
                total_rows += row_count
                
                # 检查向量列
                vector_cols = self.execute_query(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = '{table_name}' AND table_schema = 'public'
                    AND data_type = 'USER-DEFINED';
                """)
                
                status_icon = "📊" if row_count > 0 else "⚪"
                vector_info = f" [向量列: {len(vector_cols)}]" if vector_cols else ""
                print(f"  {status_icon} {table_name}: {row_count} 行{vector_info}")
            
            print(f"\n📈 总计: {total_rows} 行数据")
            
        finally:
            self.disconnect()
    
    def list_data(self, table_name: str = None, limit: int = 10):
        """列出数据"""
        print(f"📋 数据列表 (限制{limit}条)")
        print("=" * 50)
        
        if not self.connect():
            return
        
        try:
            if table_name:
                tables = [(table_name,)]
            else:
                tables = self.execute_query("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                """)
            
            for (tbl_name,) in tables:
                print(f"\n🗂️  表: {tbl_name}")
                
                # 获取列信息
                columns = self.execute_query(f"""
                    SELECT column_name, data_type FROM information_schema.columns 
                    WHERE table_name = '{tbl_name}' AND table_schema = 'public'
                    AND column_name != 'vector'
                    ORDER BY ordinal_position;
                """)
                
                if not columns:
                    print("  ⚠️  表不存在或无权限")
                    continue
                
                # 构建查询（排除向量列）
                col_names = [col[0] for col in columns]
                col_list = ', '.join(col_names)
                
                rows = self.execute_query(f"""
                    SELECT {col_list} FROM {tbl_name} 
                    ORDER BY created_at DESC LIMIT {limit};
                """)
                
                if rows:
                    for i, row in enumerate(rows, 1):
                        print(f"\n  记录 {i}:")
                        for j, (col_name, _) in enumerate(columns):
                            value = row[j]
                            if isinstance(value, dict):
                                value = json.dumps(value, ensure_ascii=False)
                            elif isinstance(value, str) and len(value) > 100:
                                value = value[:100] + "..."
                            print(f"    {col_name}: {value}")
                else:
                    print("  📭 表为空")
        
        finally:
            self.disconnect()
    
    def clear_table(self, table_name: str, confirm: bool = False):
        """清空表数据"""
        if not confirm:
            response = input(f"⚠️  确定要清空表 '{table_name}' 的所有数据吗? (y/N): ")
            if response.lower() != 'y':
                print("❌ 操作已取消")
                return
        
        print(f"🗑️  清空表: {table_name}")
        
        if not self.connect():
            return
        
        try:
            # 检查表是否存在
            exists = self.execute_query("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = %s AND table_schema = 'public';
            """, (table_name,))
            
            if not exists or exists[0][0] == 0:
                print(f"❌ 表 '{table_name}' 不存在")
                return
            
            # 获取删除前的行数
            before_count = self.execute_query(f"SELECT COUNT(*) FROM {table_name};")[0][0]
            
            # 清空表
            self.execute_query(f"DELETE FROM {table_name};")
            
            print(f"✅ 已删除 {before_count} 行数据")
            
        except Exception as e:
            print(f"❌ 清空表失败: {e}")
        finally:
            self.disconnect()
    
    def test_vector_ops(self):
        """测试向量操作"""
        print("🧪 测试向量操作")
        print("=" * 50)
        
        if not self.connect():
            return
        
        try:
            # 检查test_remote表
            test_table = "test_remote"
            
            # 检查向量维度
            dim_result = self.execute_query(f"SELECT vector_dims(vector) FROM {test_table} LIMIT 1;")
            if dim_result:
                print(f"✅ 向量维度: {dim_result[0][0]}")
            
            # 测试向量相似度搜索
            print("\n🔍 测试向量相似度搜索:")
            similarity_test = self.execute_query(f"""
                SELECT text, vector <-> (SELECT vector FROM {test_table} LIMIT 1) as distance
                FROM {test_table}
                ORDER BY distance
                LIMIT 3;
            """)
            
            if similarity_test:
                print("  相似度排序结果:")
                for i, (text, distance) in enumerate(similarity_test, 1):
                    print(f"    {i}. {text} (距离: {distance:.4f})")
            
            # 测试向量内积
            print("\n📊 测试向量内积:")
            dot_product_test = self.execute_query(f"""
                SELECT text, vector <#> (SELECT vector FROM {test_table} LIMIT 1) as dot_product
                FROM {test_table}
                ORDER BY dot_product DESC
                LIMIT 3;
            """)
            
            if dot_product_test:
                print("  内积排序结果:")
                for i, (text, dot_product) in enumerate(dot_product_test, 1):
                    print(f"    {i}. {text} (内积: {dot_product:.4f})")
            
            print("\n✅ 向量操作测试完成")
            
        except Exception as e:
            print(f"❌ 向量操作测试失败: {e}")
        finally:
            self.disconnect()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PostgreSQL向量库管理工具')
    parser.add_argument('command', choices=['status', 'list', 'clear', 'test'], 
                       help='执行的命令')
    parser.add_argument('--table', '-t', help='指定表名')
    parser.add_argument('--limit', '-l', type=int, default=10, help='限制显示行数')
    parser.add_argument('--yes', '-y', action='store_true', help='自动确认操作')
    
    args = parser.parse_args()
    
    manager = PostgreSQLVectorManager()
    
    try:
        if args.command == 'status':
            manager.status()
        elif args.command == 'list':
            manager.list_data(args.table, args.limit)
        elif args.command == 'clear':
            if not args.table:
                print("❌ 清空操作需要指定表名 (--table)")
                sys.exit(1)
            manager.clear_table(args.table, args.yes)
        elif args.command == 'test':
            manager.test_vector_ops()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有参数，显示状态
        manager = PostgreSQLVectorManager()
        manager.status()
    else:
        main()