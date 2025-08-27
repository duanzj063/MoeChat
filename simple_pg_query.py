#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的PostgreSQL向量库查询工具
快速检查数据库状态
"""

import psycopg2
import yaml
import sys
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def connect_db(config: Dict[str, Any]):
    """连接数据库"""
    db_config = config['VectorStore']['remote']['db_config']
    return psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )

def main():
    """主函数"""
    print("🔍 PostgreSQL向量库数据检查")
    print("=" * 50)
    
    try:
        # 加载配置并连接
        config = load_config()
        conn = connect_db(config)
        cursor = conn.cursor()
        
        # 检查pgvector扩展
        cursor.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        if vector_ext:
            print(f"✅ pgvector扩展: {vector_ext[0]} v{vector_ext[1]}")
        else:
            print("❌ pgvector扩展未安装")
        
        # 查询所有用户表
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        print(f"\n📊 数据库表统计 (共{len(tables)}个表):")
        total_rows = 0
        
        for (table_name,) in tables:
            # 获取行数
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            total_rows += row_count
            
            # 获取表结构信息
            cursor.execute(f"""
                SELECT COUNT(*) as col_count
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND table_schema = 'public';
            """)
            col_count = cursor.fetchone()[0]
            
            # 检查是否有向量列
            cursor.execute(f"""
                SELECT column_name
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND table_schema = 'public'
                AND data_type = 'USER-DEFINED';
            """)
            vector_cols = cursor.fetchall()
            
            status = "📊" if row_count > 0 else "⚪"
            vector_info = f" [向量列: {len(vector_cols)}]" if vector_cols else ""
            print(f"  {status} {table_name}: {row_count} 行, {col_count} 列{vector_info}")
            
            # 如果有数据，显示最新记录的时间
            if row_count > 0:
                try:
                    cursor.execute(f"""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = '{table_name}' AND table_schema = 'public'
                        AND (column_name LIKE '%created%' OR column_name LIKE '%updated%')
                        ORDER BY column_name LIMIT 1;
                    """)
                    time_col = cursor.fetchone()
                    if time_col:
                        cursor.execute(f"SELECT MAX({time_col[0]}) FROM {table_name};")
                        latest_time = cursor.fetchone()[0]
                        if latest_time:
                            print(f"    └─ 最新数据: {latest_time}")
                except:
                    pass
        
        print(f"\n📈 总计: {total_rows} 行数据")
        
        if total_rows == 0:
            print("\n💡 建议:")
            print("  1. 尝试与AI进行对话，生成一些向量数据")
            print("  2. 检查应用程序是否正确配置为远程模式")
            print("  3. 查看应用程序日志是否有错误信息")
        else:
            print("\n✅ 数据库运行正常，已有向量数据存储")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()