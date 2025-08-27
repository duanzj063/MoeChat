#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
记忆表检查工具
专门检查MoeChat系统中记忆相关的PostgreSQL表
"""

import psycopg2
import yaml
from datetime import datetime

def load_config():
    """加载配置文件"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_memory_tables():
    """检查记忆相关的PostgreSQL表"""
    print("MoeChat 记忆表检查工具")
    print("=" * 50)
    
    try:
        config = load_config()
        char_name = config.get('Agent', {}).get('char', '姬子')
        
        # 连接数据库
        db_config = config['VectorStore']['remote']['db_config']
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        
        print(f"\n当前角色: {char_name}")
        print(f"数据库: {db_config['host']}:{db_config['port']}/{db_config['database']}")
        
        # 预期的记忆表名
        expected_tables = {
            'long_memory': '长期记忆(日记)',
            'core_memory': '核心记忆', 
            f'lore_books_{char_name}': f'知识库({char_name})'
        }
        
        print(f"\n=== 预期的记忆表 ===")
        for table_name, description in expected_tables.items():
            print(f"  {table_name}: {description}")
        
        # 检查所有表
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        all_tables = [row[0] for row in cursor.fetchall()]
        print(f"\n=== 数据库中的所有表 ({len(all_tables)}个) ===")
        
        memory_tables_found = []
        other_tables = []
        
        for table_name in all_tables:
            if any(expected in table_name for expected in expected_tables.keys()):
                memory_tables_found.append(table_name)
            else:
                other_tables.append(table_name)
        
        # 显示记忆相关表
        print(f"\n📊 记忆相关表 ({len(memory_tables_found)}个):")
        for table_name in memory_tables_found:
            print(f"  ✅ {table_name}")
            
            # 获取表结构
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            
            columns = cursor.fetchall()
            print(f"     字段: {len(columns)}个")
            for col_name, col_type, nullable in columns:
                null_info = "NULL" if nullable == 'YES' else "NOT NULL"
                print(f"       - {col_name}: {col_type} ({null_info})")
            
            # 获取数据行数
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"     数据行数: {count}")
            
            # 如果有数据，显示最新的几条
            if count > 0:
                cursor.execute(f"""
                    SELECT * FROM {table_name} 
                    ORDER BY created_at DESC 
                    LIMIT 3
                """)
                samples = cursor.fetchall()
                print(f"     最新数据样本:")
                for i, sample in enumerate(samples, 1):
                    # 截断长文本
                    truncated_sample = []
                    for item in sample:
                        if isinstance(item, str) and len(item) > 50:
                            truncated_sample.append(item[:50] + "...")
                        else:
                            truncated_sample.append(item)
                    print(f"       {i}: {truncated_sample}")
            print()
        
        # 显示其他表
        print(f"\n📋 其他表 ({len(other_tables)}个):")
        for table_name in other_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  📄 {table_name}: {count} 行")
        
        # 检查缺失的记忆表
        missing_tables = []
        for expected_table in expected_tables.keys():
            if expected_table not in all_tables:
                missing_tables.append(expected_table)
        
        if missing_tables:
            print(f"\n⚠️  缺失的记忆表 ({len(missing_tables)}个):")
            for table_name in missing_tables:
                description = expected_tables[table_name]
                print(f"  ❌ {table_name}: {description}")
                print(f"     可能原因: 该功能尚未使用或表未自动创建")
        
        # 总结
        print(f"\n=== 总结 ===")
        print(f"✅ 找到记忆表: {len(memory_tables_found)}个")
        print(f"❌ 缺失记忆表: {len(missing_tables)}个")
        print(f"📄 其他表: {len(other_tables)}个")
        
        total_memory_rows = 0
        for table_name in memory_tables_found:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            total_memory_rows += count
        
        print(f"📊 记忆数据总行数: {total_memory_rows}")
        
        if total_memory_rows == 0:
            print("\n💡 建议:")
            print("  1. 确认Agent功能已启用 (config.yaml中Agent.is_up: true)")
            print("  2. 确认记忆功能已启用 (long_memory, is_core_mem, lore_books)")
            print("  3. 进行一些对话，让系统自动创建记忆数据")
            print("  4. 检查应用日志，确认记忆模块正常工作")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_memory_tables()