#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聊天记录存储检查工具
检查MoeChat系统中聊天记录的存储情况
"""

import os
import yaml
import psycopg2
from ruamel.yaml import YAML
from datetime import datetime
import json

def load_config():
    """加载配置文件"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_yaml_history(char_name):
    """检查YAML格式的聊天历史文件"""
    print("\n=== 检查YAML聊天历史文件 ===")
    
    # 检查data目录结构
    data_dir = "./data"
    agents_dir = f"./data/agents"
    char_dir = f"./data/agents/{char_name}"
    history_file = f"./data/agents/{char_name}/history.yaml"
    
    print(f"数据目录: {data_dir}")
    print(f"存在: {os.path.exists(data_dir)}")
    
    print(f"\nAgents目录: {agents_dir}")
    print(f"存在: {os.path.exists(agents_dir)}")
    
    print(f"\n角色目录: {char_dir}")
    print(f"存在: {os.path.exists(char_dir)}")
    
    print(f"\n历史文件: {history_file}")
    print(f"存在: {os.path.exists(history_file)}")
    
    if os.path.exists(history_file):
        try:
            # 使用ruamel.yaml读取文件
            yaml_loader = YAML()
            with open(history_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"\n文件大小: {len(content)} 字符")
                print(f"文件内容预览 (前500字符):")
                print(content[:500])
                
                # 尝试解析YAML内容
                f.seek(0)
                try:
                    data = yaml_loader.load(f)
                    if data:
                        print(f"\n解析到的消息数量: {len(data) if isinstance(data, list) else '非列表格式'}")
                        if isinstance(data, list) and len(data) > 0:
                            print(f"最新消息: {data[-1]}")
                    else:
                        print("\n文件为空或无有效数据")
                except Exception as e:
                    print(f"\nYAML解析错误: {e}")
        except Exception as e:
            print(f"读取文件错误: {e}")
    else:
        print("\n历史文件不存在，可能原因:")
        print("1. 还没有进行过对话")
        print("2. data目录结构未创建")
        print("3. Agent功能未正确启用")

def check_postgresql_memory(config):
    """检查PostgreSQL中的记忆数据"""
    print("\n\n=== 检查PostgreSQL记忆数据 ===")
    
    try:
        # 连接数据库
        db_config = config['vector_store']['remote']['db_config']
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        
        # 检查所有表
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"\n数据库中的表: {len(tables)}个")
        
        for table in tables:
            table_name = table[0]
            print(f"\n--- 表: {table_name} ---")
            
            # 获取表结构
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            
            columns = cursor.fetchall()
            print(f"字段: {[f'{col[0]}({col[1]})' for col in columns]}")
            
            # 获取行数
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"数据行数: {count}")
            
            # 如果有数据，显示样本
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                samples = cursor.fetchall()
                print(f"样本数据:")
                for i, sample in enumerate(samples, 1):
                    print(f"  {i}: {sample}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"PostgreSQL连接错误: {e}")

def check_memory_modules():
    """检查记忆模块的配置"""
    print("\n\n=== 记忆模块配置检查 ===")
    
    config = load_config()
    agent_config = config.get('Agent', {})
    
    print(f"Agent功能启用: {agent_config.get('is_up', False)}")
    print(f"角色名称: {agent_config.get('char', 'N/A')}")
    print(f"用户名称: {agent_config.get('user', 'N/A')}")
    print(f"长期记忆(日记): {agent_config.get('long_memory', False)}")
    print(f"核心记忆: {agent_config.get('is_core_mem', False)}")
    print(f"知识库: {agent_config.get('lore_books', False)}")
    print(f"上下文长度: {agent_config.get('context_length', 0)}")
    
    return config

def main():
    """主函数"""
    print("MoeChat 聊天记录存储检查工具")
    print("=" * 50)
    
    try:
        # 检查记忆模块配置
        config = check_memory_modules()
        char_name = config.get('Agent', {}).get('char', '姬子')
        
        # 检查YAML历史文件
        check_yaml_history(char_name)
        
        # 检查PostgreSQL记忆数据
        check_postgresql_memory(config)
        
        print("\n\n=== 总结 ===")
        print("聊天记录存储机制:")
        print("1. 实时对话上下文: 存储在内存中的msg_data列表")
        print("2. 持久化聊天历史: 保存到./data/agents/{角色名}/history.yaml")
        print("3. 长期记忆(日记): 通过long_mem模块存储到PostgreSQL")
        print("4. 核心记忆: 通过core_mem模块存储到PostgreSQL")
        print("5. 知识库: 通过lore_books存储到PostgreSQL")
        
    except Exception as e:
        print(f"检查过程中出现错误: {e}")

if __name__ == "__main__":
    main()