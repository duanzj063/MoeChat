#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查向量数据内容的工具
"""

import psycopg2
import yaml
import json
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
    print("🔍 检查向量数据内容")
    print("=" * 50)
    
    try:
        config = load_config()
        conn = connect_db(config)
        cursor = conn.cursor()
        
        # 检查test_remote表的数据
        print("\n📊 test_remote表数据详情:")
        cursor.execute("""
            SELECT id, text, metadata, created_at
            FROM test_remote 
            ORDER BY created_at DESC;
        """)
        
        rows = cursor.fetchall()
        for i, (id, text, metadata, created_at) in enumerate(rows, 1):
            print(f"\n记录 {i}:")
            print(f"  ID: {id}")
            print(f"  文本: {text}")
            print(f"  元数据: {json.dumps(metadata, ensure_ascii=False) if metadata else 'None'}")
            print(f"  创建时间: {created_at}")
        
        # 检查向量维度
        print("\n🔢 向量维度检查:")
        cursor.execute("SELECT vector_dims(vector) as dims FROM test_remote LIMIT 1;")
        dims_result = cursor.fetchone()
        if dims_result:
            print(f"  向量维度: {dims_result[0]}")
        
        # 检查lore_books_姬子表
        print("\n📚 lore_books_姬子表状态:")
        cursor.execute("SELECT COUNT(*) FROM lore_books_姬子;")
        count = cursor.fetchone()[0]
        print(f"  数据行数: {count}")
        
        if count == 0:
            print("  ⚠️  该表为空，可能原因:")
            print("    - 还没有为角色'姬子'创建知识库数据")
            print("    - 需要手动导入或通过对话生成数据")
        
        cursor.close()
        conn.close()
        
        print("\n💡 数据分析:")
        if rows:
            print(f"  - test_remote表有 {len(rows)} 条测试数据")
            print("  - 这些是测试向量数据，用于验证系统功能")
            print("  - 实际应用数据应该存储在lore_books_姬子等表中")
        
        print("\n🎯 下一步建议:")
        print("  1. 与AI进行对话，触发向量数据的生成")
        print("  2. 检查应用程序是否正确处理用户输入")
        print("  3. 查看是否有其他角色相关的表")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()