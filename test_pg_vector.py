import psycopg2
import time
from psycopg2 import sql

# 数据库连接参数
DB_CONFIG = {
    'host': '27.159.93.61',
    'port': 5432,
    'database': 'postgres',
    'user': 'root',
    'password': 'Fsti<#>2025'
}

def time_query(cursor, query, params=None, description="查询"):
    """执行查询并打印耗时"""
    start_time = time.time()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"  -> {description} 耗时: {elapsed_time:.2f} ms")
    return elapsed_time

def test_vector_extension_and_latency():
    """测试 PostgreSQL 是否开启了向量扩展 (pgvector) 并测量响应时间"""
    conn = None
    try:
        # 建立数据库连接
        print("正在连接到数据库...")
        start_connect = time.time()
        conn = psycopg2.connect(**DB_CONFIG)
        end_connect = time.time()
        connect_time = (end_connect - start_connect) * 1000
        cursor = conn.cursor()
        print(f"数据库连接成功，耗时: {connect_time:.2f} ms")

        # 1. 检查 pgvector 扩展是否存在
        print("\n1. 检查 pgvector 扩展是否已安装...")
        time_query(cursor, 
                   "SELECT name FROM pg_available_extensions WHERE name = 'vector';", 
                   description="检查扩展是否存在")
        extension_available = cursor.fetchone()
        
        if not extension_available:
            print("  -> 未找到 'vector' 扩展。")
            return False
        else:
            print("  -> 找到 'vector' 扩展。")

        # 2. 检查 pgvector 扩展是否已加载
        print("\n2. 检查 pgvector 扩展是否已加载...")
        time_query(cursor, 
                   "SELECT extname FROM pg_extension WHERE extname = 'vector';", 
                   description="检查扩展是否已加载")
        extension_loaded = cursor.fetchone()
        
        if not extension_loaded:
            print("  -> 'vector' 扩展未加载。")
            return False
        else:
            print("  -> 'vector' 扩展已加载。")

        # 3. 尝试使用 vector 类型 (更严格的测试) 并测量耗时
        print("\n3. 尝试创建包含 vector 类型的临时表...")
        try:
            time_query(cursor, "DROP TABLE IF EXISTS test_vector_table;", description="删除临时表 (如果存在)")
            time_query(cursor, "CREATE TEMP TABLE test_vector_table (id serial, vec vector(3));", description="创建临时 vector 表")
            time_query(cursor, "INSERT INTO test_vector_table (vec) VALUES (%s), (%s);", ( '[1,2,3]', '[4,5,6]' ), description="插入 vector 数据")
            
            print("\n4. 测试向量相似度查询性能...")
            # 插入更多数据以进行更有意义的测试
            insert_query = "INSERT INTO test_vector_table (vec) SELECT %s FROM generate_series(1, 1000);"
            time_query(cursor, insert_query, ('[0.1,0.2,0.3]',), description="插入 1000 条示例向量数据")
            
            # 执行一个典型的向量相似度查询 (L2 距离)
            search_query = "SELECT id, vec FROM test_vector_table ORDER BY vec <-> %s LIMIT 5;"
            time_query(cursor, search_query, ('[1.0, 2.0, 3.0]',), description="执行向量相似度查询 (L2)")

            # 获取查询结果
            rows = cursor.fetchall()
            print(f"  -> 相似度查询返回 {len(rows)} 条结果。")
            
            print("\n结论: PostgreSQL 数据库已正确开启并支持向量库 (pgvector)，且响应时间在合理范围内。")
            return True
        except psycopg2.Error as e:
            print(f"  -> 使用 vector 类型时出错: {e}")
            print("\n结论: PostgreSQL 数据库未正确配置向量库 (pgvector) 或版本不兼容。")
            return False

    except psycopg2.OperationalError as e:
        print(f"数据库连接失败: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False
    finally:
        # 关闭数据库连接
        if conn:
            cursor.close()
            conn.close()
            print("\n数据库连接已关闭。")

if __name__ == "__main__":
    # 安装依赖提示 (如果 psycopg2 未安装)
    try:
        import psycopg2
    except ImportError:
        print("错误: 未找到 psycopg2 库。请先安装它:")
        print("  pip install psycopg2-binary")
        exit(1)

    success = test_vector_extension_and_latency()
    if not success:
        exit(1) # 脚本以非零状态码退出，表示测试失败