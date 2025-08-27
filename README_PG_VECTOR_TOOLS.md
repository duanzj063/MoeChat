# PostgreSQL向量库查询工具

本目录包含了几个用于查询和管理PostgreSQL向量库的工具，帮助您检查数据库状态、查看数据内容和测试向量操作。

## 工具列表

### 1. `pg_vector_manager.py` - 综合管理工具 ⭐推荐

最完整的PostgreSQL向量库管理工具，支持多种操作：

```bash
# 查看数据库状态（默认操作）
python pg_vector_manager.py
python pg_vector_manager.py status

# 列出所有表的数据
python pg_vector_manager.py list

# 列出指定表的数据
python pg_vector_manager.py list --table test_remote

# 限制显示行数
python pg_vector_manager.py list --limit 5

# 清空指定表（需要确认）
python pg_vector_manager.py clear --table test_remote

# 自动确认清空操作
python pg_vector_manager.py clear --table test_remote --yes

# 测试向量操作
python pg_vector_manager.py test
```

### 2. `simple_pg_query.py` - 简化查询工具

快速查看数据库状态的轻量级工具：

```bash
python simple_pg_query.py
```

输出示例：
```
🔍 PostgreSQL向量库数据检查
==================================================
✅ pgvector扩展: vector v0.8.0

📊 数据库表统计 (共2个表):
  ⚪ lore_books_姬子: 0 行, 6 列 [向量列: 1]
  📊 test_remote: 5 行, 6 列 [向量列: 1]
    └─ 最新数据: 2025-08-27 05:00:24.217386

📈 总计: 5 行数据
```

### 3. `check_vector_data.py` - 数据内容检查工具

查看向量数据的具体内容：

```bash
python check_vector_data.py
```

显示每条记录的详细信息，包括ID、文本、元数据和创建时间。

### 4. `query_pg_vector.py` - 详细查询工具

最详细的数据库检查工具，显示完整的表结构和数据：

```bash
python query_pg_vector.py
```

## 当前数据库状态

根据最新检查结果：

- **pgvector扩展**: ✅ 已安装 (v0.8.0)
- **向量维度**: 1024
- **数据表**:
  - `lore_books_姬子`: 0行数据（角色知识库表，当前为空）
  - `test_remote`: 5行测试数据（用于验证向量功能）

## 测试结果

向量操作测试显示：
- ✅ 向量相似度搜索正常工作
- ✅ 向量内积计算正常工作
- ✅ 1024维向量存储和检索正常

## 数据分析

### 现有数据
- `test_remote`表包含5条AI相关的测试文本
- 所有向量数据都是1024维度
- 向量搜索功能正常工作

### 缺失数据
- `lore_books_姬子`表为空，说明还没有为角色"姬子"创建知识库数据
- 需要通过与AI对话或手动导入来生成实际应用数据

## 下一步建议

1. **生成实际数据**: 与AI进行对话，触发向量数据的生成
2. **检查应用配置**: 确认应用程序正确配置为远程模式
3. **监控数据增长**: 定期使用这些工具检查数据库状态

## 故障排除

如果遇到连接问题：
1. 检查`config.yaml`中的数据库配置
2. 确认PostgreSQL服务器可访问
3. 验证用户名和密码正确
4. 检查防火墙设置

如果向量操作失败：
1. 确认pgvector扩展已安装
2. 检查向量维度是否一致
3. 验证表结构是否正确

## 配置信息

工具会自动从`config.yaml`读取数据库配置：

```yaml
VectorStore:
  remote:
    db_config:
      host: "27.159.93.61"
      port: 5432
      database: "postgres"
      user: "root"
      password: "Fsti<#>2025"
```

确保配置文件存在且格式正确。