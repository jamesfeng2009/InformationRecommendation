# Elasticsearch 索引实现文档

## 概述

本文档描述了智能信息推荐系统中 Elasticsearch 索引的实现，包括索引映射配置、新闻文档索引服务以及零停机时间重建索引的支持。

## 功能特性

### 1. 索引映射配置 (Requirements: 19.4)

- **中文分词器**: 使用 `ik_max_word` 分词器进行索引，`ik_smart` 分词器进行搜索
- **字段类型配置**: 为新闻的各个字段配置合适的类型和分析器
- **索引别名**: 支持索引别名以实现零停机时间重建索引

### 2. 新闻索引服务 (Requirements: 19.2)

- **创建时索引**: 新闻创建时自动索引到 Elasticsearch
- **修改时更新**: 新闻修改时自动更新索引
- **批量索引**: 支持批量索引用于初始数据加载

## 架构设计

### 核心组件

```
app/
├── core/
│   └── elasticsearch.py          # Elasticsearch 客户端配置
├── services/
│   ├── elasticsearch_index.py    # 索引管理服务
│   ├── news_indexing.py          # 新闻索引服务
│   └── news.py                   # 新闻业务服务
└── api/
    ├── news.py                   # 新闻 API 端点
    └── elasticsearch_admin.py    # 索引管理 API 端点
```

### 数据流

```
创建新闻:
  用户请求 → API → NewsService.create_news() → PostgreSQL
                                              → NewsIndexingService.index_news() → Elasticsearch

更新新闻:
  用户请求 → API → NewsService.update_news() → PostgreSQL
                                              → NewsIndexingService.update_news() → Elasticsearch

批量索引:
  管理员请求 → API → NewsService.batch_index_news() → 从 PostgreSQL 读取
                                                    → NewsIndexingService.bulk_index_news() → Elasticsearch
```

## 索引映射

### 字段配置

| 字段名 | 类型 | 分析器 | 说明 |
|--------|------|--------|------|
| id | integer | - | 新闻ID |
| title | text | ik_max_word | 新闻标题（中文分词） |
| content | text | ik_max_word | 新闻正文（中文分词） |
| translated_title | text | ik_max_word | 翻译后的标题 |
| translated_content | text | ik_max_word | 翻译后的正文 |
| summary | text | ik_max_word | 摘要 |
| keywords | keyword | - | 关键词数组 |
| category | keyword | - | 分类 |
| source_name | keyword | - | 来源名称 |
| author | keyword | - | 作者 |
| location | keyword | - | 地理位置 |
| language | keyword | - | 语言 |
| publish_time | date | - | 发布时间 |
| crawl_time | date | - | 爬取时间 |
| hot_score | float | - | 热度分数 |
| content_hash | keyword | - | 内容哈希 |

### 索引设置

```json
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
      "analyzer": {
        "ik_max_word": {
          "type": "custom",
          "tokenizer": "ik_max_word"
        },
        "ik_smart": {
          "type": "custom",
          "tokenizer": "ik_smart"
        }
      }
    }
  }
}
```

## API 使用指南

### 初始化索引

首次部署时需要初始化 Elasticsearch 索引和别名：

```bash
POST /api/v1/admin/elasticsearch/initialize
```

响应：
```json
{
  "success": true,
  "message": "Elasticsearch index and alias initialized successfully",
  "index_name": "news_v1",
  "alias_name": "news"
}
```

### 创建新闻（自动索引）

```bash
POST /api/v1/news
Content-Type: application/json

{
  "title": "测试新闻标题",
  "content": "这是新闻内容...",
  "source_url": "https://example.com/news/1",
  "source_name": "测试新闻源",
  "language": "zh",
  "category": "军事",
  "keywords": ["测试", "新闻"],
  "summary": "这是摘要"
}
```

### 更新新闻（自动更新索引）

```bash
PATCH /api/v1/news/{news_id}
Content-Type: application/json

{
  "category": "政治",
  "summary": "更新后的摘要"
}
```

### 批量索引现有数据

用于初始数据加载或重建索引：

```bash
POST /api/v1/news/batch-index
Content-Type: application/json

{
  "limit": 1000,
  "offset": 0
}
```

响应：
```json
{
  "success": 950,
  "errors": 50,
  "total": 1000,
  "message": "Indexed 950 news items successfully"
}
```

### 重建所有索引

```bash
POST /api/v1/news/reindex-all
```

## 零停机时间重建索引

### 步骤

1. **创建新索引**
```bash
POST /api/v1/admin/elasticsearch/index
Content-Type: application/json

{
  "index_name": "news_v2"
}
```

2. **重建索引数据**
```bash
POST /api/v1/admin/elasticsearch/reindex
Content-Type: application/json

{
  "source_index": "news_v1",
  "dest_index": "news_v2"
}
```

3. **检查重建状态**
```bash
GET /api/v1/admin/elasticsearch/reindex/status/{task_id}
```

4. **切换别名**
```bash
POST /api/v1/admin/elasticsearch/alias
Content-Type: application/json

{
  "index_name": "news_v2"
}
```

5. **删除旧索引（可选）**
```bash
DELETE /api/v1/admin/elasticsearch/index/news_v1
```

### 优势

- **零停机**: 别名切换是原子操作，用户无感知
- **回滚能力**: 如果新索引有问题，可以快速切回旧索引
- **数据安全**: 旧索引在确认新索引正常后才删除

## 环境配置

在 `.env` 文件中配置 Elasticsearch 连接：

```bash
# Elasticsearch settings
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX_PREFIX=news
ELASTICSEARCH_TIMEOUT=30
ELASTICSEARCH_MAX_RETRIES=3
```

## 性能优化

### 批量索引优化

- **分块大小**: 默认 500 条/批次，可根据文档大小调整
- **并发控制**: 使用 `async_bulk` 实现异步批量索引
- **错误处理**: 单个文档失败不影响其他文档

### 索引刷新

```bash
POST /api/v1/admin/elasticsearch/refresh
```

手动刷新索引使最新更改立即可搜索（默认 1 秒自动刷新）。

## 监控与维护

### 检查文档数量

```python
from app.services.news_indexing import NewsIndexingService

count = await indexing_service.count_documents()
print(f"Total documents: {count}")
```

### 检查文档是否存在

```python
exists = await indexing_service.exists(news_id=123)
```

### 获取文档内容

```python
doc = await indexing_service.get_document(news_id=123)
```

## 故障排查

### 问题：索引创建失败

**可能原因**：
- Elasticsearch 服务未启动
- IK 分词器插件未安装
- 索引已存在

**解决方案**：
1. 检查 Elasticsearch 服务状态
2. 安装 IK 分词器：`elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v8.12.0/elasticsearch-analysis-ik-8.12.0.zip`
3. 删除已存在的索引或使用不同的索引名

### 问题：批量索引速度慢

**优化建议**：
- 增加分片数量（`number_of_shards`）
- 调整批量大小（`chunk_size`）
- 临时禁用副本（`number_of_replicas: 0`），索引完成后再启用
- 增加刷新间隔（`refresh_interval: 30s`）

### 问题：中文分词效果不佳

**解决方案**：
- 确认 IK 分词器正确安装
- 更新 IK 词典
- 考虑使用自定义词典

## 测试

运行单元测试：

```bash
pytest tests/unit/test_elasticsearch_indexing.py -v
```

测试覆盖：
- 索引创建和删除
- 别名管理
- 单个文档索引
- 批量索引
- 文档更新和删除
- 零停机时间重建索引

## 依赖要求

- Python >= 3.11
- Elasticsearch >= 8.12.0
- elasticsearch[async] >= 8.12.0
- IK 分词器插件

## 参考资料

- [Elasticsearch 官方文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [IK 分词器](https://github.com/medcl/elasticsearch-analysis-ik)
- [Python Elasticsearch 客户端](https://elasticsearch-py.readthedocs.io/)
