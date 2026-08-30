# Talker 智能客服（RAG Pipeline）

基于「文档检索 + 大模型生成」的产品智能客服：

- 离线：产品文档 → 切分 → bge-m3 向量化 → Chroma
- 在线：用户问题 → 向量检索 Top-K → 拼 Prompt → DeepSeek 生成 → 返回答案+引用

## 环境准备

```bash
conda create -n talker python=3.11 -y
conda activate talker
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 模型下载（huggingface.co 不可达时用镜像）
export HF_ENDPOINT=https://hf-mirror.com

# DeepSeek API Key
cp .env.example .env   # 然后编辑 .env 填入真实 Key
```

## 使用

```bash
# 1. 离线建库（文档放 data/ 目录，.txt / .md 均可）
python ingest.py

# 2. 启动服务（绑定 0.0.0.0，局域网可访问）
python app.py

# 3. 浏览器打开聊天页面
#    http://127.0.0.1:8000/   或   http://<本机IP>:8000/
#    支持多轮追问（如「Pro 多少钱？」→「那标准版呢？」）

# 4. 或用 API 提问
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "音箱防水吗？"}'
```

API 返回示例：`{"answer": "...", "sources": [{"source": "sample_faq.txt", "chunk_idx": 8, ...}]}`
`/ask` 支持可选字段 `history`（对话历史 `[{role, content}]`）和 `top_k`（召回数）。

## 目录结构

```
data/          产品文档（换真实文档时直接替换）
rag.py         公共逻辑：切分 / 向量化 / 检索 / Prompt / LLM
ingest.py      离线建库脚本
app.py         FastAPI 服务（/ask、/health）
chroma_db/     向量库（ingest 后生成）
```

## 调参

| 参数 | 位置 | 说明 |
|---|---|---|
| CHUNK_SIZE / CHUNK_OVERLAP | rag.py | 片段长度与重叠，默认 500/80 |
| TOP_K | rag.py 或 /ask 的 top_k 字段 | 召回片段数，默认 4 |
| SYSTEM_PROMPT | rag.py | 客服角色与回答约束 |

## 后续可扩展

- 混合检索：关键词检索（BM25）与向量检索加权融合
- 重排序：bge-reranker 对召回结果二次排序
- 多轮对话：携带历史问题改写当前 query
- 日志与评估：记录问答对，人工标注后评估检索命中率
