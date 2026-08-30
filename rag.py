"""RAG 核心模块：切分、向量化、检索、Prompt 组装、LLM 调用。

离线建库用 ingest.py，在线问答用 app.py，公共逻辑都放在这里。
"""

import os
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500          # 片段目标长度（字符）
CHUNK_OVERLAP = 80        # 相邻片段重叠长度（字符）
TOP_K = 4                 # 默认召回片段数

DOC_DIR = Path(__file__).parent / "data"       # 产品文档目录
CHROMA_DIR = Path(__file__).parent / "chroma_db"  # 向量库持久化目录
COLLECTION_NAME = "product_kb"

# bge 系列模型的查询侧指令前缀（官方推荐，能显著提升检索效果）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# bge-m3 模型在 HuggingFace 上的仓库 id
EMBED_MODEL_ID = "BAAI/bge-m3"

SYSTEM_PROMPT = """你是星语科技的智能客服助手，负责解答用户关于 Talker 智能音箱产品的问题。

请严格遵守以下规则：
1. 只根据【产品资料】中的内容回答，不要编造资料中没有的信息。
2. 如果资料中找不到答案，请如实告知用户无法回答，并建议联系人工客服，此时不要标注参考编号。
3. 回答要简洁、准确、口语化，直接回答用户的问题，不要复述资料原文。
4. 只有当回答使用了资料内容时，才在末尾用「参考：[1][2]」的格式标注所引用的资料编号。"""


# ---------------------------------------------------------------------------
# 文档加载与切分
# ---------------------------------------------------------------------------

def load_documents(doc_dir: Path) -> list[dict]:
    """读取目录下所有 .txt / .md 文档，返回 [{source, text}]。"""
    docs = []
    for path in sorted(doc_dir.glob("*")):
        if path.suffix.lower() in (".txt", ".md"):
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按句切分后贪心合并为不超过 chunk_size 的片段，相邻片段保留 overlap 重叠。

    超长单句会被硬切。返回片段列表。
    """
    text = text.strip()
    # 按句末标点或换行切句
    sentences = [s for s in re.split(r"(?<=[。！？；!?;])\s*|\n+", text) if s.strip()]

    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= chunk_size:
            cur += s
            continue
        # 当前句放不下：收尾当前片段，带回 overlap 开新片段
        if cur:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap < len(cur) else cur
        if len(s) > chunk_size:
            # 超长单句硬切（不带重叠，避免内容翻倍）
            chunks.extend(s[i:i + chunk_size] for i in range(0, len(s), chunk_size))
            cur = ""
        else:
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

_embed_model = None


def get_embed_model() -> SentenceTransformer:
    """懒加载 embedding 模型（进程内单例）。"""
    global _embed_model
    if _embed_model is None:
        device = "cuda" if _has_cuda() else "cpu"
        _embed_model = SentenceTransformer(EMBED_MODEL_ID, device=device)
    return _embed_model


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化（文档侧，不加查询指令）。"""
    return get_embed_model().encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    """向量化用户问题（加查询侧指令前缀）。"""
    return get_embed_model().encode(
        QUERY_INSTRUCTION + query, normalize_embeddings=True
    ).tolist()


# ---------------------------------------------------------------------------
# 向量库
# ---------------------------------------------------------------------------

_client = None


def get_collection() -> chromadb.Collection:
    """懒加载 Chroma 持久化客户端与集合。"""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def ingest_documents() -> int:
    """离线建库：加载文档 → 切分 → 向量化 → 写入 Chroma。返回片段总数。"""
    collection = get_collection()
    docs = load_documents(DOC_DIR)
    if not docs:
        raise FileNotFoundError(f"{DOC_DIR} 下没有 .txt/.md 文档")

    ids, texts, embeddings, metadatas = [], [], [], []
    for doc in docs:
        chunks = split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['source']}::{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_idx": i})

    embeddings = embed_texts(texts)

    # 重建集合，保证幂等（重复执行不会残留旧数据）
    _client.delete_collection(COLLECTION_NAME)
    collection = _client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(ids)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """检索 Top-K 相关片段，返回 [{source, chunk_idx, text, distance}]。"""
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "source": meta["source"],
            "chunk_idx": meta["chunk_idx"],
            "text": text,
            "distance": dist,
        }
        for text, meta, dist in zip(result["documents"][0],
                                    result["metadatas"][0],
                                    result["distances"][0])
    ]


# ---------------------------------------------------------------------------
# Prompt 组装与 LLM 调用
# ---------------------------------------------------------------------------

def build_messages(question: str, chunks: list[dict],
                   history: list[dict] | None = None) -> list[dict]:
    """组装 LLM 消息：system 约束 + 对话历史 + 当前问题（带检索资料）。"""
    materials = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # 最多携带最近 6 轮历史，控制 token 消耗
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"【产品资料】\n{materials}\n\n【用户问题】\n{question}",
    })
    return messages


def chat(question: str, chunks: list[dict],
         history: list[dict] | None = None) -> str:
    """调用 DeepSeek 生成回答（OpenAI 兼容接口）。"""
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，请先写入 .env 文件或环境变量")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=build_messages(question, chunks, history),
        temperature=0.3,
        max_tokens=1000,
    )
    return resp.choices[0].message.content
