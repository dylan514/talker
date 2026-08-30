"""智能客服在线服务：FastAPI + RAG。

用法：
    python app.py
    然后 POST http://127.0.0.1:8000/ask  {"question": "音箱防水吗？"}
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import TOP_K, chat, retrieve

load_dotenv()  # 从 .env 读取 DEEPSEEK_API_KEY

app = FastAPI(title="Talker 智能客服", version="0.2.0")

INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    history: list[dict] = Field(default_factory=list, description="对话历史 [{role, content}]")
    top_k: int = Field(TOP_K, ge=1, le=10, description="召回片段数")


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]  # [{source, chunk_idx, text, distance}]


@app.get("/")
def index():
    """聊天页面。"""
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    chunks = retrieve(req.question, top_k=req.top_k)
    answer = chat(req.question, chunks, history=req.history)
    return AskResponse(
        answer=answer,
        sources=[
            {"source": c["source"], "chunk_idx": c["chunk_idx"],
             "text": c["text"], "distance": c["distance"]}
            for c in chunks
        ],
    )


if __name__ == "__main__":
    import uvicorn

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("警告：未设置 DEEPSEEK_API_KEY，/ask 接口将无法生成回答。")
        print("请在 .env 文件中填入：DEEPSEEK_API_KEY=sk-xxx")
    # 绑定 0.0.0.0，局域网内其他设备可访问（演示用）
    uvicorn.run(app, host="0.0.0.0", port=8000)
