"""离线建库脚本：文档 → 切分 → Embedding → 写入 Chroma。

用法：
    python ingest.py
"""

import time

from rag import DOC_DIR, ingest_documents

if __name__ == "__main__":
    start = time.time()
    print(f"正在读取 {DOC_DIR} 下的文档...")
    n = ingest_documents()
    print(f"完成：共写入 {n} 个片段，耗时 {time.time() - start:.1f}s")
    print("向量库已保存到 ./chroma_db，可以启动服务了：python app.py")
