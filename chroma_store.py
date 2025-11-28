"""
Простая обёртка для ChromaDB + Google embeddings.
Хранит сообщения как отдельные документы:
 - текст = "<role>: <content>"
 - metadata содержит role, chat_id, ts
При поиске возвращаем page_content (строки) верхних k документов.
"""
from typing import List, Optional
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaStore:
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.persist_directory = persist_directory
        # Embeddings - укажи модель если нужно
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Инициализируем Chroma
        self.chroma = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="chat_history"
        )
        # You may need to create collection first - Chroma wrapper will handle.

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Добавляет один документ в Chroma."""
        text = f"{role}: {content}"
        meta = metadata or {}
        meta.update({"role": role, "ts": metadata.get("ts") if metadata else datetime.utcnow().isoformat()})
        # Chroma.add_texts принимает список
        self.chroma.add_texts([text], metadatas=[meta], ids=None)

    def get_relevant(self, query: str, k: int = 4) -> List[str]:
        """Возвращает список page_content наиболее релевантных документов (строки)."""
        try:
            docs = self.chroma.similarity_search(query, k=k)
            return [d.page_content for d in docs]
        except Exception:
            return []

    def persist(self):
        """Принудительно записать на диск, если требуется."""
        try:
            self.chroma.persist()
        except Exception:
            pass
