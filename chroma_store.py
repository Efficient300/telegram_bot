"""
Простая обёртка для ChromaDB + Google embeddings.
Хранит сообщения как отдельные документы:
 - текст = "<role>: <content>"
 - metadata содержит role, chat_id, ts (timestamp)
 При поиске возвращаем page_content (строки) верхних k документов.
"""
from typing import List, Optional
from datetime import datetime, timezone
import logging

# Обновленные импорты для LangChain 0.2.x (Community и HuggingFace)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class ChromaStore:
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.persist_directory = persist_directory
        
        # Embeddings: Используем ту же модель для совместимости с вашими настройками
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Инициализируем Chroma
        # NOTE: LangChain будет выдавать DeprecationWarning, если не использовать langchain_chroma
        self.chroma = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="chat_history"
        )

    def _get_current_ts(self) -> str:
        """Возвращает текущее время UTC в формате ISO (без предупреждений)."""
        # Использование datetime.now(timezone.utc) вместо устаревшего datetime.utcnow()
        return datetime.now(timezone.utc).isoformat()

    def add_message(self, role: str, content: str, metadata: Optional[dict] = None):
        """Добавляет один документ в Chroma."""
        text = f"{role}: {content}"
        
        # 1. Объединяем и обновляем метаданные
        meta = metadata.copy() if metadata else {}
        
        # 2. Убеждаемся, что role и ts присутствуют
        meta["role"] = role
        # Если ts не передан, генерируем его сейчас
        if "ts" not in meta:
            meta["ts"] = self._get_current_ts()
        
        try:
            # Chroma.add_texts принимает список
            self.chroma.add_texts([text], metadatas=[meta], ids=None)
        except Exception as e:
            logger.error(f"Ошибка при добавлении сообщения в Chroma: {e}")

    def get_relevant(self, query: str, k: int = 4) -> List[str]:
        """Возвращает список page_content наиболее релевантных документов (строки)."""
        try:
            # Используем similarity_search для поиска по истории
            docs = self.chroma.similarity_search(query, k=k)
            
            # Опционально: можно отсортировать по времени (ts) для более логичного RAG
            # docs.sort(key=lambda x: x.metadata.get("ts", ""), reverse=True)
            
            return [d.page_content for d in docs]
        except Exception as e:
            logger.error(f"Ошибка при поиске релевантных документов в Chroma: {e}")
            return []

    def persist(self):
        """
        Метод persist() теперь пустой. 
        ChromaDB (начиная с версии 0.4.x) сохраняет данные автоматически.
        """
        pass