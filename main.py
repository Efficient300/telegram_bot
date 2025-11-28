import os
import asyncio
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from chroma_store import ChromaStore
from prompts import SYSTEM_PROMPT

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Please set BOT_TOKEN in .env")

# Chroma persist directory (можно менять)
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")

# Инициализация бота
dp = Dispatcher()
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Инициализация Chroma store (обёртка)
chroma = ChromaStore(persist_directory=CHROMA_DIR)

# Фабрика LLM (можно вынести в отдельную функцию/класс)
def make_llm():
    # Обрати внимание: параметры модели подставь свои по необходимости
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_retries=2,
    )

llm = make_llm()


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        f"Привет, {html.bold(message.from_user.full_name)}!\n"
        "Я бот с контекстной памятью (Chroma) + Gemini.\n"
        "Просто напиши сообщение — я отвечу и сохраню историю."
    )


@dp.message()
async def handle_message(message: Message) -> None:
    """
    Основной обработчик текста:
    1. Сохраняет user message в Chroma
    2. Достаёт релевантный контекст из Chroma (RAG)
    3. Формирует messages: SystemMessage (PROMPT) + Context (как System/Hints) + HumanMessage
    4. Вызывает LLM и отправляет ответ
    5. Сохраняет ответ в Chroma
    """
    user_text = message.text or ""
    if not user_text.strip():
        # Для нетекстовых типов можно отправить копию/ошибку
        try:
            await message.send_copy(chat_id=message.chat.id)
        except Exception:
            await message.answer("Поддерживаются только текстовые сообщения.")
        return

    # 1) Сохранить пользовательское сообщение в Chroma
    ts = datetime.utcnow().isoformat()
    chroma.add_message(role="user", content=user_text, metadata={"chat_id": str(message.chat.id), "ts": ts})

    # 2) Получить релевантный контекст (k верхних)
    context_docs = chroma.get_relevant(user_text, k=4)  # список строк

    # 3) Сформировать сообщения для LLM
    # SystemMessage с PROMPT
    system_msg = SystemMessage(content=SYSTEM_PROMPT)

    # Вставим найденный контекст как дополнительный SystemMessage (или можно как assistant/hint)
    if context_docs:
        # Соберём контекст в одну строку — коротко
        context_text = "\n\n".join(context_docs)
        # Пометим, что это прошлые сообщения (metadata chat etc handled in chroma)
        system_context_msg = SystemMessage(content=f"Релевантный контекст из истории:\n{context_text}")
        messages = [system_msg, system_context_msg, HumanMessage(content=user_text)]
    else:
        messages = [system_msg, HumanMessage(content=user_text)]

    # 4) Вызов LLM (в отдельном потоке — чтобы не блокировать aiogram loop)
    try:
        # llm.invoke может быть блокирующим, поэтому выполняем в worker
        response = await asyncio.to_thread(lambda: llm.invoke(messages))
        # response может быть объектом, у которого .content хранит текст
        # Подстраховываемся: если это dict-like — извлечём подходящее поле
        ai_text = ""
        if hasattr(response, "content"):
            ai_text = response.content
        elif isinstance(response, dict):
            # Иногда langchain возвращает {"content": "..."} или {"output": "..."}
            ai_text = response.get("content") or response.get("output") or str(response)
        else:
            ai_text = str(response)
    except Exception as e:
        logger.exception("LLM call failed")
        await message.answer(f"Ошибка при обращении к LLM: {e}")
        return

    # 5) Отправить ответ юзеру
    try:
        # Ограничиваем длину ответа в telegram (по желанию)
        await message.answer(ai_text)
    except Exception:
        # Если слишком большой, можно отправить частями
        for chunk in (ai_text[i:i + 4000] for i in range(0, len(ai_text), 4000)):
            await message.answer(chunk)

    # 6) Сохранить ответ ассистента в Chroma
    ts2 = datetime.utcnow().isoformat()
    chroma.add_message(role="assistant", content=ai_text, metadata={"chat_id": str(message.chat.id), "ts": ts2})

    # Сохраняем persist (на всякий случай; ChromaStore вызывает persist внутри)
    chroma.persist()


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await bot.delete_webhook()
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
