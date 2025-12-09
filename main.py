import os
import asyncio
import logging
import sys
import base64
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, Voice
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# import mimetypes # Не требуется, если мы жестко задаем 'audio/ogg' для голосовых Telegram

# Импорты ваших локальных модулей
# Убедитесь, что файлы chroma_store.py и prompts.py находятся в том же каталоге
from chroma_store import ChromaStore
from prompts import SYSTEM_PROMPT

# ================== Настройки ==================
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TOKEN:
    raise RuntimeError("Please set BOT_TOKEN in .env")
if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in .env")

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
os.makedirs(CHROMA_DIR, exist_ok=True)

# Логирование
logging.basicConfig(
    level=logging.INFO, 
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Инициализация бота
dp = Dispatcher()
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

# Кэш для хранилищ пользователей
user_chromas = {}

# Инициализация LLM
def make_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_retries=2,
        api_key=GOOGLE_API_KEY
    )

llm = make_llm()

# ================== Утилиты ==================

async def run_sync(func, *args, **kwargs) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)

async def get_user_chroma(chat_id: str) -> ChromaStore:
    if chat_id not in user_chromas:
        user_dir = os.path.join(CHROMA_DIR, f"user_{chat_id}")
        await run_sync(os.makedirs, user_dir, exist_ok=True)
        user_chromas[chat_id] = await run_sync(ChromaStore, persist_directory=user_dir)
    return user_chromas[chat_id]

def get_utc_now_iso():
    """Возвращает текущее время UTC в формате ISO (без предупреждений)."""
    return datetime.now(timezone.utc).isoformat()

async def send_long_message(message: Message, text: str):
    if not text: return
    try:
        # Разбиваем на чанки по 4000 символов для соответствия лимиту Telegram
        for chunk in (text[i:i + 4000] for i in range(0, len(text), 4000)):
            await message.answer(chunk)
    except Exception as e:
        logger.error(f"Error sending msg: {e}")

# ================== Хендлеры ==================

@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        f"Привет, {html.bold(message.from_user.full_name)}!\n"
        "Отправь мне **фото**, **голосовое сообщение** или **текст**, и я отвечу."
    )
    
@dp.message(F.voice)
async def handle_voice(message: Message):
    """
    Обработка голосовых сообщений (STT + LLM) с использованием Gemini.
    """
    chat_id_str = str(message.chat.id)
    voice: Voice = message.voice
    
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    status_msg = await message.answer("🎧 Распознаю голосовое и думаю...")

    try:
        # 1. Скачиваем голосовое сообщение в память (OGG/Opus)
        voice_file = await bot.get_file(voice.file_id)
        voice_bytes_io = BytesIO()
        await bot.download_file(voice_file.file_path, voice_bytes_io)
        voice_data = voice_bytes_io.getvalue()
        
        # 2. Кодируем в Base64 для Gemini
        b64_audio = base64.b64encode(voice_data).decode('utf-8')
        mime_type = 'audio/ogg' 
        
        # 3. Поиск контекста
        user_chroma = await get_user_chroma(chat_id_str)
        context_docs = await run_sync(user_chroma.get_relevant, "Предыдущий разговор", k=4) 
        
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system_msg]

        if context_docs:
            context_text = "\n---\n".join(context_docs)
            messages.append(SystemMessage(content=f"Контекст из памяти:\n{context_text}"))
        
        # 4. Подготовка мультимодального сообщения (ИСПРАВЛЕНО)
        # Используем универсальный формат: {'data': Base64, 'mime_type': MIME}
        message_content = [
            {"type": "text", "text": "Расшифруй это голосовое сообщение и ответь на него, используя контекст нашей предыдущей беседы. Сначала дай расшифровку, а потом ответ."},
            {
                "data": b64_audio, 
                "mime_type": mime_type # 'audio/ogg'
            }
        ]
        
        human_msg = HumanMessage(content=message_content)
        messages.append(human_msg)

        # 5. Запрос к LLM (STT + Чат)
        ai_response = await llm.ainvoke(messages)
        ai_text = ai_response.content
        
        # 6. Отправляем ответ пользователю
        await status_msg.delete()
        await send_long_message(message, ai_text)

        # 7. Сохраняем в память (RAG)
        # Сохраняем "запрос" пользователя (факт отправки аудио + ответ ИИ как контекст)
        save_user_content = f"[Пользователь отправил голосовое сообщение]. Расшифровка и ответ: {ai_text}"
        ts = get_utc_now_iso()
        await run_sync(user_chroma.add_message, role="user", content=save_user_content, metadata={"ts": ts})
        
        # Сохраняем ответ ассистента
        ts2 = get_utc_now_iso()
        await run_sync(user_chroma.add_message, role="assistant", content=ai_text, metadata={"ts": ts2})
        
        try:
            await run_sync(user_chroma.persist)
        except AttributeError:
            pass 

    except Exception as e:
        logger.exception("Ошибка при обработке голосового сообщения")
        await status_msg.edit_text(f"⚠️ Ошибка при обработке голосового: {e}")


@dp.message(F.photo)
async def handle_photo(message: Message):
    """Прямая обработка фото через Gemini Vision."""
    chat_id_str = str(message.chat.id)
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    status_msg = await message.answer("👀 Смотрю на фото...")

    try:
        # 1. Скачиваем фото в память
        photo = message.photo[-1]
        photo_file = await bot.get_file(photo.file_id)
        photo_bytes_io = BytesIO()
        await bot.download_file(photo_file.file_path, photo_bytes_io)
        photo_data = photo_bytes_io.getvalue()

        # 2. Кодируем в Base64 для Gemini
        b64_image = base64.b64encode(photo_data).decode('utf-8')

        # 3. Подготовка сообщений (Мультимодальный запрос)
        message_content = [
            {"type": "text", "text": "Опиши как решить проблему с растением если на фото картина с растением"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
        ]
        
        human_msg = HumanMessage(content=message_content)
        
        # Запрос к LLM (Vision)
        ai_response = await llm.ainvoke([human_msg])
        ai_text = ai_response.content

        # 4. Отправляем ответ пользователю
        await status_msg.delete()
        await send_long_message(message, ai_text)

        # 5. Сохраняем в память (RAG)
        user_chroma = await get_user_chroma(chat_id_str)
        ts = get_utc_now_iso()
        
        save_content = f"[Пользователь отправил фото]. Содержание фото: {ai_text}"
        await run_sync(user_chroma.add_message, role="user", content=save_content, metadata={"ts": ts})
        
        ts2 = get_utc_now_iso()
        await run_sync(user_chroma.add_message, role="assistant", content=ai_text, metadata={"ts": ts2})
        
        try:
            await run_sync(user_chroma.persist)
        except AttributeError:
            pass

    except Exception as e:
        logger.exception("Ошибка при обработке фото")
        await status_msg.edit_text(f"⚠️ Ошибка: {e}")

@dp.message(F.text)
async def handle_text(message: Message):
    chat_id_str = str(message.chat.id)
    user_text = message.text
    
    if not user_text:
        return

    try:
        user_chroma = await get_user_chroma(chat_id_str)

        # 1. Сохраняем вопрос
        ts = get_utc_now_iso()
        await run_sync(user_chroma.add_message, role="user", content=user_text, metadata={"ts": ts})

        # 2. Ищем контекст
        context_docs = await run_sync(user_chroma.get_relevant, user_text, k=4)
        
        system_msg = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system_msg]

        if context_docs:
            context_text = "\n---\n".join(context_docs)
            messages.append(SystemMessage(content=f"Контекст из памяти:\n{context_text}"))
        
        messages.append(HumanMessage(content=user_text))

        await bot.send_chat_action(chat_id=message.chat.id, action="typing")
        ai_response = await llm.ainvoke(messages)
        ai_text = ai_response.content

        await send_long_message(message, ai_text)
        
        ts2 = get_utc_now_iso()
        await run_sync(user_chroma.add_message, role="assistant", content=ai_text, metadata={"ts": ts2})
        
        try:
            await run_sync(user_chroma.persist)
        except AttributeError:
            pass

    except Exception as e:
        logger.exception("Ошибка при обработке текста")
        await message.answer("Произошла ошибка при генерации ответа.")

async def main() -> None:
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("Бот запущен (Native Gemini Vision/Audio mode)...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")