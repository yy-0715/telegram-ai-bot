import logging
from telegram import Update, Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model():
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
    return pipe

pipe = load_model()

def extract_after_marker(text, marker="<|assistant|>"):
    marker_position = text.find(marker)
    if marker_position != -1:
        return text[marker_position + len(marker):]
    else:
        return ""

def generate_response(user_input):
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds briefly",
        },
        {   "role": "user", "content": user_input},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
    response = extract_after_marker(outputs[0]["generated_text"])
    return response

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello! I am your AI assistant. Ask me anything!")

def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    logger.info(f"Received message: {user_message}")
    
    try:
        response = generate_response(user_message)
        update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        update.message.reply_text("Sorry, I encountered an error while processing your request.")

if __name__ == '__main__':
    API_TOKEN = "7520639497:AAGZawUjUay9sUtAQAPKMYkFEdNyu8OZKtA"  # 替换为您的Telegram Bot API Token

    updater = Updater(token=API_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    # start
    logger.info("Starting bot...")
    updater.start_polling()
    updater.idle()
