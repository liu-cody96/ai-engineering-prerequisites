from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config


def run_llm(provider, model_name, messages, max_tokens=500):
    if provider == "OpenAI":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "Groq":
        client = Groq(api_key=config.GROQ_API_KEY)
    else:
        client = genai.Client(api_key=config.GOOGLE_API_KEY)

    if provider == "Google":
        # Gemini takes plain text, not role/content dicts
        return client.models.generate_content(
            model=model_name,
            contents=[m["content"] for m in messages],
        ).text
    elif provider == "Groq":
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
        ).choices[0].message.content
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning_effort="minimal",  # keeps OpenAI costs down
        ).choices[0].message.content
