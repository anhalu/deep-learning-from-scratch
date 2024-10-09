from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "anhalu"
openai_api_base = "http://localhost:11434/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
)
print("Chat response:", chat_response)