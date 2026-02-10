from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

load_dotenv()

llm = ChatOllama(
    model="mistral",
    temperature=0.7,
)

