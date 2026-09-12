from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


grader_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


generator_model = ChatOpenAI(
    model="gpt-5-mini",
)
