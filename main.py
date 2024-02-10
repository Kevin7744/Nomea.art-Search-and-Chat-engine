from flask import Flask, request, jsonify
import os
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores.supabase import SupabaseVectorStore
from dotenv import load_dotenv
from supabase import create_client

# app = Flask(__name__)

load_dotenv()

# Initialize environment variables
supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
if not supabase_key:
    raise ValueError("Expected SUPABASE_SERVICE_KEY environment variable")

url = os.getenv('SUPABASE_URL')
if not url:
    raise ValueError("Expected SUPABASE_URL environment variable")

async def run():
    # Create a Supabase client
    client = create_client(url, supabase_key)

    # Initialize Supabase vector store
    vector_store = await SupabaseVectorStore.from_texts(
        ['Hello world', 'Bye bye', "What's this?"],
        [{'id': 2}, {'id': 1}, {'id': 3}],
        OpenAIEmbeddings(),
        client=client,
        table_name='documents',
        query_name='match_documents'
    )

    # Perform similarity search and return the result
    result_one = await vector_store.similarity_search('Hello world', 1)
    print(result_one)

# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
# system_message = SystemMessage(
#     content="""
#     "You are lila, an AI-art advisory"
#     "You help users with questions regarding art and offer advises to them also"
#     "You have a search tool that you can use to search a superbase vector database and retrieve the necessary art work related to the user's query."
#     "   - For example: I want to some art related to da vinci style"
#     "   - Use the search tool and search Da-vinci style art-work"
#     "Your responses should be as short as possible"
#     """)

# tools = []

# agent_kwargs = {
#     "extra_prompt_message": [MessagesPlaceholder(variable_name="memory")],
#     "system_message": system_message,
# }

# memory = ConversationSummaryBufferMemory(memory_key="memory",
#                                          return_messages=True,
#                                          llm=llm,
#                                          max_token_limit=250)

# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     verbose=True,
#     agent_kwargs=agent_kwargs,
#     memory=memory,
#     user_input_key="input"
# )

# # Define the route for the API endpoint
# @app.route("/chat", methods=["POST"])
# def chat():
#     try:
#         user_input = request.json["input"]
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400  
#     if user_input.lower() == "end":
#         return jsonify({
#             "message": "Have a good day!"
#         })

#     agent_response = agent({"input": user_input})
#     print("Agent Response:", agent_response)  

#     return jsonify({"response": agent_response})

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
    # app.run(host="0.0.0.0", port=5000, debug=True)
