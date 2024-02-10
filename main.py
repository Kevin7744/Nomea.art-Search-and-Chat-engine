from os import environ
from dotenv import load_dotenv
from flask import Flask, jsonify, request
# from flask_restful import reqparse, abort, Api, Resource
from supabase.client import Client, create_client

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector


load_dotenv()  



app = Flask(__name__)



TABLE = "documents"

load_dotenv()

url: str = environ.get("SUPABASE_URL")
key: str = environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)



if __name__ == '__main__':
    app.run(debug=True, port=environ.get("PORT", 5000), host='0.0.0.0')