from flask import Flask, request
from flask_restful import Resource, Api

import joblib
import json

from chatbot import chatbot

app = Flask(__name__)
api = Api(app)

class ChatBotInterface(Resource):
    def get(self):
        return "Hello from chatbot :)" 
    def post(self):
        data = request.json
        query = data['query'].strip()
        resp = chatbot(query)
        bot_response = {
          "query": query,
          "response": resp,
        }
        return bot_response


api.add_resource(ChatBotInterface, '/chat-bot-interface')


if __name__ == "__main__":
  vectorizer = joblib.load('vectorizer.pkl')  
  with open('./data.json', 'r') as file:
    data = json.load(file)

    app.run(debug=True)

