from flask import Flask, render_template, request, jsonify
from chatbot import Chatbot

app = Flask(__name__)
bot = Chatbot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message", "")
    mode = data.get("mode", "retrieval")
    response = bot.get_response(msg, mode=mode)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)