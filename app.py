from flask import Flask, render_template, request
from markupsafe import Markup
import markdown
import google.generativeai as genai
from rag.context_retrieval import retrieve_context

app = Flask(__name__)
chat_history = []

genai.configure(api_key="AIzaSyCToFmHzEJJeYSvVZsdwjNhzp6SwU9CAdo")
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form["user_input"]

        context = retrieve_context(user_input)
        prompt = f"""Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"""

        response = model.generate_content(prompt)
        bot_reply = response.text
        html_reply = markdown.markdown(bot_reply)

        chat_history.append({"role": "user", "text": user_input})
        chat_history.append({"role": "bot", "text": Markup(html_reply)})

    return render_template("chat.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)