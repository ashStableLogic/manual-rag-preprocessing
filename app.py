from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    session,
    request,
    send_from_directory,
)
from chatbot import Chatbot
import os

app = Flask(__name__)
chatbot = Chatbot()


@app.route("/get_document_image", methods=["GET"])
def get_image():

    filename = request.args.get("filename").partition("/documents")[-1]

    print(send_from_directory("/documents", path=filename))

    return send_from_directory("/documents", path=filename)


@app.route("/ask", methods=["GET", "POST"])
def ask():
    session.clear()

    image_returned = False

    if request.method == "POST":
        question = request.form["question"]

        print(f"QUESTION IS: {question}")

        answer, image_path, chunk_id, image_id = chatbot.ask(question)

        if image_path != None:
            image_path = image_path.partition("/documents/")[-1]
            image_returned = True

        print(f"CHUNK ID: {chunk_id}")
        print(f"IMAGE ID: {image_id}")

        return render_template(
            "ask.html",
            answer=answer,
            image_returned=image_returned,
            answer_image=image_path,
        )

    else:
        return render_template("home.html")


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


if __name__ == "__main__":

    app.debug = True
    app.run()
