from database import Database

from transformers import pipeline
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
)

import os
from dotenv import load_dotenv

from openai import OpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings as Embedder


class Chatbot(object):

    def __init__(self) -> None:

        self.db = Database()
        self.embedder = Embedder()

        self.prompt = [
            {
                "role": "system",
                "content": """Using the information contained in the context based around text from an instruction manual and optionally a summary of the most relavent image,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant. Reference the image when possible.
        If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}""",
            },
        ]

        self.openai_client = OpenAI()

    def get_context(self, question: str) -> str:

        most_relevant_chunk = self.db.get_most_relavent_chunk(question)

        most_relavent_image_path, most_relavent_summary = (
            self.db.get_most_relevant_image_paths_and_summary(question)
        )

        return most_relevant_chunk, most_relavent_summary, most_relavent_image_path

    def ask(self, question):

        chunk, image_summary, image_path = self.get_context(question)

        if image_summary != None:

            context = chunk + "\n" + image_summary
        else:
            context = chunk

        print(image_path)

        prompt = self.prompt.copy()

        prompt[1]["content"] = prompt[1]["content"].format(
            question=question, context=context
        )

        response = self.openai_client.chat.completions.create(
            messages=prompt, model="gpt-3.5-turbo"
        )

        answer = response.choices[0].message.content

        return answer


def main():
    load_dotenv()

    chatbot = Chatbot()

    while True:
        question = input("\nAsk a question about CDJ-3000 or R4731 1:\t")

        answer = chatbot.ask(question)

        print(answer)


if __name__ == "__main__":

    main()
