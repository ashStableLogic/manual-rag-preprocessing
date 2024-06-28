from database import Database

from transformers import pipeline
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
)

import os
import dotenv

from openai import OpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings as Embedder


class Chatbot(object):

    def __init__(self) -> None:

        self.db = Database()
        self.embedder = Embedder()

        self.prompt = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant.
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

        most_relavent_context = self.db.get_most_relavent_chunk(question)

        return most_relavent_context

    def ask(self, question):

        context = self.get_context(question)

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
    dotenv.load_dotenv()

    chatbot = Chatbot()

    while True:
        question = input("\nAsk a question about CDJ-3000 or R4731 1:\t")

        answer = chatbot.ask(question)

        print(answer)


if __name__ == "__main__":

    main()
