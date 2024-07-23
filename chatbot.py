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

from langchain_huggingface.embeddings import HuggingFaceEmbeddings as Embedder

import re


class Chatbot(object):

    def __init__(self) -> None:
        load_dotenv()

        self.db = Database()
        self.embedder = Embedder()

        self.prompt = [
            {
                "role": "system",
                "content": """Using the information contained in the context based around text from an instruction manual and optionally a summary of the most relavent image,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Provide the number of the source document when relevant. Reference the image when available.
        If the answer cannot be deduced from the context, do not give an answer.
        Do not include a source for the context in your answer""",
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

    def get_context(self, question: str) -> tuple[str | None]:

        most_relevant_chunk_id, most_relevant_chunk = self.db.get_most_relavent_chunk(
            question
        )

        most_relevant_image_id = None
        most_relavent_image_path = None
        most_relavent_summary = None

        rtn = self.db.get_most_relevant_image_paths_and_summary(
            question, most_relevant_chunk_id
        )

        if rtn != None:
            most_relevant_image_id, most_relavent_image_path, most_relavent_summary = (
                rtn
            )

        return (
            most_relevant_chunk,
            most_relevant_chunk_id,
            most_relavent_summary,
            most_relavent_image_path,
            most_relevant_image_id,
        )

    def ask(self, question: str) -> str:

        chunk, chunk_id, image_summary, image_path, image_id = self.get_context(
            question
        )

        if image_summary != None:
            context = chunk + "\nImage Summary:\n" + image_summary
        else:
            context = chunk

        prompt = self.prompt.copy()

        prompt[1]["content"] = prompt[1]["content"].format(
            question=question, context=context
        )

        response = self.openai_client.chat.completions.create(
            messages=prompt, model="gpt-3.5-turbo"
        )

        answer = response.choices[0].message.content

        return answer, image_path, chunk_id, image_id


def main():
    chatbot = Chatbot()

    while True:
        question = input("\nAsk a question about a product:\n\n")

        answer, image_path = chatbot.ask(question)
        print()

        if image_path != None:
            print(f"IMAGE PATH: {image_path}\n")

        print(answer)


if __name__ == "__main__":

    main()
