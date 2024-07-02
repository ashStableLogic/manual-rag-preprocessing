# I don't have any openAI keys,
# so I used the HuggingFace embedder as a placeholder
from langchain_community.embeddings import HuggingFaceEmbeddings as Embedder

from langchain.text_splitter import CharacterTextSplitter as TextSplitter

from pgvector.psycopg2 import register_vector
import psycopg2

import numpy as np

import re

from dotenv import load_dotenv

import cv2

from PIL import Image
import base64

import torch

from openai import OpenAI

import os

import requests

VISION_MODEL_NAME = "gpt-4-turbo"

###EXTRACTION MINIMUM PX SIZES
MIN_IMG_PX_HEIGHT = 20
MIN_IMG_PX_WIDTH = 20


class Database(object):

    def __init__(self):

        openai_api_key = os.getenv("OPENAI_API_KEY")

        self.embedder = Embedder()

        # (
        #     self.image_question_tokenizer,
        #     self.image_model,
        #     self.image_processor,
        #     self.image_question_context_len,
        # ) = load_pretrained_model(
        #     model_path=IMAGE_SUMMARY_MODEL_PATH,
        #     model_base=None,
        #     model_name=get_model_name_from_path(IMAGE_SUMMARY_MODEL_PATH),
        #     device="cpu",
        # )

        self.image_summary_header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }

        self.image_summary_payload_template = {
            "model": VISION_MODEL_NAME,
            "messages": [
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
                    "content": [
                        {
                            "type": "text",
                            "text": """Context:
                    {context}
                    ---
                    Now here is the question you need to answer.

                    Question: What does this image show?""",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 100,
        }

        self.openai_client = OpenAI()

        self.document_insert_string = "INSERT INTO documents (document_name,document_type) VALUES (%s,%s) RETURNING document_id;"
        self.chunk_insert_string = "INSERT INTO chunks (document_id, chunk, embedding) VALUES (%s, %s, %s) RETURNING chunk_id;"
        self.image_insert_string = "INSERT INTO images (chunk_id,image_name,image_filepath,image_summary,embedding) VALUES (%s,%s,%s,%s,%s) RETURNING image_id"

        self.conn = psycopg2.connect(
            host="localhost", database="at_docs", user="postgres", password="password"
        )

        self.cursor = self.conn.cursor()

        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        self.conn.commit()

        register_vector(conn_or_curs=self.cursor)

        self.text_splitter = TextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )

        self.embedder = Embedder()

    def get_b64_image(self, image_path):

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def summarise_image(self, image_path, context):

        image_summary_payload = self.image_summary_payload_template.copy()

        image_summary_payload["messages"][1]["content"][0]["text"] = (
            image_summary_payload["messages"][1]["content"][0]["text"].format(
                context=context
            )
        )

        image_summary_payload["messages"][1]["content"][1]["image_url"][
            "url"
        ] = image_summary_payload["messages"][1]["content"][1]["image_url"][
            "url"
        ].format(
            base64_image=self.get_b64_image(image_path)
        )

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.image_summary_header,
            json=self.image_summary_payload_template,
        ).json()

        answer = response["choices"][0]["message"]["content"]

        return answer

    def store_content(
        self, document_text, absolute_document_path, document_name, document_type
    ) -> None:

        folder_path = os.path.dirname(absolute_document_path)

        document_data_to_insert = (
            document_name,
            document_type,
        )

        self.cursor.execute(
            self.document_insert_string,
            document_data_to_insert,
        )

        document_id = self.cursor.fetchone()[0]

        self.conn.commit()

        chunks = self.text_splitter.split_text(document_text)

        embeddings = self.embedder.embed_documents(chunks)

        for embedding, chunk in zip(embeddings, chunks):

            embedding = np.array(embedding)

            chunk_data_to_insert = (
                document_id,
                chunk,
                embedding,
            )

            self.cursor.execute(self.chunk_insert_string, chunk_data_to_insert)

            chunk_id = self.cursor.fetchone()[0]

            image_paths_in_chunk = [
                os.path.join(folder_path, image_name)
                for image_name in re.findall(r"\[(.*\.png)\]\(\1\)", chunk)
            ]

            for image_path in image_paths_in_chunk:

                height, width = cv2.imread(image_path, 0).shape[:2]

                if height >= MIN_IMG_PX_HEIGHT and width > MIN_IMG_PX_WIDTH:

                    image_summary = self.summarise_image(image_path, chunk)

                    image_summary_embedding = self.embedder.embed_query(image_summary)

                    image_data_to_insert = (
                        chunk_id,
                        os.path.basename(image_path)[:-4],
                        image_path,
                        image_summary,
                        image_summary_embedding,
                    )

                    self.cursor.execute(self.image_insert_string, image_data_to_insert)

        self.conn.commit()

    def get_k_relavent_chunks(self, question, k_num=5):

        question_select_string = (
            f"SELECT chunk FROM chunks ORDER BY embedding <-> (%s) LIMIT {k_num}"
        )

        embedded_question = np.array(self.embedder.embed_query(question))

        data_to_insert = (embedded_question,)

        self.cursor.execute(question_select_string, data_to_insert)

        top_k_chunks = self.cursor.fetchall()

        return top_k_chunks

    def get_most_relavent_chunk(self, question) -> str:
        return self.get_k_relavent_chunks(question, k_num=1)[0]

    def get_most_relevant_image_paths_and_summaries(self, question, k_num=5):

        question_select_string = f"SELECT image_path,image_summary FROM images ORDER BY embedding <-> (%s) LIMIT {k_num}"

        embedded_question = np.array(self.embedder.embed_query(question))

        data_to_insert = (embedded_question,)

        self.cursor.execute(question_select_string, data_to_insert)

        top_k_images_and_summaries = self.cursor.fetchall()

        return top_k_images_and_summaries

    def get_most_relevant_image_paths_and_summary(self, question) -> str:
        return self.get_most_relevant_image_paths_and_summaries(question, k_num=1)[0]
