from langchain_huggingface.embeddings import HuggingFaceEmbeddings as Embedder

from langchain_text_splitters import MarkdownTextSplitter as TextSplitter

from pgvector.psycopg2 import register_vector
import psycopg2

import numpy as np

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm, trange

import ray, psutil

import re

import cv2

from PIL import Image
import base64

import torch

import os

import requests

from image_summary import ImageSummariser

###EXTRACTION MINIMUM PX SIZES
MIN_IMG_PX_HEIGHT = 300
MIN_IMG_PX_WIDTH = 300


class Database(object):

    def __init__(self) -> None:

        self.embedder = Embedder()

        self.document_insert_string = "INSERT INTO documents (document_name,document_type,product_name) VALUES (%s,%s,%s) RETURNING document_id;"
        self.chunk_insert_string = "INSERT INTO chunks (document_id, chunk, embedding) VALUES (%s, %s, %s) RETURNING chunk_id;"
        self.image_insert_string = "INSERT INTO images (chunk_id,image_name,image_filepath,image_summary,embedding) VALUES (%s,%s,%s,%s,%s) RETURNING image_id"

        self.conn = psycopg2.connect(
            host="localhost", database="at_docs", user="postgres", password="password"
        )

        self.cursor = self.conn.cursor()

        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # self.cursor.execute("CREATE INDEX IF NOT EXISTS ON chunks USING hnsw (embedding vector_l2_ops)")
        # Make an index AFTER loading in tons of data.

        ###index will use L2 distance (RMS), so use <-> operator ONLY in select statements (at least for chunks ^)

        self.conn.commit()

        register_vector(conn_or_curs=self.cursor)

        self.text_splitter = TextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        self.embedder = Embedder()

        return

    def init_image_summary_model(self):

        self.image_summariser = ImageSummariser.remote()
        ray.get(self.image_summariser.actual_init.remote())

        return

    def get_b64_image(self, image_path: str) -> str:

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def to_iterator(self, obj_ids):
        """from ray git forum"""
        while obj_ids:
            done, obj_ids = ray.wait(obj_ids)
            yield ray.get(done[0])

    def store_chunk(
        self,
        chunk,
        embedding,
        document_id,
        folder_path,
    ) -> None:

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

        self.conn.commit()

        valid_image_paths = [
            image_path
            for image_path in image_paths_in_chunk
            if (
                lambda dims: dims[0] >= MIN_IMG_PX_HEIGHT
                and dims[1] >= MIN_IMG_PX_WIDTH
            )(cv2.imread(image_path, 0).shape[:2])
        ]

        image_summaries = ray.get(
            [
                self.image_summariser.summarise_image.remote(image_path, chunk)
                for image_path in valid_image_paths
            ]
        )

        image_embeddings = map(self.embedder.embed_query, image_summaries)

        for path, summary, embedding in zip(
            valid_image_paths, image_summaries, image_embeddings
        ):
            image_data_to_insert = (
                chunk_id,
                os.path.basename(path)[:-4],
                path,
                summary,
                embedding,
            )

            self.cursor.execute(self.image_insert_string, image_data_to_insert)

        self.conn.commit()

        return

    def store_content(
        self,
        document_text: str,
        absolute_document_path: str,
        document_name: str,
        document_type: str,
        product_name: str,
    ) -> None:

        folder_path = os.path.dirname(absolute_document_path)

        document_data_to_insert = (document_name, document_type, product_name)

        self.cursor.execute(
            self.document_insert_string,
            document_data_to_insert,
        )

        document_id = self.cursor.fetchone()[0]

        self.conn.commit()

        chunks = self.text_splitter.split_text(document_text)

        embeddings = self.embedder.embed_documents(chunks)

        num_chunks = len(chunks)

        # with tqdm(total=num_chunks, desc="Storing data") as prog_bar:
        #     with ThreadPoolExecutor(max_workers=2) as exec:

        #         futures = [
        #             exec.submit(
        #                 self.store_page, chunk, embedding, document_id, folder_path
        #             )
        #             for chunk, embedding in zip(chunks, embeddings)
        #         ]

        #         for future in as_completed(futures):
        #             result = future.result()
        #             prog_bar.update(1)

        for idx in trange(num_chunks, desc="Storing chunks and images"):

            self.store_chunk(chunks[idx], embeddings[idx], document_id, folder_path)

        return

    def get_k_relavent_chunks(self, question: str, k_num: int = 5) -> list[tuple[str]]:

        question_select_string = f"SELECT chunk_id,chunk FROM chunks ORDER BY embedding <-> (%s) LIMIT {k_num}"

        embedded_question = np.array(self.embedder.embed_query(question))

        data_to_insert = (embedded_question,)

        self.cursor.execute(question_select_string, data_to_insert)

        top_k_chunks = self.cursor.fetchall()

        return top_k_chunks

    def get_most_relavent_chunk(self, question) -> tuple[str]:
        return self.get_k_relavent_chunks(question, k_num=1)[0]

    def get_most_relevant_image_paths_and_summaries(
        self, question: str, chunk_ids: list[str], k_num: int = 5
    ) -> list[tuple[str]]:

        question_select_string = f"SELECT image_id,image_filepath,image_summary FROM images WHERE chunk_id IN (%s) ORDER BY embedding <-> (%s) LIMIT {k_num}"
        # question_select_string = f"SELECT image_id,image_filepath,image_summary FROM images ORDER BY embedding <-> (%s) LIMIT {k_num}"

        embedded_question = np.array(self.embedder.embed_query(question))

        data_to_insert = (
            chunk_ids,
            embedded_question,
        )

        self.cursor.execute(question_select_string, data_to_insert)

        top_k_images_and_summaries = self.cursor.fetchall()

        return top_k_images_and_summaries

    def get_most_relevant_image_paths_and_summary(
        self, question: str, chunk_id: str
    ) -> tuple[str] | None:

        most_rel = self.get_most_relevant_image_paths_and_summaries(
            question, tuple([chunk_id]), k_num=1
        )

        if most_rel:
            return most_rel[0]

        return
