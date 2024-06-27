# I don't have any openAI keys,
# so I used the HuggingFace embedder as a placeholder
from langchain_community.embeddings import HuggingFaceEmbeddings as Embedder

from langchain.text_splitter import CharacterTextSplitter as TextSplitter

from pgvector.psycopg2 import register_vector
import psycopg2

import numpy as np


class Database(object):

    def __init__(self):

        self.embedder = Embedder()

        self.document_insert_string = "INSERT INTO documents (document_name,document_type) VALUES (%s,%s) RETURNING document_id;"
        self.embedding_insert_string = "INSERT INTO embeddings (document_id, chunk, embedding) VALUES (%s, %s, %s);"

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

    def store_text(self, document_text, document_name, document_type) -> None:

        data_to_insert = (
            document_name,
            document_type,
        )
        self.cursor.execute(
            self.document_insert_string,
            data_to_insert,
        )

        document_id = self.cursor.fetchone()[0]

        self.conn.commit()

        chunks = self.text_splitter.split_text(document_text)

        embeddings = self.embedder.embed_documents(chunks)

        for embedding, chunk in zip(embeddings, chunks):

            embedding = np.array(embedding)

            data_to_insert = (
                document_id,
                chunk,
                embedding,
            )
            self.cursor.execute(self.embedding_insert_string, data_to_insert)

        self.conn.commit()

    def get_k_relavent_embeddings(self, question, k_num=5) -> str:

        question_select_string = f"SELECT embedding FROM embeddings ORDER BY embedding <-> (%s) LIMIT {k_num}"

        embedded_question = np.array(self.embedder.embed_query(question))

        data_to_insert = (embedded_question,)

        self.cursor.execute(question_select_string, data_to_insert)

        top_k_embeddings = self.cursor.fetchall()

        return top_k_embeddings

    def get_most_relavent_embeddings(self, question):

        return self.get_k_relavent_embeddings(question, k_num=1)[0]
