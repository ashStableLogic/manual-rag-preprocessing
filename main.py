from abc import ABC, abstractmethod

import os

import numpy as np

import pymupdf
import pymupdf4llm

# I don't have any openAI keys,
# so I used the HuggingFace embedder as a placeholder
# from langchain.text_splitter import CharacterTextSplitter as TextSplitter

from langchain.text_splitter import CharacterTextSplitter as TextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings as Embedder

import psycopg2

import argparse

###EXTRACTION MINIMUM PX SIZES
MIN_PX_HEIGHT = 20
MIN_PX_WIDTH = 20

###FIGURE EXTRACTION CONSTS
ZOOM_X, ZOOM_Y = 2.0, 2.0  # This stops image resolution from being quartered
ZOOM_MAT = pymupdf.Matrix(ZOOM_X, ZOOM_Y)

FIGURE_X_GROUPING_ADJACENCY_PX = 3
FIGURE_Y_GROUPING_ADJACENCY_PX = 3


class Extractor(ABC):

    @abstractmethod
    def run(self, document: pymupdf.Document):
        raise NotImplementedError("Implement run method")


class Fetcher(Extractor):

    def run(self, document: pymupdf.Document) -> list:
        """Returns a list of fetched things for each page

        Args:
            document (pymupdf.Document): Given document to process

        Returns:
            list: List of things where each element is a collection of that thing for each page
        """
        rtn = []

        for page in document:
            rtn.append(self.fetch(page))

        return rtn

    @abstractmethod
    def fetch(self, page: pymupdf.Page):
        raise NotImplementedError("Implement this class' fetch method")


class TextFetcher(Fetcher):

    def __init__(self):

        pass

    def fetch(self, page: pymupdf.Page):

        page_text = page.get_text(sort=True)

        return page_text


class Redactor(Extractor):

    def run(self, document: pymupdf.Document) -> pymupdf.Document:
        """Iterates through document pages, extracting object type from them

        Args:
            document (pymupdf.Document): Given PDF

        Returns:
            pymupdf.Document: Document with object types from page extracted, redacted and those redactions applied
        """
        for page_index, page in enumerate(document):
            self.save_and_redact(page, page_index)

        return document

    @abstractmethod
    def save_and_redact(self, page: pymupdf.Page, save_path: str) -> pymupdf.Page:
        """Extracts object type from page, saving object externally
        and return page after applying redactions

        Args:
            page (pymupdf.Page): Given page object

        Raises:
            NotImplementedError: If method not implemented

        Returns:
            pymupdf.Page: Page object with redactions applied
        """

        raise NotImplementedError("Implement object-from-page extraction method")


class ImageRedactor(Redactor):

    folder_name = "images"

    def __init__(self, relative_output_folder_path):
        self.relative_output_folder_path = os.path.join(
            relative_output_folder_path, ImageRedactor.folder_name
        )

    def save_and_redact(self, page, page_index, use_absolute_file_path=True):
        page_name = f"page {page_index+1}"

        image_blocks = [  # I'm using image blocks instead of pymupdf.Page.get_images()
            block  # because it can extract inline images too
            for block in page.get_text("dict", sort=True)["blocks"]
            if block["type"] == 1
        ]

        ##Second pass to get rid of images that are too small

        image_blocks = [
            image_block
            for image_block in image_blocks
            if image_block["height"] > MIN_PX_HEIGHT
            and image_block["width"] > MIN_PX_WIDTH
        ]

        if image_blocks:
            print(f"{len(image_blocks)} image(s) found on page {page_index+1}")

            relative_page_path = os.path.join(
                self.relative_output_folder_path, page_name
            )
            absolute_page_path = os.path.abspath(relative_page_path)

            os.makedirs(absolute_page_path, exist_ok=True)
        else:
            print(f"No images on page {page_index+1}")
            return page

        for image_index, image_block in enumerate(image_blocks):

            image_name = f"image {image_index+1}.png"

            relative_image_file_path = os.path.join(
                self.relative_output_folder_path, page_name, image_name
            )
            absolute_image_file_path = os.path.abspath(relative_image_file_path)

            image_bbox = image_block["bbox"]
            image_data = image_block["image"]

            with open(absolute_image_file_path, "wb") as file:
                file.write(image_data)

            annotation_text = (
                absolute_image_file_path
                if use_absolute_file_path
                else relative_image_file_path
            )

            page.add_redact_annot(image_bbox, text=annotation_text, cross_out=False)

        page.apply_redactions(1, 2, 1)

        return page


class FigureRedactor(Redactor):

    folder_name = "figures"

    def __init__(self, relative_output_folder_path):
        self.relative_output_folder_path = os.path.join(
            relative_output_folder_path, FigureRedactor.folder_name
        )

    def save_and_redact(self, page, page_index, use_absolute_file_path=True):
        page_name = f"page {page_index+1}"

        figures = [
            figure
            for figure in page.cluster_drawings()
            if figure.height > MIN_PX_HEIGHT and figure.width > MIN_PX_WIDTH
        ]

        if figures:
            print(f"{len(figures)} figures(s) found on page {page_index+1}")

            relative_page_path = os.path.join(
                self.relative_output_folder_path, page_name
            )
            absolute_page_path = os.path.abspath(relative_page_path)

            os.makedirs(absolute_page_path, exist_ok=True)
        else:
            print(f"No figures on page {page_index+1}")
            return page

        for figure_index, figure in enumerate(figures):

            figure_name = f"image {figure_index+1}.png"

            relative_figure_file_path = os.path.join(
                self.relative_output_folder_path, page_name, figure_name
            )
            absolute_figure_file_path = os.path.abspath(relative_figure_file_path)

            figure_pixmap = page.get_pixmap(matrix=ZOOM_MAT, clip=figure)
            # figure_pixmap = page.get_pixmap(clip=figure)

            figure_pixmap.pil_save(absolute_figure_file_path)

            annotation_text = (
                absolute_figure_file_path
                if use_absolute_file_path
                else relative_figure_file_path
            )

            page.add_redact_annot(figure, text=annotation_text, cross_out=False)

        page.apply_redactions(1, 2, 1)

        return page


class DB(object):

    def __init__(self):
        self.embedder = Embedder()

        # self.db_connection = psycopg2.connect(
        #     host="localhost", database="db", user="user", password="password"
        # )
        # self.cursor = self.db_connection.cursor()

        self.text_splitter = TextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )

    def store_text(self, sub_documents, document_name) -> None:

        # insert_query = "INSERT INTO service_manuals_file (file_name) VALUES (%s);"
        # data_to_insert = (document_name,)
        # self.cursor.execute(insert_query, data_to_insert)
        # self.db_connection.commit()

        chunks = self.text_splitter.split_text(sub_documents)

        for chunk in chunks:
            # embedding = self.embedder.embed_documents(chunk)
            # insert_query = "INSERT INTO service_manuals_file_data (service_manuals_file_id, text, embedding) VALUES (%s, %s, %s);"
            # data_to_insert = (document_name, chunk, embedding)
            # self.cursor.execute(insert_query, data_to_insert)

            pass


class PdfEmbedder(object):

    def __init__(self, args):
        self.documents = args.documents
        self.redo = args.redo_processed_manuals
        self.output_folder_prefix = args.output_folder

        self.text_fetcher = TextFetcher()

        self.db = DB()

    def set_file_paths(
        self, relative_document_path: str, document_filename: str
    ) -> None:
        """Inits filepaths to where the input pdf is
        and where the finished product and the extracted
        content is saved

        Args:
            relative_document_path (str): _description_
            document_filename (str): _description_
        """

        document_name = document_filename.split(".")[0]

        self.absolute_document_path = os.path.abspath(relative_document_path)

        relative_finished_document_path = os.path.join(
            self.output_folder_prefix, document_name, document_filename
        )

        self.absolute_finished_document_path = os.path.abspath(
            relative_finished_document_path
        )

        self.relative_output_folder_path = os.path.join(
            self.output_folder_prefix, document_name
        )

        self.absolute_output_folder_path = os.path.abspath(
            self.relative_output_folder_path
        )

    def process_document(
        self, relative_document_path: str, document_filename: str
    ) -> None:

        self.set_file_paths(relative_document_path, document_filename)

        if not os.path.exists(self.absolute_output_folder_path) or self.redo:
            print(f"========processing {document_filename}========")

            os.makedirs(self.absolute_output_folder_path, exist_ok=True)

            document = pymupdf.open(
                os.path.join(self.absolute_document_path, document_filename)
            )
            redactors = [
                ImageRedactor(self.relative_output_folder_path),
                FigureRedactor(self.relative_output_folder_path),
            ]

            for redactor in redactors:
                redactor.run(document)

            document.save(self.absolute_finished_document_path)

        else:
            print(f"========{document_filename} already processed========")

            document = pymupdf.open(self.absolute_finished_document_path)

        # markdown_document = pymupdf4llm.to_markdown(document)
        # markdown_document = pymupdf4llm.to_markdown(
        #     os.path.join(self.absolute_document_path, document_filename)
        # )

        # self.db.store_markdown_document(markdown_document, document_filename)

        document_text = " ".join(self.text_fetcher.run(document))

        document_name = document_filename.split(".")[0]

        self.db.store_text(document_text, document_name)

        return

    def run(self):
        if os.path.isfile(os.path.abspath(self.documents)):
            assert self.documents[-4:] == ".pdf"

            self.process_document(
                os.path.dirname(self.documents), os.path.basename(self.documents)
            )
        else:
            for document_filename in os.listdir(self.documents):
                if document_filename[-4:] == ".pdf":
                    self.process_document(self.documents, document_filename)
                else:
                    print(f"Skipping {document_filename}, not a PDF")


def main(args):

    pdf_embedder = PdfEmbedder(args)

    pdf_embedder.run()


if __name__ == "__main__":
    """USAGE IS
    python main.py
    -d "Path to single document or folder"
    -r "add switch if you want to redo already processed documents"
    -o "output folder path"
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--documents(s)", dest="documents", required=True)

    parser.add_argument(
        "-r",
        "--redo-processed-manuals",
        dest="redo_processed_manuals",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "-o", "--output-folder", dest="output_folder", type=str, default="output"
    )

    args = parser.parse_args()

    main(args)
