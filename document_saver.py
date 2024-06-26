from abc import ABC, abstractmethod

import os
from pathlib import Path

import numpy as np

import pymupdf
import pymupdf4llm

import argparse

from database import Database

###EXTRACTION MINIMUM PX SIZES
MIN_IMG_PX_HEIGHT = 20
MIN_IMG_PX_WIDTH = 20

###FIGURE EXTRACTION CONSTS
ZOOM_X, ZOOM_Y = 2.0, 2.0  # This stops image resolution from being quartered
ZOOM_MAT = pymupdf.Matrix(ZOOM_X, ZOOM_Y)

FIGURE_X_GROUPING_ADJACENCY_PX = 3
FIGURE_Y_GROUPING_ADJACENCY_PX = 3

PER_PAGE_GRAPHICS_LIMIT = 10000


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

    def __init__(self, apply_redactions):

        self.apply_redactions = apply_redactions

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

    def __init__(self, relative_output_folder_path, apply_redactions=False):
        super().__init__(apply_redactions)

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
            if image_block["height"] > MIN_IMG_PX_HEIGHT
            and image_block["width"] > MIN_IMG_PX_WIDTH
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

        if self.apply_redactions:
            page.apply_redactions(1, 2, 1)

        return page


class FigureRedactor(Redactor):

    folder_name = "figures"

    def __init__(self, relative_output_folder_path, apply_redactions=False):
        super().__init__(apply_redactions)

        self.relative_output_folder_path = os.path.join(
            relative_output_folder_path, FigureRedactor.folder_name
        )

    def save_and_redact(self, page, page_index, use_absolute_file_path=True):
        page_name = f"page {page_index+1}"

        figures = [
            figure
            for figure in page.cluster_drawings()
            if figure.height > MIN_IMG_PX_HEIGHT and figure.width > MIN_IMG_PX_WIDTH
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

        if self.apply_redactions:

            page.apply_redactions(1, 2, 1)

        return page


class PdfEmbedder(object):

    def __init__(self, args):
        self.document_type = "PDF"

        self.documents_folder = args.documents_folder
        self.redo = args.redo_processed_manuals

        self.text_fetcher = TextFetcher()

        self.db = Database()

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

    def process_document(self, absolute_document_path: str) -> None:

        document_filename = os.path.basename(absolute_document_path)[:-4]

        markdown_document_text = pymupdf4llm.to_markdown(
            absolute_document_path,
            write_images=True,
            dpi=300,
            margins=0,
            # graphics_limit=PER_PAGE_GRAPHICS_LIMIT,
        )

        self.db.store_text(
            markdown_document_text, document_filename, self.document_type
        )

        return

    def run(self):

        for file_path in Path(self.documents_folder).rglob("*.pdf"):

            self.process_document(file_path)


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

    parser.add_argument(
        "-d", "--documents-folder", dest="documents_folder", required=True
    )

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
