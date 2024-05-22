from abc import ABC, abstractmethod

import os

import numpy as np

import pymupdf

from argparse import ArgumentParser

###EXTRACTION MINIMUM PX SIZES
MIN_PX_HEIGHT = 20
MIN_PX_WIDTH = 20

###FIGURE EXTRACTION CONSTS
ZOOM_X, ZOOM_Y = 2.0, 2.0  # This stops image resolution from being quartered
ZOOM_MAT = pymupdf.Matrix(ZOOM_X, ZOOM_Y)

FIGURE_X_GROUPING_ADJACENCY_PX = 3
FIGURE_Y_GROUPING_ADJACENCY_PX = 3


def get_class_leaves(cls: object) -> list[object]:
    """Gets all subclasses with no subclasses of their own

    Args:
        cls (object): Any class

    Returns:
        list[object]: list of classes
    """
    subclasses = []

    for subclass in cls.__subclasses__():
        if subclass.__subclasses__():
            return subclasses.extend(subclass.__subclasses__())
        else:
            return subclass


class Extractor(ABC):

    def extract(self, document: pymupdf.Document) -> pymupdf.Document:
        """Iterates through document pages, extracting object type from them

        Args:
            document (pymupdf.Document): Given PDF

        Returns:
            pymupdf.Document: Document with object types from page extracted, redacted and those redactions applied
        """
        for page_index, page in enumerate(document):
            self.save_and_redact(page, page_index)

        return document


class Redactor(Extractor):

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
        self.relative_output_folder_path = os.join(
            relative_output_folder_path, ImageRedactor.folder_name
        )

    def save_and_redact(self, page, page_index):
        page_name = f"page {page_index+1}"

        image_blocks = [
            block
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

            page.add_redact_annot(
                image_bbox, text=relative_image_file_path, cross_out=False
            )

        page.apply_redactions(1, 2, 1)

        return page


class FigureRedactor(Redactor):

    folder_name = "figures"

    def __init__(self, relative_output_folder_path):
        self.relative_output_folder_path = os.join(
            relative_output_folder_path, FigureRedactor.folder_name
        )

    def save_and_redact(self, page, page_index):
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

            page.add_redact_annot(
                figure, text=relative_figure_file_path, cross_out=False
            )

        page.apply_redactions(1, 2, 1)

        return page


class TextExtractor(Extractor):

    def __init__(self):

        pass


class PdfChunker(object):

    def __init__(self, args):
        self.manuals_folder = args.manuals_folder
        self.redo = args.redo_processed_manuals
        self.output_folder_prefix = args.output_folder

        self.extractor_types = get_class_leaves(Extractor)

    def set_file_paths(
        self, relative_document_path: str, document_filename: str
    ) -> None:
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

            document = pymupdf.open(self.absolute_document_path)

            extractors = [
                cls.__init__(self.relative_output_folder_path)
                for cls in self.extractor_types
            ]

            for extractor in extractors:
                extractor.extract(document)

            document.save(self.absolute_finished_document_path)
        else:
            print(f"========{document_filename} already processed========")

        return

    def run(self):

        manual_filenames = [
            file
            for file in os.listdir(self.manuals_folder)
            if file.split(".")[-1] == "pdf"  # Get PDF extension files only
        ]

        assert manual_filenames, "no manuals to process"

        for manual_filename in manual_filenames:

            relative_manual_path = os.path.join(self.manuals_folder, manual_filename)
            absolute_manual_path = os.path.abspath(relative_manual_path)


def main(args):

    pdf_chunker = PdfChunker(args)

    pdf_chunker.run()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-m", "--manuals-folder", dest="manuals_folder", required=True)

    parser.add_argument(
        "-r",
        "--redo-processed-manuals",
        dest="redo_processed_manuals",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-o", "--output-folder", dest="output_folder", type=str, default="output"
    )

    parser.add_argument(
        "-t", "--temp-folder", dest="temp_folder", type=str, default="temp"
    )

    args = parser.parse_args()

    main(args)
