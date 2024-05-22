import os

import numpy as np

import pymupdf

from argparse import ArgumentParser

###EXTRACTION MINIMUM PX SIZES
MIN_PX_HEIGHT = 30
MIN_PX_WIDTH = 30

###FIGURE EXTRACTION CONSTS
ZOOM_X, ZOOM_Y = 2.0, 2.0  # This stops image resolution from being quartered
ZOOM_MAT = pymupdf.Matrix(ZOOM_X, ZOOM_Y)

FIGURE_X_GROUPING_ADJACENCY_PX = 3
FIGURE_Y_GROUPING_ADJACENCY_PX = 3


def pixmap_2_numpy_array(pixmap):
    pixmap_height = pixmap.height
    pixmap_width = pixmap.width
    pixmap_channels = pixmap.n

    np_image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap_height, pixmap_width, pixmap_channels
    )

    np.ascontiguousarray(np_image[..., [2, 1, 0]])

    return np_image


def process_pdf_images(page, page_index, relative_image_folder_path):
    """Extracts images on a given page and redacts them with the path
    they are saved to

    Args:
        page (pymupdf.Page): Given page
        page_index (int): Index of given page
        relative_image_folder_path (str): Relative path of extracted image save location

    Returns:
        Page: Given page with redactions made, but not applied
    """

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
        if image_block["height"] > MIN_PX_HEIGHT and image_block["width"] > MIN_PX_WIDTH
    ]

    if image_blocks:
        print(f"{len(image_blocks)} image(s) found on page {page_index+1}")

        relative_page_path = os.path.join(relative_image_folder_path, page_name)
        absolute_page_path = os.path.abspath(relative_page_path)

        os.makedirs(absolute_page_path, exist_ok=True)
    else:
        print(f"No images on page {page_index+1}")
        return page

    for image_index, image_block in enumerate(image_blocks):

        image_name = f"image {image_index+1}.png"

        relative_image_file_path = os.path.join(
            relative_image_folder_path, page_name, image_name
        )
        absolute_image_file_path = os.path.abspath(relative_image_file_path)

        image_bbox = image_block["bbox"]
        image_data = image_block["image"]

        with open(absolute_image_file_path, "wb") as file:
            file.write(image_data)

        page.add_redact_annot(
            image_bbox, text=relative_image_file_path, cross_out=False
        )

    return page


def process_pdf_figures(page, page_index, relative_figure_folder_path):

    page_name = f"page {page_index+1}"

    figures = [
        figure
        for figure in page.cluster_drawings()
        if figure.height > MIN_PX_HEIGHT and figure.width > MIN_PX_WIDTH
    ]

    if figures:
        print(f"{len(figures)} figures(s) found on page {page_index+1}")

        relative_page_path = os.path.join(relative_figure_folder_path, page_name)
        absolute_page_path = os.path.abspath(relative_page_path)

        os.makedirs(absolute_page_path, exist_ok=True)
    else:
        print(f"No figures on page {page_index+1}")
        return page

    for figure_index, figure in enumerate(figures):

        figure_name = f"image {figure_index+1}.png"

        relative_figure_file_path = os.path.join(
            relative_figure_folder_path, page_name, figure_name
        )
        absolute_figure_file_path = os.path.abspath(relative_figure_file_path)

        figure_pixmap = page.get_pixmap(matrix=ZOOM_MAT, clip=figure)
        # figure_pixmap = page.get_pixmap(clip=figure)

        figure_pixmap.pil_save(absolute_figure_file_path)

        page.add_redact_annot(figure, text=relative_figure_file_path, cross_out=False)

    return page


def process_pdf_pages(
    manual_name,
    manual_path,
    output_file_path,
    relative_output_folder_path,
):

    relative_image_folder_path = os.path.join(relative_output_folder_path, "images")

    relative_figure_folder_path = os.path.join(relative_output_folder_path, "figures")

    write_doc = pymupdf.open(manual_path)

    for page_index, page in enumerate(write_doc):

        page.clean_contents()

        page = process_pdf_images(page, page_index, relative_image_folder_path)

        page.apply_redactions(1, 2, 1)

        page = process_pdf_figures(page, page_index, relative_figure_folder_path)

        print()

        page.apply_redactions(1, 2, 1)

    write_doc.save(output_file_path)

    return


def main(args):
    manuals_folder = args.manuals_folder
    redo = args.redo_processed_manuals
    output_folder_prefix = args.output_folder

    manual_filenames = [
        file
        for file in os.listdir(manuals_folder)
        if file.split(".")[-1] == "pdf"  # Get PDF extension files only
    ]

    assert manual_filenames, "no manuals to process"

    for manual_filename in manual_filenames:
        manual_name = manual_filename.split(".")[0]

        relative_manual_path = os.path.join(manuals_folder, manual_filename)
        absolute_manual_path = os.path.abspath(relative_manual_path)

        relative_output_folder_path = os.path.join(output_folder_prefix, manual_name)
        relative_output_file_path = os.path.join(
            output_folder_prefix, manual_name, manual_filename
        )

        absolute_output_folder_path = os.path.abspath(relative_output_folder_path)

        if not os.path.exists(absolute_output_folder_path) or redo:
            print(f"========processing {manual_name}========")

            os.makedirs(absolute_output_folder_path, exist_ok=True)

            process_pdf_pages(
                manual_name,
                absolute_manual_path,
                relative_output_file_path,
                relative_output_folder_path,
            )

    return


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
