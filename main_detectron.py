import os

import numpy as np
from PIL import Image

import pymupdf

import layoutparser as lp

from argparse import ArgumentParser

###EXTRACTION MINIMUM PX SIZES
MIN_PX_HEIGHT = 30
MIN_PX_WIDTH = 30

###FIGURE EXTRACTION CONSTS
ZOOM_X, ZOOM_Y = 2.0, 2.0  # This stops image resolution from being quartered
ZOOM_MAT = pymupdf.Matrix(ZOOM_X, ZOOM_Y)

FIGURE_X_GROUPING_ADJACENCY_PX = 3
FIGURE_Y_GROUPING_ADJACENCY_PX = 3

###DETECTRON2 CONSTS
DETECTRON_CONFIG = "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
DETECTRON_EXTRA_CONFIG = [
    "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
    0.7,
    "MODEL.DEVICE",
    "cpu",
]
DETECTRON_CONFIG_LABELS = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure",
}


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


def process_pdf_figures(page, page_index, relative_figure_folder_path, layout_model):
    """Uses detectron-powered layout finding CV model to extract drawings

    Args:
        page (pymupdf.Page): Given page
        page_index (int): Index of given page
        relative_image_folder_path (str): Relative path of drawing output folder

    Returns:
        Page: Given page with redactions made, but not applied
    """

    page_name = f"page {page_index+1}"

    drawings = page.get_drawings()

    if drawings:
        page_pixmap = page.get_pixmap(matrix=ZOOM_MAT)

        page_array = pixmap_2_numpy_array(page_pixmap)

        layout_results = layout_model.detect(page_array)

        figure_blocks = [block for block in layout_results if block.type == "Figure"]

        if figure_blocks:
            print(f"{len(figure_blocks)} figure(s) on page {page_index+1}")

            relative_page_path = os.path.join(relative_figure_folder_path, page_name)
            absolute_page_path = os.path.abspath(relative_page_path)

            os.makedirs(absolute_page_path, exist_ok=True)
        else:
            print(f"No figures in page {page_index+1}")
            return page
    else:
        print(f"No figures in page {page_index+1}")
        return page

    for figure_block_index, figure_block in enumerate(figure_blocks):
        figure_name = f"image {figure_block_index+1}.png"

        relative_figure_file_path = os.path.join(
            relative_figure_folder_path, page_name, figure_name
        )
        absolute_figure_file_path = os.path.abspath(relative_figure_file_path)

        # page_array_to_save = Image.fromarray(page_array.astype("uint8"), "RGB")
        # page_array_to_save.save(
        #     "/home/ash/projects/manual-rag-preprocessing/test_output/CDJ-3000_manual_EN_ONE_PAGE/page_array.png"
        # )

        cropped_figure = figure_block.crop_image(page_array)

        cropped_figure_image = Image.fromarray(cropped_figure.astype("uint8"), "RGB")

        cropped_figure_image.save(absolute_figure_file_path)

        ###Going from layout parser to pymupdf positional naming scheme
        figure_block_position = figure_block.block
        x_0 = (figure_block_position.x_1 / ZOOM_X) + 1
        y_0 = (figure_block_position.y_1 / ZOOM_Y) + 1
        x_1 = (figure_block_position.x_2 / ZOOM_X) + 1
        y_1 = (figure_block_position.y_2 / ZOOM_Y) + 1

        drawing_rect = pymupdf.Rect(x_0, y_0, x_1, y_1)
        # drawing_rect = figure.to_rectangle()

        page.add_redact_annot(
            drawing_rect, text=relative_figure_file_path, cross_out=False
        )

    return page


def process_pdf_pages(
    manual_name,
    manual_path,
    output_file_path,
    relative_output_folder_path,
):

    layout_model = lp.Detectron2LayoutModel(
        DETECTRON_CONFIG,
        extra_config=DETECTRON_EXTRA_CONFIG,
        label_map=DETECTRON_CONFIG_LABELS,
    )

    relative_image_folder_path = os.path.join(relative_output_folder_path, "images")

    relative_figure_folder_path = os.path.join(relative_output_folder_path, "figures")

    write_doc = pymupdf.open(manual_path)

    for page_index, page in enumerate(write_doc):

        page.clean_contents()

        page = process_pdf_images(page, page_index, relative_image_folder_path)

        page.apply_redactions(1, 2, 1)

        # process_pdf_figures uses computer vision to separate actual figures
        # from superfluous drawings and find their positional boundaries
        # VERY SLOW ON COMPANY LAPTOPS

        # (if you have a cuda-enabled gpu handy and have built detectron2
        # with it in mind, remove the MODEL.DEVICE and cpu strings from
        # the extra detectron config const)

        page = process_pdf_figures(
            page, page_index, relative_figure_folder_path, layout_model
        )
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
