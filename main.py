import os

import pymupdf
from argparse import ArgumentParser

SEP = os.pathsep


def process_pdf_images(manual_path, output_folder, output_image_folder):

    doc = pymupdf.open(manual_path)

    for page_index, page in enumerate(doc):

        image_refs = page.get_images()

        if image_refs:
            print(f"{len(image_refs)} found on page {page_index+1}")
        else:
            print(f"No images on page {page_index+1}")

        for image_index, image_ref in enumerate(image_refs):

            image_name = f"page_{page_index+1}_image_{image_index+1}"

            image_path = output_image_folder + SEP + image_name

            xref = image_ref[0]
            pix = pymupdf.Pixmap(doc, xref)
            bbox = pymupdf.Page.get_image_bbox(page, xref)

            if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            pix.save(image_path)


def main(args):
    manuals_folder = os.path.abspath(args.manuals_folder)
    redo = args.redo_processed_manuals

    manual_filenames = [
        file
        for file in os.listdir(manuals_folder)
        if file.split(".")[-1] == "pdf"  # Get PDF extension files only
    ]

    assert manual_filenames, "no manuals to process"

    for manual_filename in manual_filenames:
        manual_path = os.path.abspath(f"./manuals/{manual_filename}")
        output_folder = f"output" + SEP + "{manual_filename}"
        output_images_folder = f"output" + SEP + {manual_filename} + SEP + "images"

        if not os.path.exists(os.path.abspath(output_folder)) or redo:
            os.makedirs(os.path.abspath("./output/"), exist_ok=True)
            os.makedirs(os.path.abspath("./output/images"), exist_ok=True)

            process_pdf_images(manual_path, output_folder, output_images_folder)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-m", "--manuals-folder", dest="manuals_folder", required=True)

    parser.add_argument(
        "-r", "--redo-processed-manuals", dest="redo_processed_manuals", default=False
    )

    args = parser.parse_args()

    main(args)
