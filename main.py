import os

import pymupdf
from argparse import ArgumentParser

SEP = "\\"

DEAD_RECT = pymupdf.Rect(1.0, 1.0, -1.0, -1.0)


def process_pdf_images(
    manual_path,
    output_file_path,
    output_folder_name,
    output_folder_path,
    output_image_folder_name,
    output_image_folder_path,
):

    doc = pymupdf.open(manual_path)

    for page_index, page in enumerate(doc):

        page_name = f"page {page_index+1}"

        image_refs = page.get_images(full=True)

        if image_refs:
            print(f"{len(image_refs)} image(s) found on page {page_index+1}")
            os.makedirs(output_image_folder_path + SEP + page_name, exist_ok=True)
        else:
            print(f"No images on page {page_index+1}")

        for image_index, image_ref in enumerate(image_refs):

            image_name = page_name + SEP + f"image {image_index+1}"

            image_file_name = output_image_folder_name + SEP + image_name
            image_file_path = os.path.abspath(image_file_name)

            xref = image_ref[0]
            pix = pymupdf.Pixmap(doc, xref)

            page.clean_contents()
            bbox = page.get_image_bbox(image_ref)
            # rects = page.get_image_rects(image_ref)

            if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            pix.save(f"{image_file_path}.png")

            if bbox == DEAD_RECT:
                rects = page.get_image_rects(image_ref)
                for rect in rects:
                    page.add_redact_annot(rect, text=image_file_name)
                    print(
                        f"Applying annotation for page {page_index+1}, image {image_index+1} at rect {rect}"
                    )
            else:
                print(
                    f"Applying annotation for page {page_index+1}, image {image_index+1} at bbox {bbox}"
                )

                page.add_redact_annot(bbox, text=image_file_name)

            page.apply_redactions()

    doc.save(output_file_path)


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
        manual_name = manual_filename.split(".")[0]
        manual_path = os.path.abspath(f"./manuals/{manual_filename}")

        output_folder_name = "output" + SEP + manual_name
        output_file_name = "output" + SEP + manual_name + SEP + manual_filename
        output_images_folder_name = f"output" + SEP + manual_name + SEP + "images"

        output_folder_path = os.path.abspath(output_folder_name)
        output_file_path = os.path.abspath(output_file_name)
        output_images_folder_path = os.path.abspath(output_images_folder_name)

        if not os.path.exists(os.path.abspath(output_folder_name)) or redo:
            print(f"processing {manual_name}")

            os.makedirs(output_folder_path, exist_ok=True)
            os.makedirs(output_images_folder_path, exist_ok=True)

            process_pdf_images(
                manual_path,
                output_file_path,
                output_folder_name,
                output_folder_path,
                output_images_folder_name,
                output_images_folder_path,
            )


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

    args = parser.parse_args()

    main(args)
