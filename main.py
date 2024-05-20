import os

import pymupdf
from argparse import ArgumentParser


def process_pdf_images(filename):

    # manual_path=os.path.abspath(f"./manuals/{}")
    manual_name = "".join(filename.split(".")[:-1])

    os.makedirs(os.path.abspath(f"./output/{manual_name}"), exist_ok=True)

    doc = pymupdf.open()


def main(args):
    manuals_path = os.path.abspath(args.manuals_path)

    manuals = [file for file in os.listdir(manuals_path) if os.path.isfile(file)]

    for manual in manuals:
        print(manual)


if __name__ == "__main__":

    parser = ArgumentParser()

    args = parser.parse_args()

    main(args)
