import json
import os

from llm_factory import generate
from loguru import logger
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader, PdfWriter
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.settings import settings
from tqdm import tqdm
from vertexai.generative_models import Image as IMG
from vertexai.generative_models import Part


def split_pdf(input_folder: str, split_folder: str):
    pdf_files = os.listdir(input_folder)

    logger.info("Splitting the following PDFs")
    logger.info(pdf_files)

    for pdf_name in pdf_files:
        pdf_path = input_folder + pdf_name
        output_path = split_folder + pdf_name[:-4] + "/"

        os.mkdir(output_path)

        inputpdf = PdfReader(open(pdf_path, "rb"))

        for i in tqdm(range(len(inputpdf.pages))):
            output = PdfWriter()
            output.add_page(inputpdf.pages[i])
            with open(output_path + pdf_name + str(i) + ".pdf", "wb") as outputStream:
                output.write(outputStream)

    return


def get_image_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    return images[0]


def get_markdowns(split_folder: str, processed_folder: str):
    pdf_names = os.listdir(split_folder)
    models = load_all_models()

    for pdf_name in pdf_names:
        split_pdf_folder = split_folder + pdf_name + "/"
        processed_pdf_folder = processed_folder + pdf_name + "/"
        os.mkdir(processed_pdf_folder)

        pdf_page_files = os.listdir(split_pdf_folder)

        for pdf_page_file in pdf_page_files:
            split_pdf_page_file = split_pdf_folder + pdf_page_file
            pdf_page_name = pdf_page_file[:-4]
            processed_pdf_page_folder = processed_pdf_folder + pdf_page_name + "/"
            os.mkdir(processed_pdf_page_folder)

            markdown = convert_single_pdf(split_pdf_page_file, models)[0]
            image = get_image_from_pdf(split_pdf_page_file)

            image.save(processed_pdf_page_folder + "page.jpg", "JPEG")
            with open(processed_pdf_page_folder + "text.md", "w") as f:
                f.write(markdown)

    return


def get_table_caption(table_image_path: str, page_image_path: str, page_md: str):
    system_prompt = """
You will be provided with:

1. An image of a table.
2. An image of the page containing the table.
3. Text from the page in markdown format.

Using the above, generate a concise caption for the table that highlights:

- The main subject or purpose of the table.
- Key insights or information it adds to the surrounding text.
- Use the context to understand and include how the table aligns with and 
supports the info on the page

Return only a caption and nothing else
"""

    table_image = Part.from_image(IMG.load_from_file(table_image_path))
    page_image = Part.from_image(IMG.load_from_file(page_image_path))

    response_dict = generate(system_prompt, [table_image, page_image, page_md])

    if response_dict["code"] == 200:
        caption = response_dict["response"]

    else:
        caption = "NA"

    return caption


def get_figure_caption(figure_image_path: str, page_image_path: str, page_md: str):
    system_prompt = """
You will be provided with:

1. An image of a figure.
2. An image of the page containing the figure.
3. Text from the page in markdown format.

Using the above, generate a concise caption for the figure that highlights:

- The main subject or purpose of the figure.
- Key insights or information it adds to the surrounding text.
- Use the context to understand and include how the figure aligns with and 
supports the info on the page

Return only a caption and nothing else
"""

    figure_image = Part.from_image(IMG.load_from_file(figure_image_path))
    page_image = Part.from_image(IMG.load_from_file(page_image_path))

    response_dict = generate(system_prompt, [figure_image, page_image, page_md])

    if response_dict["code"] == 200:
        caption = response_dict["response"]

    else:
        caption = "NA"

    return caption


def get_tables_figures(processed_folder: str):
    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    pdf_names = os.listdir(processed_folder)
    for pdf_name in pdf_names:
        processed_pdf_folder = processed_folder + pdf_name + "/"
        pdf_page_names = os.listdir(processed_pdf_folder)

        for pdf_page_name in pdf_page_names:
            processed_pdf_page_folder = processed_pdf_folder + pdf_page_name + "/"
            page_image_path = processed_pdf_page_folder + "page.jpg"

            image = Image.open(page_image_path)

            line_predictions = batch_text_detection([image], det_model, det_processor)
            layout_predictions = batch_layout_detection(
                [image], model, processor, line_predictions
            )

            bboxes = layout_predictions[0].model_dump()["bboxes"]

            table_num = 0
            fig_num = 0

            with open(processed_pdf_page_folder + "text.md", "r") as f:
                page_md = f.read()

            if page_md == "":
                page_md = "This page has no text."

            captions = {}

            for bbox in bboxes:
                if bbox["label"] == "Table":
                    table = image.crop(bbox["bbox"])
                    table_image_path = (
                        processed_pdf_page_folder + "table" + str(table_num) + ".jpg"
                    )
                    table.save(table_image_path, "JPEG")

                    caption = get_table_caption(
                        table_image_path, page_image_path, page_md
                    )
                    captions["table" + str(table_num)] = caption

                    table_num += 1

                if bbox["label"] == "Figure":
                    figure = image.crop(bbox["bbox"])
                    figure_image_path = (
                        processed_pdf_page_folder + "fig" + str(fig_num) + ".jpg"
                    )
                    figure.save(figure_image_path, "JPEG")

                    caption = get_figure_caption(
                        figure_image_path, page_image_path, page_md
                    )
                    captions["fig" + str(fig_num)] = caption

                    fig_num += 1

            with open(
                processed_pdf_page_folder + "captions.json", "w", encoding="utf-8"
            ) as f:
                json.dump({"captions": captions}, f, ensure_ascii=False, indent=4)

    return
