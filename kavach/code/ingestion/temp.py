from index import make_docs, create_sparse_dense_index
from loguru import logger


if __name__ == "__main__":

    processed_folder = "../data/processed_pdfs/"
    encoder_dir = "../data/encoder/"

    docs = make_docs(processed_folder)
    logger.info("Created docs from processed files")

    create_sparse_dense_index(docs, encoder_dir, "kavach2", "f5097558-0103-4a94-947e-aa86f17c571d")
    logger.info("Created an index by embedding the docs")