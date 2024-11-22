from preprocess import split_pdf, get_markdowns, get_tables_figures
from index import make_docs, create_index, create_sparse_dense_index, get_docs
from loguru import logger



if __name__ == "__main__":

    input_folder = "../data/input_pdfs/"
    split_folder = "../data/split_pdfs/"
    processed_folder = "../data/processed_pdfs/"
    encoder_dir = "../data/encoder/"
    index_type = "hybrid"
    index_name = "iocl"

    '''
    split_pdf(input_folder, split_folder)
    logger.info("Split the pdfs into single pages")

    get_markdowns(split_folder, processed_folder)
    logger.info("Extracted markdown from the pages")

    get_tables_figures(processed_folder)
    logger.info("Extracted tables and figures from the pages")
    
    make_docs(processed_folder)
    logger.info("Created docs from processed files")
    '''

    docs = get_docs(processed_folder)
    logger.info("Created docs from processed files")


    if index_type == "hybrid":
        create_sparse_dense_index(docs, encoder_dir, index_name, "f5097558-0103-4a94-947e-aa86f17c571d")
        logger.info("Created a sparse-dense hybrid index by embedding the docs")

    else:
        index = create_index(docs, index_name, "f5097558-0103-4a94-947e-aa86f17c571d")
        logger.info("Created an index by embedding the docs")