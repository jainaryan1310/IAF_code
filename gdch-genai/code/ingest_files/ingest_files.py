import configparser
import os
import random
from datetime import datetime
from loguru import logger
import magic
from ingest_text import ingest_text
from graph import add_entity, add_property, add_relation


def read_config():
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read('config.ini')

    llm = config.get("MODELS", "llm")
    source = config.get("INPUT", "source")

    os.environ["LLM"] = llm

    return {
        "source": source
    }


if __name__ == "__main__":

    config = read_config()
    source = config["source"]

    logger.info("LOADED THE CONFIG FILE")
    logger.info(config)

    output_folder = f"./logs/{datetime.now()}/"
    os.mkdir(output_folder)

    for file in os.listdir(source):
        try:
            r = random.randint(0, 10)*10

            logger.info(f"Ingesting file : {file}")
            file_type = magic.from_file(source+file, mime=True)
            if file_type == "text/plain":
                info = ingest_text(source, file, output_folder)
            else:
                continue

            entities = info["entities"]
            relations = info["relations"]
            properties = info["properties"]
            date = r

            for entity in entities:
                add_entity(entity, date)

            for edge in relations:
                add_relation(edge, date)

            for ppt in properties:
                add_property(ppt, date) 

        except Exception as e:
            logger.error(e)
            continue