from loguru import logger
import time
import os
import ollama

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0,
    "top_p": 1,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


def gemini_generate(system_prompt, input_list):
    """Generate an output given a system prompt and an input list

    Args:
        system_prompt (str): the system prompt for the LLM
        input_list (list): a list of inputs for the LLM containing text and images(Part objects)

    Returns:
        dict: {"code": <response code (int)>, "response": <response text (str)>}
    """
    vertexai.init(project="iocl-426409", location="asia-south1")
    model = GenerativeModel(os.environ["LLM"], system_instruction=[system_prompt])
    try:
        response = model.generate_content(
            input_list,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        output = response.text
        return {"code": 200, "response": output}

    except Exception as e:
        logger.warning(f"An error occured during response generation using the {os.environ['LLM']} model, error : {e}")
        time.sleep(2)
        logger.warning("Sleeping for 60 seconds before trying again")
        time.sleep(60)

        try:
            response = model.generate_content(
                input_list,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            output = response.text
            return {"code": 200, "response": output}

        except Exception as e:
            return {
                "code": 512,
                "response": f"An error occured during response generation using the {os.environ['LLM']} model \n\n Error : {e}",
            }


def ollama_create():
    modelfile = '''
    FROM mistral:latest

    PARAMETER temperature 1
    PARAMETER top_p 1

    SYSTEM Follow the instructions and do not return anything other than the response
    '''

    ollama.create(model='gdch', modelfile=modelfile)
    return

def ollama_generate(prompt, model='gdch'):
    return ollama.generate(model=model, prompt=prompt)['response']