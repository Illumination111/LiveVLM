import numpy as np

def load_pretrained_model_arch(model_path, model_arch="llava_next"):

    if model_arch == "llava_next":
        from models.llava_next.model.builder import load_pretrained_model
        from models.llava_next.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(model_path)

        overwrite_config = None
       
        overwrite_config = {}
        overwrite_config["mm_newline_position"] = "no_token"
     
    
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, overwrite_config=overwrite_config)
        
    return tokenizer, model, image_processor


def format_mcqa_prompt(question, options):
   
    option_prompt = ""
    for idx, option in enumerate(options):
        choice_letter = chr(ord("A") + idx)
        if not ( option.startswith(f"{choice_letter}.") or option.startswith(f'({choice_letter})') or option.startswith(f'{choice_letter})') ):
            option = f"{choice_letter}. " + option
        option_prompt += option + "\n"
    
    formatted_question = f"Question: {question}\nOptions:\n{option_prompt}\nOnly give the best option."

    return formatted_question
