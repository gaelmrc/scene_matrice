import sys
from yachalk import chalk
sys.path.append("..")
import json
import ollama.client as client
from typing import List, Dict, Any
import re

def extract_json_from_response(response_text):
    try:
        # Utiliser une expression régulière pour extraire le bloc JSON
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            print("Aucun bloc JSON trouvé dans la réponse.")
            return None
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON : {e}")
        return None

def entity_extract_prompt(input: str, metadata={}, model="llama3:latest"):
    model_info = client.show(model_name=model)
    #print( chalk.blue(model_info))
    SYS_PROMPT = (
    "Format your output as a JSON object with a single key 'entities' which is a list of entity objects. "
    "Each entity object must have exactly one key: 'name' (the name of the entity)"
    "The JSON object should be formatted exactly as follows:\n"
    "[\n"
        "   {\n"
        '       "entity": "An entity extracted from the text",\n'
        "       \n"
        "   }, {...}\n" #indique qu'il peut y avoir plusieurs éléments dans la liste
        "]"
    "For example, if the input text is:\n"
    "\"Alice went to Microsoft headquarters in New York on January 1, 2021, at 10:00 AM. She spent $100 on her trip, which accounted for 50% of her budget.\"\n"
    "The output should be:\n"
    "[\n"
    '    {"entity" : "Alice"},\n'
    '    {"entity" : "Microsoft"},\n'
    '    {"entity" : "New York"},\n'
    '    {"entity" : "January 1, 2021"},\n'
    '    {"entity" : "10:00 AM"},\n'
    '    {"entity" : "$100"},\n'
    '    {"entity" : "50 %"}\n'
    "  ]"
    "}"
)
    

    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    try:
        response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
        result = extract_json_from_response(response)
        result = [dict(item, **metadata) for item in result]
        
    except json.JSONDecodeError as e:
        print(f"\n\nJSON decode error: {e}\nResponse: {response}\n\n")
        result = None
    except Exception as e:
        print(f"\n\nERROR ### {str(e)}\nResponse: {response}\n\n")
        result = None
    return result