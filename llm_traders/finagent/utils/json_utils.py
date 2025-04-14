import json
import json5
import numpy as np
import re
from bs4 import BeautifulSoup

def load_json(file_path):
    with open(file_path, mode='r', encoding='utf8') as fp:
        json_dict = json5.load(fp)
        return json_dict
    
def save_json(json_dict, file_path, indent=4):
    with open(file_path, mode='w', encoding='utf8') as fp:
        try:
            if indent == -1:
                json.dump(json_dict, fp, ensure_ascii=False)
            else:
                json.dump(json_dict, fp, ensure_ascii=False, indent=indent)
        except Exception as e:
            if indent == -1:
                json5.dump(json_dict, fp, ensure_ascii=False)
            else:
                json5.dump(json_dict, fp, ensure_ascii=False, indent=indent)

def check_json(json_string):
    try:
        json5.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

def refine_json(json_string):
    json_string = json_string.strip().replace('\r', '').replace('\t', '').replace('\n', '').replace('\\', '')

    json_string = re.sub(r',\s*([}\]])', r'\1', json_string)
    json_string = re.sub(r'\.\s*([}\]])', r'\1', json_string)

    pattern = r"\{(.*)\}"
    match = re.search(pattern, json_string, re.DOTALL)

    if match:
        json_string = "{" + match.group(1) + "}"
        if check_json(json_string):
            return json_string
        else:
            return json_string

    return json_string


def parse_semi_formatted_json(json_string):

    obj = None

    try:
        response = refine_json(json_string)
        print(response)
        obj = json5.loads(response)

    except Exception as e:
        raise e

    return obj

def parse_semi_formatted_xml(text):
    text = text.strip().replace('\r', '').replace('\t', '').replace('\n', '')

    # text = text.replace('```xml', '').replace('```', '')

    soup = BeautifulSoup(text, 'html.parser')

    obj = {}

    output = soup.find('output')
    all_children = output.find_all(recursive=False)

    for child in all_children:
        tag_name = child.name
        name = child.get('name').lower()

        if tag_name == 'string':
            contents = child.text.replace('\r', '').replace('\t', '').split('\n')
            contents = [content for content in contents if content != '']
            content = "\n".join(contents)
            obj[name] = content
        elif tag_name == "list":
            obj[name] = []
            for item_tag in child.find_all('item'):
                item = {}
                for item_child in item_tag.find_all(recursive=False):
                    item_name = item_child.name.lower()
                    item_content = item_child.text.replace('\r', '').replace('\t', '').replace('\n', '')
                    item[item_name] = item_content
                obj[name].append(item)

        elif tag_name == "map":
            obj[name] = {}
            for item_tag in child.find_all('string'):
                item_name = item_tag.get('name').lower()
                item_content = item_tag.text.replace('\r', '').replace('\t', '').replace('\n', '')
                obj[name][item_name] = item_content
    return obj, soup

def convert_to_json_serializable(data):
    """
    Recursively converts int64 and float64 to int and float in a dictionary.

    Parameters:
    - data (dict): Input dictionary with potentially non-serializable types.

    Returns:
    - dict: Output dictionary with serializable types.
    """
    for key, value in data.items():
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            data[key] = int(value)
        elif isinstance(value, np.float64):
            data[key] = float(value)
        elif isinstance(value, dict):
            # Recursively convert nested dictionaries
            data[key] = convert_to_json_serializable(value)
    return data