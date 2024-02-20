import json
import re

import torch

def extract_function_calls(completion):
    if isinstance(completion, str):
        content = completion
    else:
        content = completion.content

    pattern = r"<multiplefunctions>(.*?)</multiplefunctions>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None

    multiplefn = match.group(1)
    functions = []
    for fn_match in re.finditer(r"<functioncall>(.*?)</functioncall>", multiplefn, re.DOTALL):
        fn_text = fn_match.group(1)
        try:
            functions.append(json.loads(fn_text))
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON

    return functions

def generate_hermes(prompt, model, tokenizer, similarity_search_tool, generation_config_overrides={}):
    with torch.inference_mode():
        completion = model.invoke([{"role": "user", "content": prompt}])

    if isinstance(completion, str):
        # Handle the case where completion is a string
        content = completion.strip()
    else:
        # Handle the case where completion is an AIMessage object
        content = completion.content.strip()

    functions = extract_function_calls(content)

    if functions:
        for function_call in functions:
            function_name = function_call.get("name")
            arguments = function_call.get("arguments")
            if function_name == "SimilaritySearchTool":
                response = similarity_search_tool.run(**arguments)
                print(response)
            else:
                print(f"Unknown function: {function_name}")
    else:
        print(content)
    print("="*100)
