import sys
import requests
import time
from message_templates import get_message_template

import re

def remove_markdown_code_formatting(markdown_text):
    """
    Removes Markdown code block formatting from the given text.

    Parameters:
        markdown_text (str): A string containing Markdown-formatted code block(s).

    Returns:
        str: The text with Markdown code block formatting removed.
    """
    # Regular expression pattern to match code blocks with optional language specifier
    pattern = r'```(?:\w+)?\n([\s\S]*?)```'

    # Function to replace the matched pattern with the captured code
    def replace_code_block(match):
        return match.group(1)

    # Substitute code blocks in the markdown text
    result = re.sub(pattern, replace_code_block, markdown_text)

    return result


def get_payload(prefix, suffix, lang="python"):

    messages = get_message_template(prefix, suffix, lang)

    return {
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.0,
        "stop": "<|endoftext|>"
    }


def call_chat_api(model_url, api_key, prompt, suffix, lang="python"):

    # Set up the headers with the API key for authentication
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    payload = get_payload(prompt, suffix, lang)

    # Make the POST request to the Azure OpenAI Chat API
    response = requests.post(model_url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the generated text
        response_data = response.json()
        generated_message = response_data["choices"][0]["message"]["content"]
        generated_message = remove_markdown_code_formatting(generated_message)
        # if not generated_message.endswith("\n"):
        #    generated_message += "\n"

        return generated_message
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}, {response.text}")
        return None


if __name__ == "__main__":
    start = time.perf_counter()
    prompt = "def calculate_circle_area(radius): \n     pi = 3.14 \n     area = "
    suffix = "return area"
    model_url = sys.argv[1]
    api_key = sys.argv[2]
    generated_message = call_chat_api(
        model_url, api_key, prompt, suffix, lang="python")
    print(f"{generated_message}")
    print("Time elapsed: ", time.perf_counter() - start,  "seconds\n")
