import sys
import requests
import time

def call_fim_api(model_url, api_key, prompt, suffix):

    # Set up the headers with the API key for authentication
    headers = {
        "api-key": api_key,
        "Openai-Internal-SuffixMode": "SPM",
        "Openai-Internal-EnableSuffixSupport": "True"
    }

    payload = {"temperature": 0, "max_tokens": 1024, "stop": "<|endoftext|>", "prompt": prompt, "suffix": suffix} 

    # Make the POST request to the Azure OpenAI Chat API
    response = requests.post(model_url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Print the generated text
        response_data = response.json()

        generated_message = response_data["choices"][0]["text"]        
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
    start = time.perf_counter()
    generated_message = call_fim_api(model_url, api_key, prompt, suffix)
    print(f"{generated_message}")
    print("Time elapsed: ", time.perf_counter() - start,  "seconds\n")