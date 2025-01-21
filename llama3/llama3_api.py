import requests
import torch
# Define the local Ollama API endpoint
base_url = "http://localhost:11434"

# Model and prompt for the request
model_name = "llama3"  # Replace with your model's name
prompt = "What is the capital of France?"

# Prepare the request payload
payload = {"model": model_name, "prompt": prompt}

# Send the POST request to the Ollama server
response = requests.post(f"{base_url}/api/generate", json=payload)
# Print the raw response for debugging

# try:
#     result = response.json()
#     print("Response JSON:", result)
# except requests.exceptions.JSONDecodeError:
#     print("Non-JSON response received:")
#     print(response.text)


def llama_inference(prompt, model_name="llama3", base_url="http://localhost:11434"):
    """
    Query the Ollama API with a prompt and receive the response.

    Args:
        prompt (str): The prompt to send to the model.
        model_name (str): The name of the model to query. Default is "llama3".
        base_url (str): The base URL of the Ollama server. Default is "http://localhost:11434".

    Returns:
        dict or str: The JSON response from the server if valid, otherwise the raw text response.
    """
    # Prepare the request payload
    payload = {"model": model_name, "prompt": prompt}

    try:
        # Send the POST request
        response = requests.post(f"{base_url}/api/generate", json=payload)

        # Check if the response is JSON
        try:
            return response.json()  # Return parsed JSON response
        except requests.exceptions.JSONDecodeError:
            return response.text  # Return raw text response if not JSON

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    prompt = "can you take image as input?"
    result = llama_inference(prompt)