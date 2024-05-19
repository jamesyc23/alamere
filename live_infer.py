from openai import OpenAI
import os
from gsm8k_helpers import *
import numpy as np

def llama3_70b_client():
# Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_base = "https://api.runpod.ai/v2/fehv3wh9hksuwk/openai/v1"
    openai_api_key = os.environ["RUNPOD_API_KEY"]
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def llama3_8b_client():
# Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_base = "https://api.runpod.ai/v2/p8qpxyyrfn65ev/openai/v1"
    openai_api_key = os.environ["RUNPOD_API_KEY"]
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client


def construct_request(question):
    examples_text = examples_text_5_shot
    return {
        "messages": [{
            "role": "user",
            "content": get_prompt_content(question, examples_text),
        }],
        "top_logprobs": 1,
        "logprobs": True,
    }


def infer_with_fallback(question, threshold):
    client = llama3_8b_client()

    response = client.chat.completions.create(model="meta-llama/Meta-Llama-3-8B-Instruct", **construct_request(question))

    cumulative_logprobs = sum(response.choices[0].logprobs.token_logprobs)
    confidence = np.dot(np.array([cumulative_logprobs, 1]), np.array([0.01186387, 1.05021634]))
    explanation = response.choices[0].message.content
    answer = str_to_num_parser(response.choices[0].message.content)

    client = llama3_70b_client()

    if confidence < threshold:
        response = client.chat.completions.create(model="meta-llama/Meta-Llama-3-70B-Instruct", **construct_request(question))
        cumulative_logprobs = sum(response.choices[0].logprobs.token_logprobs)
        confidence = np.dot(np.array([cumulative_logprobs, 1]), np.array([0.01186387, 1.05021634]))
        return response.choices[0].message.content, confidence, "Llama3-70B-Instruct"
    else:
        return explanation, confidence, "Llama3-8B-Instruct"

if __name__ == "__main__":
    threshold = 0.9
    question = "Jack and Jill each brought a bottle of water up a hill. How many bottles of water did they bring up the hill combined?"
    print(infer_with_fallback(question, threshold))
