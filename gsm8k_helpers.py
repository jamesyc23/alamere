import pandas as pd
from math import exp

from datasets import load_dataset

def str_to_num_parser(s : str) -> float:
    if isinstance(s, float) or isinstance(s, int):
        return s
    try:
        ending = s.split("####")[-1]
        strip_non_numbers = "".join((c for c in ending if (c in "1234567890.-")))
        return float(strip_non_numbers)
    except Exception as e:
        return float("nan")
    
def is_equiv(s1, s2):
    return str_to_num_parser(s1) == str_to_num_parser(s2)
    
def get_prompt_content(question, examples_text):
    return (
        "Please answer the following question.\n\n"
        + f"Question: {question}\n\n"
        + "Please give your reasoning, then output your final answer as a single number immediately preceded by #### with nothing after.\n\n"
        + f"Examples:\n\n{examples_text}"
    )

def get_tokens_prob(logprobs):
    token_count = 0
    for token in logprobs['tokens'][::-1][1:]:
        if token == "####":
            break
        token_count += 1
    tokens_prob = exp(sum(logprobs['token_logprobs'][-(token_count):-1]))
    return tokens_prob

def get_test_df():
    return pd.DataFrame(load_dataset("gsm8k", "main")["test"])