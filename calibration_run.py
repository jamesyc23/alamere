import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
from tqdm import tqdm
import multiprocess as mp

def get_one_response_wrapper(get_prompt_content, examples_text, model, max_tokens):
    def local_client():
        from openai import OpenAI
        import os

        client = OpenAI(
            base_url="https://api.runpod.ai/v2/fehv3wh9hksuwk/openai/v1",
            api_key=os.environ["RUNPOD_API_KEY"],
        )
        return client
        

    def get_one_response(question):
        client = local_client()
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": get_prompt_content(question, examples_text)
            }],
            model=model,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=1,
        )
        return response
    
    return get_one_response

class CalibrationRun:
    def __init__(
        self,
        dataset,
        model_name,
        num_questions,
        num_attempts_per_question,
        num_shots,
        max_response_tokens,
        requests_file_path,
        results_file_pattern,
    ):
        self.dataset = dataset
        self.model_name = model_name
        self.num_questions = num_questions
        self.num_attempts_per_question = num_attempts_per_question
        self.num_shots = num_shots
        self.max_response_tokens = max_response_tokens
        self.requests_file_path = requests_file_path
        self.results_file_pattern = results_file_pattern
        self.results = None

    def write_requests_file(self):
        df = self.dataset.df.sample(self.num_questions, random_state=self.dataset.seed).copy()

        att = pd.concat([df]*self.num_attempts_per_question, ignore_index=True)
        with open(self.requests_file_path, "w") as f:
            for index, row in att.iterrows():
                dict_for_one_request = {
                    "custom_id": f"request_{index}_qid_{row.q_id}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": self.dataset.get_prompt_content(
                                    question=row.question,
                                    examples_text=self.dataset.get_examples_text(self.num_shots),
                                )
                            }
                        ],
                        "max_tokens": self.max_response_tokens,
                        "logprobs": True,
                        "top_logprobs": 1,
                    }
                }
                print(json.dumps(dict_for_one_request), file=f)

    def read_results_file(self):
        def parse_json_string(line):
            try:
                return json.loads(line)
            except ValueError:
                return None
        
        lines = []

        for file in glob.glob(self.results_file_pattern):
            with open(file, "r") as f:
                lines.extend([parse_json_string(line.rstrip()) for line in f])
        
        line_count = len(lines)
        lines = [line for line in lines if line is not None]
        lines_dropped = line_count - len(lines)
        print(f"Lines dropped: {lines_dropped}")
        df = pd.DataFrame.from_dict(lines)
        df["q_id"] = df["custom_id"].str.split("_").str[3].astype(int)
        df["attempt"] = df["response"].apply(
            lambda r: r['choices'][0]['message']['content']
        )
        df['attempt_value'] = df['attempt'].apply(self.dataset.get_value_from_response)
        df['value_tokens_prob'] = df['response'].apply(lambda r: self.dataset.get_value_tokens_prob(r['choices'][0]['logprobs']))
        df['all_tokens_logprob'] = df['response'].apply(lambda r: sum(r['choices'][0]['logprobs']['token_logprobs']))

        df = (
            df[['q_id', 'attempt', 'attempt_value', 'value_tokens_prob', 'all_tokens_logprob']]
            .merge(self.dataset.df[['q_id', 'question', 'answer']], how='left', on='q_id')
        )
        df['correct'] = df.progress_apply(lambda row: 1 if self.dataset.is_equiv(row['attempt'], row['answer']) else 0, axis=1)

        self.results = df

    def get_results_from_serverless(self):
        get_prompt_content = self.dataset.get_prompt_content
        examples_text = self.dataset.get_examples_text(self.num_shots)
        model=self.model_name
        max_tokens=self.max_response_tokens
        
        df = self.dataset.df.sample(self.num_questions, random_state=self.dataset.seed).copy()
        df = pd.concat([df]*self.num_attempts_per_question, ignore_index=True)

        with mp.Pool(50) as p:
            results = list(tqdm(p.imap(get_one_response_wrapper(
                get_prompt_content=get_prompt_content,
                examples_text=examples_text,
                model=model,
                max_tokens=max_tokens,
            ), df.question)))

        df['response'] = results

        df["attempt"] = df["response"].apply(
            lambda r: r['choices'][0]['message']['content']
        )
        df['attempt_value'] = df['attempt'].apply(self.dataset.get_value_from_response)
        df['value_tokens_prob'] = df['response'].apply(lambda r: self.dataset.get_value_tokens_prob(r['choices'][0]['logprobs']))
        df['all_tokens_logprob'] = df['response'].apply(lambda r: sum(r['choices'][0]['logprobs']['token_logprobs']))

        df = (
            df[['q_id', 'attempt', 'attempt_value', 'value_tokens_prob', 'all_tokens_logprob']]
            .merge(self.dataset.df[['q_id', 'question', 'answer']], how='left', on='q_id')
        )
        df['correct'] = df.progress_apply(lambda row: 1 if self.dataset.is_equiv(row['attempt'], row['answer']) else 0, axis=1)

        self.results = df

    def get_confs(self, confidence_estimator):
        assert self.results is not None, "results not yet parsed"
        if confidence_estimator == "sampled_conf":
            test = self.results[['q_id', 'attempt_value', 'correct']].groupby("q_id").head(5)
            confs = test.merge(
                test[['q_id', 'attempt_value']]
                .assign(sampled_conf=1/5)
                .groupby(['q_id', 'attempt_value'])
                .sum()
                .reset_index(),
                on=['q_id', 'attempt_value'],
                how='left',
            )

        elif confidence_estimator ==  "value_tokens_prob":
            confs = self.results[['q_id', 'value_tokens_prob', 'correct']]

        elif confidence_estimator ==  "all_tokens_logprob_conf":
            X = self.results[['all_tokens_logprob']].assign(ones=1).to_numpy()
            y = self.results['correct'].to_numpy()
            betas = np.linalg.inv(X.T @ X) @ X.T @ y
            confs = self.results[['q_id', 'correct']].assign(all_tokens_logprob_conf = (X @ betas))

        return confs
    
    def top1_acc(self):
        return self.get_confs(confidence_estimator="sampled_conf")['correct'].mean()
    
    def get_binned(self, confidence_estimator, qa_pairs_per_bin):
        return (
            self.get_confs(confidence_estimator)
            .sort_values(confidence_estimator)
            .reset_index(drop=True)
            .assign(bin=lambda row: row.index // qa_pairs_per_bin)[['bin', confidence_estimator, 'correct']]
            .groupby('bin')
            .mean()
        )

    def plot_estimated_confidence_vs_accuracy(self, confidence_estimator, qa_pairs_per_bin=25):
        binned = self.get_binned(confidence_estimator, qa_pairs_per_bin)
        fig, ax = plt.subplots()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.scatter(binned[confidence_estimator], binned['correct'])
        ax.plot([0,1],[0,1], transform=ax.transAxes)

        plt.title(f"{self.model_name} calibration\n({self.dataset.dataset_name} {self.num_shots}-shot, {self.num_questions} Qs, {self.num_attempts_per_question} attempts/Q)")
        plt.xlabel(confidence_estimator)
        plt.ylabel('accuracy')

        plt.show()