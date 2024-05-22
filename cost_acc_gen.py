import numpy as np
import pandas as pd

from calibration_dataset import CalibrationDataset
from calibration_run import CalibrationRun

import gsm8k_helpers

def cost_acc_curve_points(confidence_estimator, run_small, run_big, cost_small, cost_big):
    run_small_confs = run_small.get_confs(confidence_estimator)
    run_big_acc_by_q = run_big.get_confs(confidence_estimator)[['q_id', 'correct']].groupby('q_id').mean()

    curve_points = []
    for conf_threshold in np.linspace(0, 1, 100):
        attempts_above_conf_threshold = run_small_confs[run_small_confs[confidence_estimator] >= conf_threshold]
        attempts_below_conf_threshold = (
            run_small_confs[run_small_confs[confidence_estimator] < conf_threshold][['q_id']]
            .merge(run_big_acc_by_q, on='q_id', how='left')
            .reset_index()
        )
        pct_meeting_conf_threshold = len(attempts_above_conf_threshold) / len(run_small_confs)
        if confidence_estimator == "sampled_conf":
            cost = cost_small * 5 + cost_big * (1 - pct_meeting_conf_threshold)
        elif confidence_estimator == "all_tokens_logprob_conf":
            cost = cost_small + cost_big * (1 - pct_meeting_conf_threshold)
        acc = (
            attempts_above_conf_threshold['correct'].mean() * pct_meeting_conf_threshold
            + attempts_below_conf_threshold['correct'].mean() * (1 - pct_meeting_conf_threshold)
        )
        curve_points.append((conf_threshold, cost, acc))

    return pd.DataFrame(data=curve_points, columns=['conf_threshold', 'cost', 'acc'])

def generate_and_cache_cost_acc_pairs(dataset_name):
    model_name_small = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name_big = "meta-llama/Meta-Llama-3-70B-Instruct"
    confidence_estimator = "all_tokens_logprob_conf"
    cost_small = 0.1
    cost_big = 0.8
    seed = 42

    if dataset_name == "gsm8k":
        dataset = CalibrationDataset(
            dataset_name="gsm8k",
            df=gsm8k_helpers.get_test_df(),
            is_equiv=gsm8k_helpers.is_equiv,
            get_value_from_response=gsm8k_helpers.str_to_num_parser,
            get_prompt_content=gsm8k_helpers.get_prompt_content,
            get_value_tokens_prob=gsm8k_helpers.get_tokens_prob,
            seed=seed
        )
    elif dataset_name == "math":
        dataset = None
        # TODO James: implement math dataset

    run_small = CalibrationRun(
        dataset=dataset,
        model_name=model_name_small,
        num_questions=1000,
        num_attempts_per_question=20,
        num_shots=5,
        max_response_tokens=5000,
        requests_file_path=None,
        results_file_pattern=None,
    )

    run_big = CalibrationRun(
        dataset=dataset,
        model_name=model_name_big,
        num_questions=1000,
        num_attempts_per_question=20,
        num_shots=5,
        max_response_tokens=5000,
        requests_file_path=None,
        results_file_pattern=None,
    )

    run_small.get_results_from_serverless()
    run_big.get_results_from_serverless()

    curve_points = cost_acc_curve_points(confidence_estimator, run_small, run_big, cost_small, cost_big)
    curve_points.to_parquet(f"website/cached_acc_cost_pairs/{dataset_name}.pq")

    return curve_points

if __name__ == "__main__":
    generate_and_cache_cost_acc_pairs("gsm8k")