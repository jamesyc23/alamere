class CalibrationDataset:
    def __init__(
        self,
        dataset_name,
        df,
        is_equiv,
        get_value_from_response,
        get_prompt_content,
        get_value_tokens_prob,
        seed,
    ):
        self.dataset_name = dataset_name
        self.df = df
        self.is_equiv = is_equiv
        self.get_value_from_response = get_value_from_response
        self.get_prompt_content = get_prompt_content
        self.get_value_tokens_prob = get_value_tokens_prob
        self.seed = seed

        self.df['q_id'] = self.df.index

    def get_examples_text(self, num_shots):
        return "\n\n".join(
            self.df.sample(num_shots, random_state=self.seed)
            .apply(lambda row: f"Question: {row['question']}\n\nAnswer: {row['answer']}", axis=1)
        )