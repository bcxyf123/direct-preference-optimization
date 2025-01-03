import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from scipy.stats import ttest_ind
from typing import Tuple


def load_local_model(
        model_name_or_path: str, 
        ckpt_path: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    model_name_or_path = model_name_or_path
    model_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, eos_token="<|endoftext|>")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to('cuda')
    model.config.eos_token_id = model_tokenizer.eos_token_id
    model.config.pad_token_id = model_tokenizer.eos_token_id
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state']
    model.load_state_dict(state_dict)

    return model, model_tokenizer


def compute_log_likelihood(model, tokenizer, prompt, response):
    inputs = tokenizer(prompt + response, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        log_likelihood = -outputs.loss.item()
    return log_likelihood


def compute_reward_score(reward_model, tokenizer, prompt, response):
    # 假设reward_model也支持transformers接口
    inputs = tokenizer(prompt + response, return_tensors="pt")
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # 假设reward score为最后一个维度的某种归一化值
        score = torch.mean(outputs.logits).item()
    return score

def perform_statistical_test(reward_scores_1, reward_scores_2, log_likelihoods_1, log_likelihoods_2):
    reward_t_stat, reward_p_val = ttest_ind(reward_scores_1, reward_scores_2)
    likelihood_t_stat, likelihood_p_val = ttest_ind(log_likelihoods_1, log_likelihoods_2)

    return {
        "reward_model": {
            "t_stat": reward_t_stat,
            "p_val": reward_p_val,
        },
        "log_likelihood": {
            "t_stat": likelihood_t_stat,
            "p_val": likelihood_p_val,
        },
    }


def main():

    # load model
    model_name_or_path = "gpt2"
    model_ckpt1 = "path_to_gpt2_ckpt1"
    model_ckpt2 = "path_to_gpt2_ckpt2"
    reward_model_ckpt = "path_to_reward_model"

    model1, tokenizer1 = load_local_model(model_name_or_path, model_ckpt1)
    model2, tokenizer2 = load_local_model(model_name_or_path, model_ckpt2)

    # load external reward model
    reward_model = AutoModelForCausalLM.from_pretrained(reward_model_ckpt).to('cuda')

    prompts = ["Example prompt 1", "Example prompt 2"]  # 示例prompt
    responses_1, responses_2 = [], []

    # gather responses
    for prompt in prompts:
        response_1 = tokenizer1.decode(model1.generate(tokenizer1.encode(prompt, return_tensors="pt"))[0])
        response_2 = tokenizer2.decode(model2.generate(tokenizer2.encode(prompt, return_tensors="pt"))[0])
        responses_1.append(response_1)
        responses_2.append(response_2)

    # compute reward scores and log likelihoods
    reward_scores_1 = [compute_reward_score(reward_model, reward_tokenizer, p, r) for p, r in zip(prompts, responses_1)]
    reward_scores_2 = [compute_reward_score(reward_model, reward_tokenizer, p, r) for p, r in zip(prompts, responses_2)]

    log_likelihoods_1 = [compute_log_likelihood(model1, tokenizer1, p, r) for p, r in zip(prompts, responses_1)]
    log_likelihoods_2 = [compute_log_likelihood(model2, tokenizer2, p, r) for p, r in zip(prompts, responses_2)]

    # perform statistical test
    results = perform_statistical_test(reward_scores_1, reward_scores_2, log_likelihoods_1, log_likelihoods_2)

    print("Statistical Test Results:")
    print(results)



if __name__ == "__main__":
    main()
