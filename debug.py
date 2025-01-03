import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

from testing import load_local_model

def load_model_debug():

    model_name_or_path = "gpt2"
    model_ckpt1 = "save/sft/policy.pt"
    model_ckpt2 = "save/dpo/policy.pt"
    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

    model1, tokenizer1 = load_local_model(model_name_or_path, model_ckpt1)
    model2, tokenizer2 = load_local_model(model_name_or_path, model_ckpt2)

    rank_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

    question, answer = "Explain nuclear fusion like I am five", "Nuclear fusion is the process by which two or more protons and neutrons combine to form a single nucleus. It is a very important process in the universe, as it is the source of energy for stars and galaxies. Nuclear fusion is also a key process in the production of energy for nuclear power plants."
    inputs = tokenizer(question, answer, return_tensors='pt')
    score = rank_model(**inputs).logits[0].cpu().detach()
    print(score)


def generation_debug():

    model_name_or_path = "gpt2"
    model_ckpt1 = "save/sft/policy.pt"
    model_ckpt2 = "save/dpo/policy.pt"
    reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

    model1, tokenizer1 = load_local_model(model_name_or_path, model_ckpt1)
    model2, tokenizer2 = load_local_model(model_name_or_path, model_ckpt2)
    reward_model, reward_tokenizer = AutoModelForSequenceClassification.from_pretrained(reward_name), AutoTokenizer.from_pretrained(reward_name)

    question = "Explain nuclear fusion like I am five"
    inputs = tokenizer1(question, return_tensors="pt").to('cuda')
    with torch.no_grad():
        output = model1.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer1.eos_token_id)
        response1 = tokenizer1.decode(output[0])
    print(response1)

    question, answer = question, response1
    inputs = reward_tokenizer(question, answer, return_tensors='pt')
    score = reward_model(**inputs).logits[0].cpu().detach()
    print(score)

    inputs = tokenizer2(question, return_tensors="pt").to('cuda')
    with torch.no_grad():
        output = model2.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer2.eos_token_id)
        response2 = tokenizer2.decode(output[0])
    print(response2)

    question, answer = question, response2
    inputs = reward_tokenizer(question, answer, return_tensors='pt')
    score = reward_model(**inputs).logits[0].cpu().detach()
    print(score)


if __name__ == "__main__":
    
    # load_model_debug()
    generation_debug()