import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import glob
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm.auto import tqdm




NUM_SAMPLES_PER_PROMPT = 5
NUM_OF_TESTS = 50  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SFT_MODEL_PATH = "./fine-tuned-gpt2-large"


DPO_FINAL_MODEL_PATHS = {
    "DPO beta=0.05": './dpo-gpt2-large-beta-0.05',
    "DPO beta=0.1": './dpo-gpt2-large-beta-0.1',
    "DPO beta=1.0": './dpo-gpt2-large-beta-1.0'
}


DPO_CHECKPOINT_DIRS = {
    "DPO beta=0.05": './dpo_output_beta_0.05',
    "DPO beta=0.1": './dpo_output_beta_0.1',
    "DPO beta=1.0": './dpo_output_beta_1.0'
}


models_to_evaluate = []


models_to_evaluate.append({
    "name": "SFT (ref)",
    "path": SFT_MODEL_PATH,
    "beta_group": "SFT",
    "step": 0
})


print("Finding checkpoints...")
for beta_name, base_dir in DPO_CHECKPOINT_DIRS.items():
    if not os.path.isdir(base_dir):
        print(f"Warning: Checkpoint directory not found, skipping: {base_dir}")
        continue
        
    checkpoint_paths = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    
    
    def get_step_num(path):
        try:
            num_str = re.search(r'checkpoint-(\d+)', path)
            return int(num_str.group(1)) if num_str else 0
        except:
            return 0
    
    checkpoint_paths.sort(key=get_step_num)

    for ckpt_path in checkpoint_paths:
        step = get_step_num(ckpt_path)
        if step == 0 or not os.path.isdir(ckpt_path):
            continue 
            
        ckpt_name = os.path.basename(ckpt_path)
        models_to_evaluate.append({
            "name": f"{beta_name} ({ckpt_name})",
            "path": ckpt_path,
            "beta_group": beta_name,
            "step": step
        })
        

for beta_name, model_path in DPO_FINAL_MODEL_PATHS.items():
    if not os.path.isdir(model_path):
        print(f"Warning: Final model path not found, skipping: {model_path}")
        continue
        
    models_to_evaluate.append({
        "name": f"{beta_name} (Final)",
        "path": model_path,
        "beta_group": beta_name,
        "step": 99999 
    })
    
print("--- Models to be Evaluated ---")
for model_info in models_to_evaluate:
    print(f"{model_info['name']:<40} -> {model_info['path']}")
print("---------------------------------")





print(f"Loading reference SFT model from {SFT_MODEL_PATH}...")
ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(DEVICE).eval()
ref_tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
if ref_tokenizer.pad_token is None:
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
print("SFT model and tokenizer loaded.")



print("Loading sentiment classifier...")
clf_device = 0 if torch.cuda.is_available() else -1
sentiment_classifier = pipeline(
    "text-classification",
    model="siebert/sentiment-roberta-large-english",
    top_k=None,
    device=clf_device,
)
print("Classifier loaded.")


print("Loading 'imdb' test split and filtering for positive prompts...")
testing_dataset = load_dataset('stanfordnlp/imdb', split='test')


positive_dataset = testing_dataset.filter(lambda example: example['label'] == 1)

prompts = []

num_available = min(NUM_OF_TESTS, len(positive_dataset)) 

if num_available < NUM_OF_TESTS:
    print(f"Warning: Found only {num_available} positive prompts, requested {NUM_OF_TESTS}.")

for i in range(num_available):
    text = positive_dataset[i]['text'] 
    words = text.split()
    prefix_length = random.randint(2, 8)
    prompt_text = " ".join(words[:prefix_length])
    prompts.append(prompt_text)

print(f"Loaded {len(prompts)} positive test prompts.")



def get_positive_prob(scores):
    for d in scores:
        if d['label'] == 'POSITIVE':
            return d['score']
    return 0.0

def get_completion_log_probs(model, input_ids, attention_mask, completion_tokens):
    """
    Runs a forward pass to get log-probs for the generated completion tokens.
    """
    full_sequence = torch.cat([input_ids, completion_tokens.unsqueeze(0)], dim=-1)
    
    full_attention_mask = torch.cat([
        attention_mask,
        torch.ones_like(completion_tokens).unsqueeze(0)
    ], dim=-1).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(full_sequence, attention_mask=full_attention_mask)
        logits = outputs.logits
    
    completion_logits = logits[0, input_ids.shape[1]-1:-1, :]
    log_probs = torch.log_softmax(completion_logits, dim=-1)
    
    token_log_probs = log_probs.gather(dim=-1, index=completion_tokens.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.sum()



results = []

for model_info in models_to_evaluate: 
    model_name = model_info['name']
    model_path = model_info['path']
    
    print(f"\n--- Evaluating model: {model_name} ---")
    
    
    if model_path == SFT_MODEL_PATH:
        model = ref_model
        tokenizer = ref_tokenizer
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"  ERROR: Could not load model from {model_path}. Skipping. Error: {e}")
            continue

    
    generate_kwargs = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.2,
        "num_return_sequences": NUM_SAMPLES_PER_PROMPT,
        "pad_token_id": tokenizer.eos_token_id, 
        "output_scores": True,
        "return_dict_in_generate": True
    }

    prompt_rewards = []
    prompt_kls = []

    for prompt_text in tqdm(prompts, desc=f"Evaluating {model_name}"):
        
        tokenized_prompt = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=tokenized_prompt.input_ids,
                attention_mask=tokenized_prompt.attention_mask,
                **generate_kwargs
            )

        full_texts_for_sentiment = []
        completion_token_objects_for_kl = []

        for i in range(NUM_SAMPLES_PER_PROMPT):
            generated_sequence = outputs.sequences[i]
            
            
            full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            full_texts_for_sentiment.append(full_text)
            
            
            completion_tokens = generated_sequence[tokenized_prompt.input_ids.shape[1]:]
            completion_token_objects_for_kl.append(completion_tokens)


        
        all_sentiment_scores = sentiment_classifier(full_texts_for_sentiment)
        completion_rewards = [get_positive_prob(scores) for scores in all_sentiment_scores]

        
        
        completion_kls = []
        for i, completion_tokens in enumerate(completion_token_objects_for_kl):
            if model_path == SFT_MODEL_PATH:
                kl_div_item = 0.0
            else:
                
                completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True)

                
                ref_tokenized_prompt = ref_tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
                ref_completion_tokens = ref_tokenizer(completion_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(DEVICE)

                
                current_tokenized_prompt = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
                current_completion_tokens = tokenizer(completion_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(DEVICE)

                sum_log_probs_theta = get_completion_log_probs(
                    model,
                    current_tokenized_prompt.input_ids,
                    current_tokenized_prompt.attention_mask,
                    current_completion_tokens
                )

                
                sum_log_probs_ref = get_completion_log_probs(
                    ref_model, 
                    ref_tokenized_prompt.input_ids,
                    ref_tokenized_prompt.attention_mask,
                    ref_completion_tokens
                )

                
                kl_div = sum_log_probs_theta - sum_log_probs_ref
                kl_div_item = kl_div.item()

            completion_kls.append(kl_div_item)
            
        prompt_rewards.append(np.mean(completion_rewards))
        prompt_kls.append(np.mean(completion_kls))
        
    mean_reward = np.mean(prompt_rewards)
    mean_kl = np.mean(prompt_kls)
    
    print(f"Results for {model_name}:")
    print(f"  Mean Sentiment Reward: {mean_reward:.4f}")
    print(f"  Mean KL Divergence:    {mean_kl:.4f}")
    
    results.append({
        "model": model_name,
        "reward": mean_reward,
        "kl": mean_kl,
        "beta_group": model_info['beta_group'],
        "step": model_info['step']
    })
    
    
    if model_path != SFT_MODEL_PATH:
        del model
        del tokenizer 
    gc.collect()
    torch.cuda.empty_cache()


del ref_model
del ref_tokenizer
gc.collect()
torch.cuda.empty_cache()



print("\nEvaluation complete. Generating plot...")

df = pd.DataFrame(results)

df = df.sort_values(by=["beta_group", "step"]) 

print(df)

plt.figure(figsize=(12, 8))


sft_data = df[df['beta_group'] == 'SFT']
if not sft_data.empty:
    plt.scatter(sft_data['kl'], sft_data['reward'], s=200, label='SFT (ref)', marker='*', c='black', zorder=10)
    plt.annotate('SFT (ref)', (sft_data['kl'].values[0], sft_data['reward'].values[0]), xytext=(5, 5), textcoords='offset points')


beta_groups = df[df['beta_group'] != 'SFT']['beta_group'].unique()
colors = plt.cm.jet(np.linspace(0, 1, len(beta_groups)))

for i, group_name in enumerate(beta_groups):
    group_data = df[df['beta_group'] == group_name]
    
    
    if not sft_data.empty:
        plot_data = pd.concat([sft_data, group_data]).sort_values(by='step')
    else:
        plot_data = group_data
    
    
    plt.plot(plot_data['kl'], plot_data['reward'], 
             marker='o', 
             linestyle='-', 
             label=group_name,
             color=colors[i],
             alpha=0.7)
    
    
    final_point = group_data[group_data['step'] == 99999]
    if not final_point.empty:
        plt.annotate(f"{group_name} (Final)", 
                     (final_point['kl'].values[0], final_point['reward'].values[0]),
                     xytext=(5, 5), 
                     textcoords='offset points',
                     color=colors[i],
                     weight='bold')

plt.xlabel("KL(DPO || SFT)", fontsize=14)
plt.ylabel("Sentiment Reward (Positive Prob.)", fontsize=14)
plt.title("DPO Reward vs. KL Trade-off (IMDb Sentiment)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best')
plt.tight_layout()

plt.savefig("dpo_reward_vs_kl_plot_with_checkpoints.png")
print("Plot saved as 'dpo_reward_vs_kl_plot_with_checkpoints.png'")

plt.show()