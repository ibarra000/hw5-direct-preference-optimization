from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import random
import json
import os
from tqdm.auto import tqdm

NUM_PROMPTS = 1000
NUM_SAMPLES_PER_PROMPT = 4
OUTPUT_DIR = "generations"
DEVICE = "cuda" # Assumes CUDA is available

def main():
    
    prefix_dataset = load_dataset('stanfordnlp/imdb', split='train')

    # Pointing back to your local custom model
    model_name = "./fine-tuned-gpt2-large" 
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if it doesn't exist (common with custom GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentiment_analysis = pipeline("text-classification", 
                              model="siebert/sentiment-roberta-large-english", 
                              device=DEVICE)

    generate_kwargs = {
        "max_new_tokens": 100,
        "do_sample": True,
        "temperature": 0.2,
        "num_return_sequences": NUM_SAMPLES_PER_PROMPT,
        "pad_token_id": tokenizer.eos_token_id
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for index in tqdm(range(NUM_PROMPTS)): 

        filepath = os.path.join(OUTPUT_DIR, f"completion_{index}.json")
        
        if os.path.exists(filepath):
            continue
    
        data = prefix_dataset[index]
        prefix_length = random.randint(2, 8)
        prompt = " ".join(data['text'].split()[:prefix_length])
    
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        outputs = model.generate(
            **inputs,
            **generate_kwargs
        )
            
        completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        results = sentiment_analysis(completions)
        
        sentiment_results = []

        for i, completion in enumerate(completions):
            result = results[i]
            
            score = result['score']
            
            if result['label'] == 'NEGATIVE':
                score = -score
            
            sentiment_results.append({"score": score, "completion": completion})
            
        sentiment_results.sort(key=lambda x: x["score"], reverse=True) 
        
        preference_pair_list = []
        for i in range(len(sentiment_results)):
            for j in range(i + 1, len(sentiment_results)):
                preference_pair = {
                    "prompt": prompt,
                    "chosen": sentiment_results[i]["completion"],
                    "rejected": sentiment_results[j]["completion"],
                }
                preference_pair_list.append(preference_pair)
              
        result_data = {
            "task_id": index,
            "original_text": data['text'],
            "original_label": data['label'],
            "completions": sentiment_results,
            "preference_pairs": preference_pair_list
        }
        
        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=4)

if __name__ == "__main__":
    main()