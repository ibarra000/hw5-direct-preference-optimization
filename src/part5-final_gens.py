import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset



DPO_FINAL_MODEL_PATHS = {
    "DPO_beta_0.05": './dpo-gpt2-large-beta-0.05',
    "DPO_beta_0.1": './dpo-gpt2-large-beta-0.1',
    "DPO_beta_1.0": './dpo-gpt2-large-beta-1.0'
}

SFT_MODEL_PATH = "./fine-tuned-gpt2-large"

NUM_SAMPLES_PER_PROMPT = 1
NUM_OF_TESTS = 3 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


ALL_MODELS = DPO_FINAL_MODEL_PATHS.copy()
ALL_MODELS["SFT"] = SFT_MODEL_PATH


print("Loading 'stanfordnlp/imdb' test dataset...")
try:
    testing_dataset = load_dataset('stanfordnlp/imdb', split='test')
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure you have an internet connection and the 'datasets' library is installed.")
    exit()


print(f"Generating {NUM_OF_TESTS} prompts from the dataset...")
prompts = []
for i in range(NUM_OF_TESTS):
    if i < len(testing_dataset):
        text = testing_dataset[i]['text'] 
        words = text.split()
        
        max_prefix = min(8, len(words) - 1)
        if max_prefix < 2:
             
            prompt_text = text
        else:
            prefix_length = random.randint(2, max_prefix)
            prompt_text = " ".join(words[:prefix_length])
        
        prompts.append(prompt_text)
    else:
        print(f"Warning: NUM_OF_TESTS ({NUM_OF_TESTS}) is larger than dataset size. Stopping prompt generation early.")
        break

print(f"Generated {len(prompts)} prompts. Example: '{prompts[0]}'")




overall_results = {}

for model_name, model_path in ALL_MODELS.items():
    print(f"\n--- Processing Model: {model_name} ---")
    print(f"Loading from path: {model_path}")
    
    results_for_this_model = []
    
    try:
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
        
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        print(f"Successfully loaded {model_name}.")
        
        
        for prompt in prompts:
            print(f"  Generating for prompt: '{prompt[:50]}...'")
            
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
            
            
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, 
                num_return_sequences=NUM_SAMPLES_PER_PROMPT,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
            
            
            for i in range(NUM_SAMPLES_PER_PROMPT):
                
                generation_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                result_entry = {
                    "prompt": prompt,
                    "generation": generation_text
                }
                results_for_this_model.append(result_entry)

        
        output_filename = f"results_{model_name}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results_for_this_model, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully saved results for {model_name} to {output_filename}")
        overall_results[model_name] = results_for_this_model

    except OSError as e:
        print(f"Error: Could not find model or tokenizer at path {model_path}.")
        print(f"Details: {e}")
    except RuntimeError as e:
        print(f"Error: Runtime error (e.g., CUDA out of memory) while processing {model_name}.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {model_name}: {e}")

print("\n--- All models processed. ---")

