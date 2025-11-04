import os
import json
import wandb
from tqdm.auto import tqdm
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

GENERATIONS_DIR = "./generations"
NUM_PROMPTS = 1000
DEVICE = "cuda"
LEARNING_RATE = 2e-5
WANDB_NOTEBOOK_NAME = 'hw5/Direct-Preference-Optimization'
SFT_MODEL_PATH = "./fine-tuned-gpt2-large"

def load_preference_pairs() -> list[dict]:
    pairs = []
    for index in tqdm(range(NUM_PROMPTS), desc="Loading preference pairs"):
        filepath = os.path.join(GENERATIONS_DIR, f"completion_{index}.json")
        
        with open(filepath, "r") as f:
            content = json.load(f)
            pairs.extend(content['preference_pairs'])
            
    return pairs


def main():
    preference_list = load_preference_pairs()
    train_dataset = Dataset.from_list(preference_list)
    

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    betas = [1.0]

    for beta in betas:
        print(f"\n--- Starting DPO Training for beta={beta} ---")

        model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(DEVICE)

        run_name = f"dpo_beta_{beta}"
        output_dir = f"./dpo_output_beta_{beta}"
        local_save_path = f'./dpo-gpt2-large-beta-{beta}'

        dpo_training_args = DPOConfig(
            output_dir=output_dir,
            beta=beta,
            learning_rate=LEARNING_RATE,
            num_train_epochs=1.0,
            report_to='wandb',
            project=WANDB_NOTEBOOK_NAME,
            run_name=run_name,
            save_strategy='steps',
            save_steps=100,
        )
        
        trainer = DPOTrainer(
            model=model,
            ref_model=None, 
            args=dpo_training_args, 
            train_dataset=train_dataset
        )
        
        print(f"Starting training for {run_name}...")
        trainer.train(resume_from_checkpoint=True)

        print(f"Saving model to {local_save_path}...")
        trainer.save_model(local_save_path)
        print(f"Finished run for beta={beta}.")
        print("-------------------------------------------\n")
        
        wandb.finish()

    print("All DPO training runs complete.")

if __name__ == "__main__":
    main()