from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM


training_dataset = load_dataset('stanfordnlp/imdb', split='train')

WANDB_NOTEBOOK_NAME = 'hw5/Direct-Preference-Optimization'

model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-large')


training_args = SFTConfig(
    output_dir="./my_training_output",
    save_strategy = 'steps',
    save_steps = 500,
    report_to='wandb',
    project=WANDB_NOTEBOOK_NAME,
    num_train_epochs=1.0,
)

# Default Learning Rate of 2e-05

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
)
print("Starting training...")
trainer.train(resume_from_checkpoint=True)
print("Training complete.")

local_save_path = './fine-tuned-gpt2-large'
print(f"Saving model to {local_save_path}...")
trainer.save_model(local_save_path)
print("Model saved.")
