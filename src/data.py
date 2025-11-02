import os
import json
from tqdm.auto import tqdm

OUTPUT_DIR = "generations"
NUM_PROMPTS = 1000

def main():
    for index in tqdm(range(NUM_PROMPTS)): 

        filepath = os.path.join(OUTPUT_DIR, f"completion_{index}.json")
        
        with open(filepath, "r") as f:
                content = f.read()
                print(content['preference_pairs'])
                


            
if __name__ == "__main__":
    main()