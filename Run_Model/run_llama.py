from glob import glob
import json
import re
from tqdm import tqdm
import os
from langchain_community.llms import Ollama

# Set up the model and corresponding host
model = "llama3:70b-instruct-q5_K_M"
model_name = "llama3"
host = "http://13.65.249.11:6665"

# Initialize prompt sets
prompt_sets = []
prompt_map = {}
with open("prompts/prompts.json", "r", encoding="utf-8") as r:
    prompts = json.load(r)
    for k, v in prompts.items():
        prompt_sets.append(k)  # Add the prompt name to the prompt set
        prompt_map[k] = v  # Map the prompt name to its content

# Initialize the model
def init_model(model, host):
    generation_params = {
        "num_predict": 2048,
        "top_k": 25,
        "top_p": 0.6,
        "repeat_penalty": 1.2,
        "temperature": 0.65,
    }
    try:
        llm = Ollama(
            model=model,
            base_url=host,
            keep_alive=True,
            **generation_params
        )
        return llm
    except Exception as e:
        print(f"Failed to initialize model {model} on {host}: {e}")
        return None

# Function to run the model with a given prompt and content
def run_model(llm, prompt, content):
    # Remove specific index markers from the content
    content = re.sub(r"idx:\s\d+,\s", "", content)
    # Insert the content into the prompt template
    prompt = prompt.replace("<INSERT_EXTRACT>", content)

    if len(prompt) == 0:
        raise ValueError("Prompt is empty.")

    if len(content) == 0:
        raise ValueError("Content is empty.")

    try:
        result = llm.invoke(prompt)
        return result
    except Exception as e:
        print(f"Error running model: {e}")
        return None

# Initialize the model
llm = init_model(model, host)
# Check if the model was initialized successfully
if llm is None:
    raise RuntimeError(f"Failed to initialize model {model} on {host}")

# Initialize dataset collection
datasets = []
dataset_map = {}

# Iterate through all category folders and process the first50.jsonl files within them
for category_folder in glob('classify_data/*'):
    category_name = os.path.basename(category_folder)
    for ds_path in glob(f'{category_folder}/first50.jsonl', recursive=True):
        name = ds_path.split("/")[-1].split(".")[0]  # Get the file name, e.g., 'first50'
        datasets.append(name)
        # Preload the dataset
        samples = []
        with open(ds_path, "r", encoding="utf-8") as r:
            for l in tqdm(r, desc=f"Pre-loading the dataset in {category_name}!"):  # l: line -> each line in r
                sample = json.loads(l.strip())
                samples.append(sample["text"])
            dataset_map[f"{category_name}/{name}"] = samples

# Define the progress file path
progress_file_dir = "records"
progress_file = f"{progress_file_dir}/{model_name}_ds.pos.json"

# Ensure the progress file directory exists
os.makedirs(progress_file_dir, exist_ok=True)

# If the progress file doesn't exist, create an empty dictionary and save it
if not os.path.exists(progress_file):
    ds_pos = {}
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(ds_pos, f, ensure_ascii=False, indent=4)
else:
    with open(progress_file, "r", encoding="utf-8") as r:
        ds_pos = json.load(r)

# Initialize progress tracking for each dataset if not already done
for k in dataset_map.keys():
    if k not in ds_pos:
        ds_pos[k] = 0

# Function to save progress
def save_progress(ds_pos, output_path=progress_file):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ds_pos, f, ensure_ascii=False, indent=4)

# Initialize a list to store all results
all_results = []

# Iterate through each sample in the dataset
for category_name, content_list in dataset_map.items():
    # Select the dataset and prompt
    prompt = prompt_map.get(category_name.split('/')[0], "")
    if not prompt:
        print(f"No prompt found for category {category_name}")
        continue

    # Set up the output directory and file path
    output_dir = f"RewriteResults/{model_name}/{category_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/first50.json"
    
    # If the output file already exists, read the existing results
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Process each content in the content list
    for idx, content in enumerate(tqdm(content_list, desc=f"({model_name}) Processing samples in {category_name}")):
        if idx < ds_pos.get(f"{category_name}/first50", 0):
            continue
        # Run the model with the prompt and content
        result = run_model(llm, prompt, content)
        if result is not None:
            all_results.append(result)
            ds_pos[f"{category_name}/first50"] = idx + 1
            save_progress(ds_pos)
            
            # Save the results to a file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_path}")

print("All steps completed")
