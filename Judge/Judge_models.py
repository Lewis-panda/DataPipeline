import os
from litellm import completion, embedding
from openai import AzureOpenAI
import json
from tqdm import tqdm
import pandas as pd
from JudgePrompt import generate_judge_prompt
from JudgePrompt import json_to_dataframe
import time
from FindTheBestModel import calculate_total_and_average_scores

# Initialize the AzureOpenAI client
client = AzureOpenAI(
    api_version="2024-02-15-preview",
    api_key="f8c3f26d14ff4876ab9a7d23251337d5",
    azure_endpoint="https://foxbrainopenaiapieastus.openai.azure.com",
)

# Function to make a call to the LLM using AzureOpenAI
def llm_call(messages, model="gpt-4o"):
    response = client.chat.completions.create(model=model, temperature=0.0, top_p=1, messages=messages)
    return response.choices[0].message.content

# Function to read a file and return its content as a string
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to read a JSON file and return its content as a dictionary
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Function to read a JSONL file and return its content as a list of dictionaries
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Function to read a JSON file if it exists, otherwise return an empty list
def read_existing_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

# Function to write data to a JSON file
def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Function to evaluate rewritten text against the original text
def evaluate_text(original_text, rewritten_text, max_retries=3):
    judge_prompt = generate_judge_prompt(original_text, rewritten_text)
    messages = [{"role": "user", "content": judge_prompt}]
    for attempt in range(max_retries):
        try:
            evaluation_response = llm_call(messages)
            evaluation_result = json.loads(evaluation_response)
            return evaluation_result
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print("Retrying...")
            time.sleep(1)  # Wait for one second before retrying

    print("Exceeded maximum retries, could not get a valid JSON response.")
    return None

# Function to process each news article by evaluating the rewritten versions
def process_news(original_text, rewritten_texts):
    result = {}
    for model, rewritten_text in tqdm(rewritten_texts.items(), desc="Processing models"):
        evaluation_result = evaluate_text(original_text, rewritten_text)
        result[model] = evaluation_result
    return result

# Function to save progress to a JSON file
def save_progress(progress, progress_file):
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

# Function to load progress from a JSON file
def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Function to process each category of data and evaluate the performance of different models
def process_category(model_dirs, category_dirs, output_dir, progress_file='Run_Models/NewEvaluateResults/ds.pos'):
    progress = load_progress(progress_file)

    for category in tqdm(category_dirs, desc="Processing categories"):
        if category in progress and progress[category] == 'completed':
            continue

        # Load the original texts from the first50.jsonl file
        original_file_path = os.path.join('Run_Models/classify_data', category, 'first50.jsonl')
        original_texts = [item['text'] for item in read_jsonl(original_file_path)]

        rewritten_texts_per_category = {}
        for model, model_dir in model_dirs.items():
            rewrite_file_path = os.path.join(model_dir, category, 'first50', 'first50.json')
            rewritten_texts_per_category[model] = read_json(rewrite_file_path)

        start_index = progress.get(category, 0)
        results = []
        category_output_dir = os.path.join(output_dir, category)
        if not os.path.exists(category_output_dir):
            os.makedirs(category_output_dir)
        output_file_path = os.path.join(category_output_dir, 'result.json')
        
        # If result.json exists, load the existing results
        if os.path.exists(output_file_path):
            results = read_json(output_file_path)
        
        # Process each text in the category
        for i in tqdm(range(start_index, len(original_texts)), desc=f"Processing texts in {category}"):
            original_text = original_texts[i]
            rewritten_texts_per_doc = {model: rewritten_texts_per_category[model][i] for model in model_dirs}
            result = process_news(original_text, rewritten_texts_per_doc)
            results.append(result)

            write_json(results, output_file_path)
            # Update progress after each text
            progress[category] = i + 1
            save_progress(progress, progress_file)

        # Generate evaluation_result.csv and model_performance.csv for each category
        json_data = read_json(output_file_path)
        df = json_to_dataframe(json_data)
        csv_file_path = os.path.join(category_output_dir, 'evaluation_result.csv')
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

        results_df = calculate_total_and_average_scores(df)
        results_df.to_csv(os.path.join(category_output_dir, 'model_performance.csv'), index=False, encoding='utf-8-sig')

        # Mark category as completed
        progress[category] = 'completed'
        save_progress(progress, progress_file)

# Main function for testing
def main():
    model_dirs = {
        'qwen2': 'Run_Models/RewriteResults/qwen2',
        'llama3': 'Run_Models/RewriteResults/llama3',
        'yi': 'Run_Models/RewriteResults/yi',
        'gemma2': 'Run_Models/RewriteResults/gemma2',
        'commandR': 'Run_Models/RewriteResults/command',
        'llama3_1_70b': 'Run_Models/RewriteResults/llama3_1_70b',
        'deepseek': 'Run_Models/RewriteResults/deepseek',
        'mistral': 'Run_Models/RewriteResults/mistral'
    }

    category_dirs = [d for d in os.listdir('Run_Models/classify_data/') if os.path.isdir(os.path.join('Run_Models/classify_data/', d))]
    
    output_dir = 'Run_Models/NewEvaluateResults'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_category(model_dirs, category_dirs, output_dir)

if __name__ == "__main__":
    main()
