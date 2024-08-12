#!/bin/bash

# Define the models and corresponding host URLs
#models=("deepseek-v2:236b" "llama3_1_70b" "mistral-large")
#urls=("http://13.65.249.11:8885" "http://13.65.249.11:6662" "http://13.65.249.11:6665")
models=("qwen2_72b_instruct_q5_K_M" "llama3_70b_instruct_q5_K_M" "yi_34b_v1_5" "gemma2_27b" "command_r_plus" "deepseek_v2_236b")
urls=("http://13.65.249.11:8880" "http://13.65.249.11:6665" "http://13.65.249.11:8885" "http://13.65.249.11:8889" "http://13.65.249.11:8887" "http://13.65.249.11:6662")

# Loop through all models and send requests
for i in "${!models[@]}"; do
    model="${models[$i]}"
    url="${urls[$i]}"
    echo "Checking status for model $model at $url"

    # Send a curl request to the URL and print the result
    response=$(curl -s "$url")
    echo "Response from $model:"
    echo "$response"
    echo "----------------------------------------"
done
