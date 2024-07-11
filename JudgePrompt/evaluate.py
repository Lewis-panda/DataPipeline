from litellm import completion, embedding
from openai import AzureOpenAI
import json
import os
from tqdm import tqdm
import pandas as pd
import subprocess

client = AzureOpenAI(
    api_version="2024-02-15-preview",
    api_key="f8c3f26d14ff4876ab9a7d23251337d5",
    azure_endpoint="https://foxbrainopenaiapieastus.openai.azure.com",
)

def llm_call(messages, model="gpt-4o"):
    #messages = [{"role": "user", "content": messages}]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content

def get_embedding(input_text, model="text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=[input_text])
    return response.data[0].embedding

def generate_judge_prompt(original_text, rewritten_text):
    return f"""
    您是一位專業評審，負責評估給定的提示在以下五個類別中增強文本效果的有效性。您的任務是根據以下標準和步驟仔細評估給定的回答。

    評估標準（加分制，0-5分）：
    1. 簡化以適應不同閱讀水平：重寫的文本在保留原意的同時，如何簡化內容使其對不同教育水平的讀者易於理解？
    2. 增強細節：重寫的文本在豐富內容和提供更深入的見解方面，添加描述性細節的效果如何？
    3. 對比觀點：重寫的文本在引入對立意見或平衡觀點方面的效果如何？
    4. 加入同義詞：重寫的文本在保持原意不變的前提下，通過同義詞替換增加多樣性的效果如何？
    5. 主題重寫：重寫的文本在聚焦於特定主題或突出特定要點方面的效果如何？

    評估步驟：
    1. 仔細閱讀原始文本和重寫文本。
    2. 根據上述標準比較兩個文本。
    3. 為每個標準打分，範圍為0到5分。
    4. 為每個分數提供簡要說明。

    請按照以下JSON格式提供您的評估：
    {{
        "簡化以適應不同閱讀水平": score,
        "增強細節": score,
        "對比觀點": score,
        "加入同義詞": score,
        "主題重寫": score,
        "說明": "您的說明在此"
    }}

    示例：
    {{
        '問題': "重寫以下文本，重點關注五個關鍵類別：簡化以適應不同閱讀水平、增強細節、對比觀點、加入同義詞和主題重寫。",
        '回答': "原始文本: {original_text}\n重寫文本: {rewritten_text}",
        '評估': {{
            "簡化以適應不同閱讀水平": 4,
            "增強細節": 5,
            "對比觀點": 3,
            "加入同義詞": 4,
            "主題重寫": 5,
            "說明": "重寫的文本結構良好，提供了不同觀點的清晰對比，增強了讀者的理解。"
        }}
    }}

    現在，請評估以下內容：

    原始文本：
    {original_text}

    重寫文本：
    {rewritten_text}
    """

def evaluate_text(original_text, rewritten_text):
    judge_prompt = generate_judge_prompt(original_text, rewritten_text)
    messages = [{"role": "user", "content": judge_prompt}]
    evaluation_response = llm_call(messages)
    evaluation_result = json.loads(evaluation_response)
    return evaluation_result

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
        
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_existing_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def write_json(data, file_path):
    existing_data = read_existing_json(file_path)
    existing_data.append(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

        
def read_json(file_path):
    """讀取 JSON 文件並返回解析後的數據"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def process_news(news):
    """處理每個新聞項的函數，可以根據需要進行處理"""
    NewsID = news['NewsID']
    NewsType = news['NewsType']
    original_text = news['NewsContent']
    rewritten_text = news['RewriteContent']
    output_file_path = 'evaluation_result.json'
    evaluation_result = evaluate_text(original_text, rewritten_text)
    result = {
        "NewsID": NewsID,
        "NewsType": NewsType,
        "evaluation_result": evaluation_result
    }
#    print(evaluation_result)
    write_json(result, output_file_path)
    
def extract_to_excel(json_data, output_path):
    rows = []

    for news in json_data:
        row = {
            "NewsID": news["NewsID"],
            "NewsType": news["NewsType"],
            "簡化以適應不同閱讀水平": news["evaluation_result"]["簡化以適應不同閱讀水平"],
            "增強細節": news["evaluation_result"]["增強細節"],
            "對比觀點": news["evaluation_result"]["對比觀點"],
            "加入同義詞": news["evaluation_result"]["加入同義詞"],
            "主題重寫": news["evaluation_result"]["主題重寫"],
            "說明": news["evaluation_result"]["說明"]
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    
    
def main():
    # 讀取 JSON 文件
    file_path = 'data.json'
    news_data = read_json(file_path)
    
    # 迭代每個新聞項並進行處理
    print("步驟 1：生成 evaluation_result.json")
    for news in tqdm(news_data, desc="Processing news items"):
        process_news(news)
    
    # 步驟 2：讀取 evaluation_result.json
    print("步驟 2：讀取 evaluation_result.json")
    input_path = 'evaluation_result.json'
    json_data = read_json(input_path)
    
    # 步驟 3：將結果寫入 evaluation_result.xlsx
    print("步驟 3：將結果寫入 evaluation_result.xlsx")
    output_path = 'evaluation_result.xlsx'
    extract_to_excel(json_data, output_path)
    
    print("所有步驟完成")

if __name__ == "__main__":
    main()


#original_file_path = 'origin.txt'
#rewritten_file_path = 'rewrite.txt'
#output_file_path = 'evaluation_result.json'
#
#original_text = read_file(original_file_path)
#rewritten_text = read_file(rewritten_file_path)
#
#evaluation_result = evaluate_text(original_text, rewritten_text)
#write_json(evaluation_result, output_file_path)
#print(evaluation_result)





