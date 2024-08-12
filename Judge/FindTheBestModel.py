import pandas as pd

# Function to calculate total and average scores for each model
def calculate_total_and_average_scores(df):
    # Get unique models from the DataFrame
    models = df['Model'].unique()
    results = []

    # Iterate over each model to calculate scores
    for model in models:
        model_df = df[df['Model'] == model]
        # Calculate the total score by summing all specified score columns
        total_score = model_df[['清晰度_score', '保留原意_score', '描述性_score', '深度_score',
                                '觀點多樣性_score', '平衡性_score', '詞彙多樣性_score', '意思保留_score',
                                '聚焦性_score', '強調點_score']].sum(axis=1).sum()
        # Calculate the average score by summing the scores and taking the mean
        average_score = model_df[['清晰度_score', '保留原意_score', '描述性_score', '深度_score',
                                  '觀點多樣性_score', '平衡性_score', '詞彙多樣性_score', '意思保留_score',
                                  '聚焦性_score', '強調點_score']].sum(axis=1).mean()
        # Store the results for each model
        results.append({
            'Model': model,
            'Total Score': total_score,
            'Average Score': average_score
        })

    # Return the results as a DataFrame
    return pd.DataFrame(results)

def main():
    # Read the evaluation_result.csv file
    df = pd.read_csv('evaluation_result.csv', encoding='utf-8-sig')

    # Calculate total and average scores for each model
    results_df = calculate_total_and_average_scores(df)
    
    # Save the results to a CSV file
    results_df.to_csv('model_performance.csv', index=False, encoding='utf-8-sig')
    
    # Print the results
    print(results_df)

if __name__ == "__main__":
    main()
