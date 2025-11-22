import pandas as pd
from scipy import stats
import json


def extract_scores(file_path):
    df = pd.read_csv(file_path, header=None)
    scores_column = df.iloc[:, 5]
    method_scores = []
    task_scores = []
    pair_scores = []

    for score_str in scores_column:
        # 检查字符串是否包含 'nan'
        if 'nan' in score_str:
            continue  # 如果包含 'nan'，跳过这一行

        try:
            score_dict = json.loads(score_str.replace("'", '"'))
            method_scores.append(score_dict.get('CumulativeMethodScore', 0))
            task_scores.append(score_dict.get('CumulativeTaskScore', 0))
            pair_scores.append(score_dict.get('CumulativeMethodTaskPairScore', 0))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} - In string: {score_str}")

    return method_scores, task_scores, pair_scores


new_method_scores, new_task_scores, new_pair_scores = extract_scores('new_extracted_data.csv')
old_method_scores, old_task_scores, old_pair_scores = extract_scores('old_extracted_data.csv')

method_t_stat, method_p_value = stats.ttest_ind(new_method_scores, old_method_scores, equal_var=False)
task_t_stat, task_p_value = stats.ttest_ind(new_task_scores, old_task_scores, equal_var=False)
pair_t_stat, pair_p_value = stats.ttest_ind(new_pair_scores, old_pair_scores, equal_var=False)

print("Method Score t-test:", method_t_stat, "p-value:", method_p_value)
print("Task Score t-test:", task_t_stat, "p-value:", task_p_value)
print("Pair Score t-test:", pair_t_stat, "p-value:", pair_p_value)
