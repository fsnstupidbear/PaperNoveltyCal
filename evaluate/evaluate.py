import numpy as np
import pandas as pd
import json
def calculate_average_score(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)
    # 获取第六列的数据
    scores_column = df.iloc[:, 5]

    # 初始化得分总和
    total_method_score = 0
    total_task_score = 0
    total_pair_score = 0
    count = 0

    # 遍历每一行，解析JSON字符串，并累加得分
    for score_str in scores_column:
        # 尝试修正JSON字符串中的单引号问题
        fixed_json_str = score_str.replace("'", '"')

        try:
            # 解析JSON字符串
            score_dict = json.loads(fixed_json_str)
            # 检查是否包含nan值
            if any(np.isnan(value) if isinstance(value, float) else False for value in score_dict.values()):
                print(f"Skipping row with NaN values: {fixed_json_str}")
                continue

            total_method_score += score_dict.get('CumulativeMethodScore', 0)
            total_task_score += score_dict.get('CumulativeTaskScore', 0)
            total_pair_score += score_dict.get('CumulativeMethodTaskPairScore', 0)
            count += 1
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} - In string: {fixed_json_str}")

    # 计算平均得分
    if count > 0:
        avg_method_score = total_method_score / count
        avg_task_score = total_task_score / count
        avg_pair_score = total_pair_score / count
        return avg_method_score, avg_task_score, avg_pair_score
    else:
        return 0, 0, 0

# 计算每个文件的平均得分
avg_scores_new = calculate_average_score('new_extracted_data.csv')
avg_scores_old = calculate_average_score('old_extracted_data.csv')

print("Average Scores for new_extracted_data.csv:", avg_scores_new)
print("Average Scores for old_extracted_data.csv:", avg_scores_old)
