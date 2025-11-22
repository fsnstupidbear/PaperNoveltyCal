import pandas as pd
import numpy as np

def sigmoid(x):
    """计算Sigmoid函数的值"""
    return 1 / (1 + np.exp(-x))

def normalize_scores(new_scores, csv_path = 'past_socre_data_file.csv'):
    """
    根据历史数据归一化新论文的得分。

    参数:
    - new_scores: 包含新论文得分的字典，例如：
                  {'CumulativeMethodScore': 120,
                   'CumulativeTaskScore': 80,
                   'CumulativeMethodTaskPairScore': 100}
    - csv_path: 历史数据CSV文件的路径。

    返回:
    - 归一化后的得分字典。
    """
    # 读取历史数据
    df = pd.read_csv(csv_path)

    # 计算每列的平均值和标准差
    means = df.mean()
    stds = df.std()

    # 归一化新论文得分
    normalized_scores = {}
    for score_type, score_value in new_scores.items():
        if score_type in means and score_type in stds:
            normalized_score = (score_value - means[score_type]) / stds[score_type]
            normalized_score = sigmoid(normalized_score)
            normalized_scores[score_type] = normalized_score
        else:
            print(f"Warning: '{score_type}' not found in historical data. Skipping normalization.")
    return normalized_scores

# 使用示例
# new_paper_scores = {
#     'CumulativeMethodScore': 0.0000042924426435602,
#     'CumulativeTaskScore': 0,
#     'CumulativeMethodTaskPairScore': 0.0000042924426435602
# }
# # 替换为你的文件路径
# normalized_scores = normalize_scores(new_paper_scores)
# print(normalized_scores)