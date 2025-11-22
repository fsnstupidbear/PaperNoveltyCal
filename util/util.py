import h5py
import torch
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

def sigmoid(x, beta=0.5):
    """计算Sigmoid函数的值，带有陡度调整"""
    return 1 / (1 + np.exp(-beta * x))


def normalize_scores_zscore(new_scores, csv_path = 'past_socre_data_file.csv'):
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

def normalize_scores_minmax(new_scores, csv_path='past_socre_data_file.csv'):
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

    # 计算每列的最小值和最大值
    mins = df.min()
    maxs = df.max()

    # 归一化新论文得分
    normalized_scores = {}
    for score_type, score_value in new_scores.items():
        if score_type in mins and score_type in maxs:
            if maxs[score_type] == mins[score_type]:  # 防止除数为零
                print(f"Warning: No range in data for '{score_type}'. Skipping normalization.")
                normalized_scores[score_type] = 0.0  # 或者选择一个合适的默认值
            else:
                normalized_score = (score_value - mins[score_type]) / (maxs[score_type] - mins[score_type])
                normalized_scores[score_type] = normalized_score
        else:
            print(f"Warning: '{score_type}' not found in historical data. Skipping normalization.")
    return normalized_scores

def combine_tokens(ner_results):
    combined_entities = []
    current_entity = []
    current_label = None
    current_entity_score = 0
    current_entity_token_num = 0

    for entity in ner_results:
        word = entity['word'].replace("Ġ", " ")
        label = entity['entity']
        score = entity['score']

        if label.startswith("B-"):
            if current_entity:
                average_score = current_entity_score / current_entity_token_num if current_entity_token_num else 0
                combined_entities.append((current_label, average_score, "".join(current_entity).strip()))
                current_entity = []
                current_entity_score = 0
                current_entity_token_num = 0
            current_entity = [word]
            current_label = label[2:]
            current_entity_score = score
            current_entity_token_num = 1
        elif label.startswith("I-") and current_entity:
            current_entity.append(word)
            current_entity_score += score
            current_entity_token_num += 1
        else:
            if current_entity:
                average_score = current_entity_score / current_entity_token_num if current_entity_token_num else 0
                combined_entities.append((current_label, average_score, "".join(current_entity).strip()))
                current_entity = []
                current_entity_score = 0
                current_entity_token_num = 0

    if current_entity:
        average_score = current_entity_score / current_entity_token_num if current_entity_token_num else 0
        combined_entities.append((current_label, average_score, "".join(current_entity).strip()))

    return combined_entities

# 添加mark_entities函数
def mark_entities(sentence, entity1, entity2):
    return sentence.replace(entity1, f"[ENTITY1] {entity1} [/ENTITY1]").replace(entity2, f"[ENTITY2] {entity2} [/ENTITY2]")


# predict_relationship函数
def predict_relationship(pairs, tokenizer, model):
    used_for_relations = []
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("\nAll \"used-for\" relation prediction result:")
    for method, task, source_text in pairs:
        # 标记化处理
        marked_sentence = mark_entities(source_text, method, task)
        # 对句子进行编码
        inputs = tokenizer(marked_sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("Input IDs device:", inputs['input_ids'].device)
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()  # 获取预测标签

            # 打印出每一对的预测结果和softmax概率值
            # print(f"Sentence: '{marked_sentence}'")
            # print(f"Predict: {'used-for' if prediction == 1 else 'not used-for'}, Probability: {probs[0][prediction].item():.3f}")

        # 如果预测为 'used-for' 关系，添加到结果列表中
        if prediction == 1:
            used_for_relations.append((method, task, probs[0][prediction].item()))

    return used_for_relations

# 函数：为给定文本生成词向量
def generate_embedding(text, tokenizer, model):
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(text, str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():  # 推理时不计算梯度
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    else:
        return np.zeros((1, model.config.hidden_size))  # 返回零向量作为占位符

# 加载h5文件的数据
def load_embeddings(file_path):
    with h5py.File(file_path, 'r') as h5f:
        # 直接读取数据，因为它们已经是一维数组
        embeddings = h5f['Vector'][:]
        # 将对象数组转换为常规的二维数组
        # 为此，我们假设所有嵌入向量具有相同的长度
        embeddings = np.vstack(embeddings)
    return embeddings

def load_embeddings_with_titles(file_path):
    with h5py.File(file_path, 'r') as h5f:
        embeddings = h5f['Vector'][:]
        titles = h5f['Titles'][:]
    return embeddings, titles


def calculate_novelty_percentile(input_score, csv_file_path):
    """
    Calculate the novelty percentile of an input score based on historical scores from a CSV file.

    Args:
    - input_score (float): The novelty score to be evaluated.
    - csv_file_path (str): The file path to the CSV containing historical novelty scores.

    Returns:
    - The percentile rank of the input score within the historical scores, with higher scores indicating higher novelty.
    """
    # Load historical scores from the CSV file, assuming the score is in the 6th column (5th index for zero-based indexing)
    df = pd.read_csv(csv_file_path, header=None)
    scores = pd.to_numeric(df.iloc[:, 5], errors='coerce').dropna()

    # Append the input score to the historical scores to find its percentile rank
    all_scores = pd.concat([scores, pd.Series([input_score])], ignore_index=True)

    # Calculate ranks in ascending order (lower score = higher rank number)
    ranks = all_scores.rank(method='min', ascending=False)
    input_score_rank = ranks.iloc[-1]

    # Convert rank to percentile (inverted)
    percentile = (1 - input_score_rank / len(all_scores)) * 100

    return percentile
