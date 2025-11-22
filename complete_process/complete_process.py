import csv
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AutoModelForSequenceClassification
script_location = Path(__file__).absolute().parent
parent_folder = script_location.parent
sys.path.append(str(parent_folder))
from util.util import predict_relationship, generate_embedding, load_embeddings, load_embeddings_with_titles, \
    calculate_novelty_percentile, normalize_scores_minmax, normalize_scores_zscore
from getNoveltyScore.extract_entities import extract_entities
import numpy as np

def cut_similarities(similarities):
    similarity = min(1.0, similarities)  # Ensure that the similarity does not exceed 1
    similarity = max(0.0, similarity)  # Ensure that the similarity is not below 0
    return similarity

# 定义余弦相似度到新颖性得分的转换
def convert_similarity_to_novelty_score(cosine_similarity):
    return 1 - cosine_similarity


def find_top_three_unique_similar_articles(combined_embedding, embeddings, titles):
    # 计算相似度
    similarities = cosine_similarity([combined_embedding], embeddings)[0]

    # 对相似度进行排序，获取排序后的索引
    sorted_indices = np.argsort(similarities)[::-1]

    # 初始化一个列表来存储最相似的三篇文章的信息
    top_three_articles = []
    # 初始化一个集合来跟踪已选择的论文标题，确保不重复
    selected_titles = set()

    # 从排序后的索引中选择最相似的三篇不重复的论文
    for index in sorted_indices:
        current_title = titles[index]
        if current_title not in selected_titles and len(top_three_articles) < 3:
            # 添加当前论文的标题和相似度到列表
            top_three_articles.append((current_title, similarities[index]))
            # 标记当前论文标题为已选
            selected_titles.add(current_title)
        # 如果已经找到了三篇不重复的论文，就终止循环
        if len(top_three_articles) == 3:
            break
    return top_three_articles


def calculate_attention_weights(text, entities, tokenizer, model):
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 确保模型也在正确的设备上
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    attention = outputs.attentions
    entity_attention_weights = {}

    total_scores = 0  # 用于计算所有实体的总权重
    valid_entity_count = 0  # 有效实体的数量
    for entity in entities:
        entity_tokens = tokenizer.tokenize(entity)
        entity_token_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
        entity_token_ids = [tid for tid in entity_token_ids if tid < model.config.vocab_size]

        if not entity_tokens:
            print(f"No tokens found for entity '{entity}' in the text.")
            entity_attention_weights[entity] = 0
            continue

        score = 0
        token_attentions = []
        for layer_attention in attention:
            for head_attention in layer_attention:
                for token_id in entity_token_ids:
                    squeezed_ids = inputs['input_ids'].squeeze()
                    if squeezed_ids.dim() == 0:
                        squeezed_ids = squeezed_ids.unsqueeze(0)
                    token_attention = head_attention[:, :, squeezed_ids == token_id].mean()
                    if not np.isnan(token_attention.item()):
                        token_attentions.append(token_attention.item())
                        score += token_attention.item()

        entity_attention_weights[entity] = score
        total_scores += score
        valid_entity_count += 1

    # 检查并处理所有有效实体的权重为零的情况
    if total_scores == 0 and valid_entity_count > 0:
        equal_weight = 1 / valid_entity_count
        for entity in entities:
            entity_attention_weights[entity] = equal_weight
        print("All valid entities have zero total weight. Assigned equal weight to each entity.")
    elif valid_entity_count == 0:
        print("No valid entities found. Unable to assign any weights.")

    return entity_attention_weights




# 提取实体并按类型分组
def group_entities_by_type(entities):
    methods = {entity[2] for entity in entities if entity[0] == 'MethodName'}
    tasks = {entity[2] for entity in entities if entity[0] == 'TaskName'}
    return methods, tasks


#
#   params:
#   save_entity_score:如果确认存储所有实体得分，则进行以下处理,默认无以下处理步骤
#   save_batch_handle_data:批量计算论文得分结果并保存
def calculate_novelty_score(title, abstract, ner_tokenizer, ner_model, ner_model_attention,
                            relation_model, save_entity_score=False, save_batch_handle_data=False,
                            cal_single_paper=False, input_entities=False, entities=None):
    title = str(title)
    abstract = str(abstract)

    if not input_entities:
    # print("Entities in title:")
        title_entities = extract_entities(title, ner_tokenizer, ner_model)
        # for label, score, entity in title_entities:
            # print(f"Entity: {entity}, Score: {score:.3f}, Label: {label}")
        # print("\nEntities in abstract:")
        abstract_entities = extract_entities(abstract, ner_tokenizer, ner_model)
        # for label, score, entity in abstract_entities:
        #     print(f"Entity: {entity}, Score: {score:.3f}, Label: {label}")
        title_methods, title_tasks = group_entities_by_type(title_entities)
        abstract_methods, abstract_tasks = group_entities_by_type(abstract_entities)
    else:
        title_methods = entities['title_methods']
        title_tasks = entities['title_tasks']
        abstract_methods = entities['abstract_methods']
        abstract_tasks = entities['abstract_tasks']

    # 生成标题中的方法-任务组合，并包括标题文本
    title_pairs = [(method, task, title) for method in title_methods for task in title_tasks]
    # 生成摘要中的方法-任务组合，并包括摘要文本
    abstract_pairs = [(method, task, abstract) for method in abstract_methods for task in abstract_tasks]
    # 合并两组组合
    all_pairs = title_pairs + abstract_pairs
    # 调用predict_relationship函数
    all_used_for_relations = predict_relationship(all_pairs, ner_tokenizer, relation_model)

    # 打印所有预测为 'used-for' 关系的组合
    # print("\nAll 'used-for' relation combination:")
    # for method, task, score in all_used_for_relations:
    #     print(f"Method: {method}, Task: {task}, Score: {score:.3f}")

    # 检查是否有方法-任务对
    if not all_used_for_relations:
        print("No valid method-task pair was extracted")
        result = {
            'methods_with_scores': [],
            'tasks_with_scores': [],
            'all_used_for_relations': [],
            'pairs_with_scores': [],
            'total_novelty_score':  None,
            'final_top_three_articles': []
        }
        return result

    # 加载历史数据
    method_embeddings = load_embeddings('../data/method_data_50.h5')
    task_embeddings = load_embeddings('../data/task_data_50.h5')
    # method_task_comb_embeddings = load_embeddings('PaperNoveltyCal/data/method_task_comb_embeddings_50.h5')
    method_task_comb_embeddings, article_titles = load_embeddings_with_titles(
        '../data/method_task_comb_embeddings_50.h5')
    # 存储单个实体的词向量和新颖性得分
    method_embeddings_dict = {}
    task_embeddings_dict = {}
    entity_scores = {}

    # 计算实体词向量、新颖性得分和注意力权重
    for method in title_methods.union(abstract_methods):
        method_embedding = generate_embedding(method, ner_tokenizer, ner_model_attention)
        method_embeddings_dict[method] = method_embedding
        max_similarity = cut_similarities(np.max(cosine_similarity([method_embedding], method_embeddings)[0]))
        novelty_score = convert_similarity_to_novelty_score(max_similarity) * 100
        # print(f"Method '{method}' max cos similarity: {max_similarity} novelty score：{novelty_score}")
        entity_scores[method] = novelty_score

    for task in title_tasks.union(abstract_tasks):
        task_embedding = generate_embedding(task, ner_tokenizer, ner_model_attention)
        task_embeddings_dict[task] = task_embedding
        max_similarity = cut_similarities(np.max(cosine_similarity([task_embedding], task_embeddings)[0]))
        novelty_score = convert_similarity_to_novelty_score(max_similarity) * 100
        # print(f"Task '{task}' max cos similarity: {max_similarity} novelty score：{novelty_score}")
        entity_scores[task] = novelty_score

    # 用于存储所有最相似的三篇文章及其相似度的列表
    all_top_articles = []

    # 计算方法-任务组合的新颖性得分
    for method, task, _ in all_used_for_relations:
        method_embedding = method_embeddings_dict.get(method, np.zeros((1, ner_model_attention.config.hidden_size)))
        task_embedding = task_embeddings_dict.get(task, np.zeros((1, ner_model_attention.config.hidden_size)))
        combined_embedding = np.concatenate((method_embedding, task_embedding))
        max_similarity = cut_similarities(
            np.max(cosine_similarity([combined_embedding], method_task_comb_embeddings)[0]))
        novelty_score = convert_similarity_to_novelty_score(max_similarity) * 100
        # print(f"Combination '{method} - {task}' max cosine similarity: {max_similarity} novelty score：{novelty_score}")
        entity_scores[f"{method} - {task}"] = novelty_score

        # 对每个组合找出最相似的三篇论文
        top_three_articles = find_top_three_unique_similar_articles(combined_embedding, method_task_comb_embeddings,
                                                             article_titles)

        # 将这三篇论文及其相似度加入到列表中
        all_top_articles.extend(top_three_articles)
    # 选出相似度最高的三篇论文
    final_top_three_articles = all_top_articles[:3]

    # 打印或处理找到的最相似的三篇论文的标题
    print("Most similar 3 papers:")
    most_similar_3_paper = []
    for similar_paper_title, similarity in final_top_three_articles:
        similar_paper_title = similar_paper_title.decode('utf-8')
        most_similar_3_paper.append(similar_paper_title)
        print(f"Title: {similar_paper_title}, Similarity: {similarity:.3f}")

    # 计算整体新颖性得分
    combined_text = title + " " + abstract
    all_methods = title_methods.union(abstract_methods)
    all_tasks = title_tasks.union(abstract_tasks)
    all_entities = all_methods.union(all_tasks)

    # 获取所有实体的注意力权重
    attention_weights = calculate_attention_weights(combined_text, all_entities, ner_tokenizer, ner_model_attention)

    # 只考虑已提取实体的总权重
    total_weight_of_extracted_entities = sum(attention_weights.values())

    if total_weight_of_extracted_entities == 0:
        print("Warning: Total weight of extracted entities is 0. Unable to calculate scaled attention weights.")
        scaled_attention_weights = {entity: 0 for entity in all_entities}
    else:
        # 按比例分配注意力权重给已提取的实体
        scaled_attention_weights = {entity: attention_weights[entity] / total_weight_of_extracted_entities for entity in
                                    all_entities}

    # 检查是否有nan值，并打印相关信息
    for entity, weight in scaled_attention_weights.items():
        if np.isnan(weight):
            print(f"Entity '{entity}' has a NaN attention weight.")

    # 按比例分配注意力权重给已提取的实体
    scaled_attention_weights = {entity: attention_weights[entity] / total_weight_of_extracted_entities for entity in
                                all_entities}

    # 单独的方法和任务的新颖性得分
    print("Methods and their novelty scores along with attention weights:")
    for method in all_methods:
        attention_weight = scaled_attention_weights.get(method, 0)
        original_novelty_score = entity_scores.get(method, 0)
        print(
            f"Method: {method}, Attention Weight: {attention_weight}, Original Novelty Score: {original_novelty_score}")

    print("Tasks and their novelty scores along with attention weights:")
    for task in all_tasks:
        attention_weight = scaled_attention_weights.get(task, 0)
        original_novelty_score = entity_scores.get(task, 0)
        print(f"Task: {task}, Attention Weight: {attention_weight}, Original Novelty Score: {original_novelty_score}")

    # 单独的方法和任务的新颖性得分
    method_novelty_scores = {method: scaled_attention_weights[method] * entity_scores.get(method, 0) for method in
                             all_methods}
    task_novelty_scores = {task: scaled_attention_weights[task] * entity_scores.get(task, 0) for task in
                           all_tasks}

    # 计算方法-任务组合的新颖性得分
    pair_novelty_scores = {}
    for method, task, _ in all_used_for_relations:
        # 使用加权平均计算组合的新颖性得分
        method_weight = scaled_attention_weights.get(method, 0)
        task_weight = scaled_attention_weights.get(task, 0)
        combined_novelty_score = (method_weight * entity_scores.get(method, 0) + task_weight * entity_scores.get(task,
                                                                                                                 0)
                                  ) / (method_weight + task_weight)
        pair_novelty_scores[f"{method} - {task}"] = combined_novelty_score if combined_novelty_score != float(
            'nan') else 0

    # 输出单独的方法和任务的新颖性得分
    print("Method novelty score:")
    for method, score in method_novelty_scores.items():
        print(f"{method}: {score}")

    print("Task novelty score:")
    for task, score in task_novelty_scores.items():
        print(f"{task}: {score}")

    # 输出方法-任务组合的新颖性得分
    print("Method-Task combination novelty score:")
    for pair, score in pair_novelty_scores.items():
        print(f"{pair}: {score}")


    # 此部分代码用于为归一化计算数据做铺垫，用一部分数据组成过去已有论文提取的实体得分
    # 存储过往计算过论文的所有得分数据
    cumulative_method_score = sum(method_novelty_scores.values())  # 方法得分累积
    cumulative_task_score = sum(task_novelty_scores.values())  # 任务得分累积
    cumulative_pair_score = sum(pair_novelty_scores.values())  # 方法-任务组合得分累积

    # 如果传入参数save_entity_score=True确认存储所有实体得分，则进行以下处理,默认无以下处理步骤
    if save_entity_score:
        past_socre_data_file = "past_socre_data_file.csv"
        # 检查文件是否已存在且不为空，如果不存在或为空，则写入列标题
        if not os.path.exists(past_socre_data_file) or os.stat(past_socre_data_file).st_size == 0:
            with open(past_socre_data_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["CumulativeMethodScore", "CumulativeTaskScore", "CumulativeMethodTaskPairScore"])

        with open(past_socre_data_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([cumulative_method_score, cumulative_task_score, cumulative_pair_score])

    # 改为z_score归一化计算
    print("cumulative_method_score:", cumulative_method_score)
    print("cumulative_task_score:", cumulative_task_score)
    print("cumulative_method_task_pair_score:", cumulative_pair_score)

    paper_scores = {
        'CumulativeMethodScore': cumulative_method_score,
        'CumulativeTaskScore': cumulative_task_score,
        'CumulativeMethodTaskPairScore': cumulative_pair_score
    }
    final_score = normalize_scores_minmax(paper_scores)
    print("final_score:", final_score)

    # # 计算整体新颖性得分
    # total_novelty_score = sum(pair_novelty_scores.values()) + sum(method_novelty_scores.values()) + sum(
    #     task_novelty_scores.values())
    # print("Original score:", total_novelty_score)
    # # 转换为百分比，可以更清晰的理解排名
    # novelty_percentile = calculate_novelty_percentile(total_novelty_score, "../data/initial_data.csv")
    #
    # print(f"Final novelty score: {novelty_percentile}")

    # 以下代码用于把结果保存到csv文件,此部分用于存储所有论文的计算

    # 准备要保存的数据
    if save_batch_handle_data:
        methods_with_scores = [f"{method} - {score:.2f}" for method, score in method_novelty_scores.items()]
        tasks_with_scores = [f"{task} - {score:.2f}" for task, score in task_novelty_scores.items()]
        pairs_with_scores = [f"{pair} - {score:.2f}" for pair, score in pair_novelty_scores.items()]

        # result = {
        #     'methods_with_scores': [f"{method} - {score:.2f}" for method, score in method_novelty_scores.items()] if method_novelty_scores else [],
        #     'tasks_with_scores': [f"{task} - {score:.2f}" for task, score in task_novelty_scores.items()] if task_novelty_scores else [],
        #     'all_used_for_relations': all_used_for_relations if all_used_for_relations else [],
        #     'pairs_with_scores': [f"{pair} - {score:.2f}" for pair, score in pair_novelty_scores.items()] if pair_novelty_scores else [],
        #     'total_novelty_score': final_score if final_score is not None else 0,
        #     'final_top_three_articles': most_similar_3_paper if most_similar_3_paper else []
        # }
        # return result

        data_to_save = {
            "Title": title,
            "Abstract": abstract,
            "Methods": "; ".join(methods_with_scores),
            "Tasks": "; ".join(tasks_with_scores),
            "Method-Task Pairs": "; ".join(pairs_with_scores),
            "Total Novelty Score": final_score
        }

        # 如果只是正常调用API接口，则在此处直接返回数据给Java后端
        if cal_single_paper:
            return data_to_save

        # 保存数据到CSV文件
        # 每次新计算时，需要删除已有文件，避免在'a'模式下数据直接追加
        with open('new_extracted_data_minmax.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([data_to_save['Title'], data_to_save['Abstract'],
                             data_to_save['Methods'], data_to_save['Tasks'],
                             data_to_save['Method-Task Pairs'], data_to_save['Total Novelty Score']])


# 实体识别和关系提取模型初始化
ner_model_name = "sohamtiwari3120/scideberta-cs-tdm-pretrained-finetuned-ner"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
ner_model_attention = AutoModel.from_pretrained(ner_model_name, output_attentions=True)
relation_model = AutoModelForSequenceClassification.from_pretrained("../model/IdentifyScientificEntityRelation")

# 输入论文标题和摘要
paper_title = "HeroNet: A Hybrid Retrieval-Generation Network for Conversational Bots"
paper_abstract = '''Using natural language, Conversational Bot offers unprecedented ways to many
challenges in areas such as information searching, item recommendation, and
question answering. Existing bots are usually developed through retrieval-based
or generative-based approaches, yet both of them have their own advantages and
disadvantages. To assemble this two approaches, we propose a hybrid
retrieval-generation network (HeroNet) with the three-fold ideas: 1). To
produce high-quality sentence representations, HeroNet performs multi-task
learning on two subtasks: Similar Queries Discovery and Query-Response
Matching. Specifically, the retrieval performance is improved while the model
size is reduced by training two lightweight, task-specific adapter modules that
share only one underlying T5-Encoder model. 2). By introducing adversarial
training, HeroNet is able to solve both retrieval\&generation tasks
simultaneously while maximizing performance of each other. 3). The retrieval
results are used as prior knowledge to improve the generation performance while
the generative result are scored by the discriminator and their scores are
integrated into the generator's cross-entropy loss function. The experimental
results on a open dataset demonstrate the effectiveness of the HeroNet and our
code is available at https://github.com/TempHero/HeroNet.git
'''
# 示例通用使用
novelty_data = calculate_novelty_score(paper_title, paper_abstract, ner_tokenizer, ner_model, ner_model_attention, relation_model)

# 示例主动传入方法，任务实体
entities = {
    'title_methods': set({}),
    'title_tasks': set({}),
    'abstract_methods': set({'formal concept analysis', 'meta-learning algorithm'}),
    'abstract_tasks': set({'categorize objects', 'classification and outlier detection'})
}

# novelty_data = calculate_novelty_score(paper_title, paper_abstract, ner_tokenizer, ner_model,
#                                        ner_model_attention, relation_model, input_entities=True,
#                                        entities=entities)


# # 打印NER模型的配置
# print("NER Model Configuration:")
# print(ner_model.config)
#
# # 打印带注意力机制的NER模型的配置
# print("\nNER Model with Attention Configuration:")
# print(ner_model_attention.config)
#
# # 打印关系提取模型的配置
# print("\nRelation Model Configuration:")
# print(relation_model.config)
#
# # 详细的参数信息，可以使用下面的代码
# # 打印NER模型的参数
# print("\nNER Model Parameters:")
# for param_tensor in ner_model.state_dict():
#     print(param_tensor, "\t", ner_model.state_dict()[param_tensor].size())
#
# # 打印带注意力机制的NER模型的参数
# print("\nNER Model with Attention Parameters:")
# for param_tensor in ner_model_attention.state_dict():
#     print(param_tensor, "\t", ner_model_attention.state_dict()[param_tensor].size())
#
# # 打印关系提取模型的参数
# print("\nRelation Model Parameters:")
# for param_tensor in relation_model.state_dict():
#     print(param_tensor, "\t", relation_model.state_dict()[param_tensor].size())
