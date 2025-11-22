# 批量提取实体并计算新颖性得分

import pandas as pd

# 实体识别和关系提取模型初始化
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AutoModelForSequenceClassification

from complete_process import calculate_novelty_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
ner_model_name = "sohamtiwari3120/scideberta-cs-tdm-pretrained-finetuned-ner"
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(device)
ner_model_attention = AutoModel.from_pretrained(ner_model_name, output_attentions=True).to(device)
relation_model = AutoModelForSequenceClassification.from_pretrained("../model/IdentifyScientificEntityRelation").to(device)

# 定义处理论文标题和摘要的函数
def process_papers_from_csv(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 遍历每一行
    for index, row in df.iterrows():
        title = row.iloc[0]
        abstract = row.iloc[1]
        print("\none paper processing………………………………………………")
        # 使用calculate_novelty_score方法
        novelty_data = calculate_novelty_score(title, abstract, ner_tokenizer, ner_model,
                                               ner_model_attention, relation_model, save_batch_handle_data=True)

        # 检查是否有有效数据
        if novelty_data:
            print(f"处理完成: {title}")
        else:
            print(f"没有有效的方法-任务对于: {title}")
        print("one paper has been processed…………………………………………\n")

# 调用函数处理CSV文件
process_papers_from_csv('new_papers.csv')