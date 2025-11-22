import torch
from transformers import pipeline

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


def extract_entities(text, tokenizer, model):
    # # 模型名称
    # model_name = "sohamtiwari3120/scideberta-cs-tdm-pretrained-finetuned-ner"
    #
    # # 加载分词器和模型
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForTokenClassification.from_pretrained(model_name)

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建NER pipeline

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device)

    text = str(text)

    if not text.strip():
        return []
    # 使用pipeline进行NER
    ner_results = ner_pipeline(text)

    # 调用combine_tokens函数并返回结果
    return combine_tokens(ner_results)

# # 示例使用
# text = "example text here."
# combined_entities = extract_entities(text, tokenizer="" ,model="")
# for label, score, entity in combined_entities:
#     print(f"Combine Entity: {entity}, Score: {score:.3f}, Label: {label}")
