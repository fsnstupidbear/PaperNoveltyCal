import pandas as pd

# 读取数据文件
df = pd.read_csv('../semantic_scholar_data.csv')

# 确保citationCount列是数值型的
df['citationCount'] = pd.to_numeric(df['citationCount'], errors='coerce')

# 过滤引用量大于等于50的条目
filtered_df = df[df['citationCount'] >= 50]

# 重置索引
filtered_df.reset_index(drop=True, inplace=True)

# 写入新的CSV文件
filtered_df.to_csv('filtered_data_50.csv', index=False)
