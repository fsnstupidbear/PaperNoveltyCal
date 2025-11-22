import pandas as pd

# 替换为你的CSV文件路径
csv_file_path = '../complete_process/extracted_data.csv'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 排除第6列中包含'nan'的行
# 注意：pandas在读取CSV时，会将字符串"nan"解析为浮点数NaN
# 因此，这里不是直接比较字符串'nan'，而是使用pd.isna()函数来检测NaN值
df_cleaned = df[~pd.isna(df.iloc[:, 5])]
df_cleaned_two_columns = df_cleaned.iloc[:, :2]
# 可以选择将清理后的数据保存到新的CSV文件
df_cleaned_two_columns.to_csv('cleaned_final_data.csv', index=False)
