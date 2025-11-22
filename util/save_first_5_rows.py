import csv

# 原始CSV文件路径
input_csv_path = '../complete_process/new_papers.csv'
# 要保存的新CSV文件路径
output_csv_path = '../complete_process/new_papers_5rows.csv'

# 打开原始CSV文件读取内容
with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    # 打开目标CSV文件准备写入
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        # 读取并写入头五行
        for i in range(100):  # 限定循环5次
            row = next(reader, None)  # 使用next读取每一行
            if row is not None:
                writer.writerow(row)  # 写入读取的行
            else:
                break  # 如果文件行数不足5行，则提前结束循环

print(f"前五行已保存到{output_csv_path}")
