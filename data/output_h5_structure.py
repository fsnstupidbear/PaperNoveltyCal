import h5py

# 替换为您的.h5文件路径
h5_file_path = 'method_task_comb_embeddings_50.h5'


def check_for_titles(file_path):
    with h5py.File(file_path, 'r') as file:
        def print_dataset_sample(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"\nDataset name: {name}")
                print(f"Shape (Dimensions): {obj.shape}")
                print(f"Data Type: {obj.dtype}")

                # 尝试打印数据集的前几个条目
                try:
                    sample_size = min(len(obj), 5)  # 取前5个条目或者数据集长度，取较小的一个
                    sample_data = obj[:sample_size]
                    print("Sample data:")
                    for item in sample_data:
                        print(item)
                except TypeError as e:
                    print("Could not read dataset sample, might be complex structure or large data.")
                except Exception as e:
                    print(f"Error reading dataset sample: {e}")

        # 使用visititems方法遍历文件中的所有项，并尝试打印每个数据集的样本
        file.visititems(print_dataset_sample)


check_for_titles(h5_file_path)
