from datasets import load_dataset
import pandas as pd
import numpy as np


DATASETS = ['ax', 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

# dataset = load_dataset("nyu-mll/glue", "ax")

# print(dataset['test'])

# df_train = pd.DataFrame(dataset['test'])

# print(set(df_train['label'].tolist()))


# for dataset_name in DATASETS:
# dataset_name = 'mnli'
# print(f"DATASET = {dataset_name}")

# dataset = load_dataset("nyu-mll/glue", dataset_name)

# for key in dataset.keys():
#     print(f"{key} - statistics")
#     print(dataset[key]['label'])

# dataset_name = 'stsb'
# print(f"DATASET = {dataset_name}")

# dataset = load_dataset("nyu-mll/glue", dataset_name)

# for key in dataset.keys():
#     print(f"{key} - statistics")
#     print(dataset[key]['label'])
    # avg = np.mean(dataset[key]['label'])
    # print(f"avg = {avg}")
    # if avg == -1.0:
    #     print(f"No true labels for {dataset_name}: {key}!")



for dataset_name in DATASETS:
    print(f"DATASET = {dataset_name}")

    dataset = load_dataset("nyu-mll/glue", dataset_name)

    for key in dataset.keys():
        print(f"{key} - statistics:")
        avg = np.mean(dataset[key]['label'])
        print(f"avg = {avg}")
        if avg == -1.0:
            print(f"No true labels for {dataset_name}: {key}!")


    # print(dataset)
    # print(np.mean(dataset['validation']['label']))
    # print("")


# dataset = load_dataset("nyu-mll/glue", "cola")
# print(dataset)
# print(np.mean(dataset['validation']['label']))


# print(df_train = pd.DataFrame( sentences['train'] )

# ds = load_dataset("nyu-mll/glue", "cola")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "mnli")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "mrpc")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "qnli")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "qqp")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "rte")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "sst2")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "stsb")

# print(ds)

# ds = load_dataset("nyu-mll/glue", "wnli")

# print(ds)

