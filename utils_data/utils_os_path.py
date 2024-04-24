from dataclasses import dataclass
import os
import platform


os_name = platform.system()

os_type_slash = os.sep

os_unix_slash = {"single": "/", "double": r"//"}
os_windows_slash = {"single": "\\", "double": r"\\"}


def getOsSlash():
    if os_name == "Windows":
        return os_windows_slash
    elif os_name == "Linux":
        return os_unix_slash
    else:
        return None

os.getcwd()
folder_root = os.path.join("Data", "celeba_img")
folder_root = os.path.join(os.getcwd(), folder_root)

folder_root = os.path.normpath(folder_root)

_folder_source = os.path.join(folder_root, "img_align_celeba")
folder_source = os.path.join(_folder_source, "img_align_celeba")


folder_train = os.path.join(folder_root, "training")
folder_eval = os.path.join(folder_root, "validation")
folder_test = os.path.join(folder_root, "testing")


# list_attr_celeba.csv
PARTITION_PATH  = os.path.join(folder_root,  "list_eval_partition.csv")
ATTRIBUTES_PATH = os.path.join(folder_root, "list_attr_celeba.csv")
BBOX_PATH       = os.path.join(folder_root,       "list_bbox_celeba.csv")
LANDMARKS_ALLIGN_PATH = os.path.join(folder_root, "list_landmarks_align_celeba.csv")


folder_train_m = os.path.join(folder_train, "misc")
folder_eval_m = os.path.join(folder_eval, "misc")
folder_test_m = os.path.join(folder_test, "misc")


import pandas as pd

''' 
if not os.getcwd().endswith("utils_data"):
    os.chdir("utils_data")
'''

# list_eval_partition.csv
partition_dataframe = pd.read_csv(PARTITION_PATH, index_col="image_id")


@dataclass
class DataFoldersPartition:
    os_type_slash = os_type_slash

    folder_source = folder_source

    folder_train_m = folder_train_m
    folder_eval_m = folder_eval_m
    folder_test_m = folder_test_m


@dataclass
class DataFilesPath:
    partition_dataframe = partition_dataframe


@dataclass
class DataFolderTrain:
    folder_train = folder_train
    folder_test = folder_test

    labels_attributes_path = ATTRIBUTES_PATH