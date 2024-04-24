import os
import shutil

import zipfile
from pathlib import Path


# unzip zip file to a directory
def unzip_file(zip_file=Path("Data/celeba_img/archive.zip"), output_dir=""):
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(output_dir)


from utils_os_path import DataFoldersPartition, DataFilesPath

data_folders_partition = DataFoldersPartition()
data_files_path = DataFilesPath()


def move_image(file_name):

    folder_file = os.path.join(data_folders_partition.folder_source, file_name)

    if data_files_path.partition_dataframe.at[file_name, "partition"] == 0:
        shutil.move(
            folder_file, os.path.join(data_folders_partition.folder_train_m, file_name)
        )

    elif data_files_path.partition_dataframe.at[file_name, "partition"] == 1:
        shutil.move(
            folder_file, os.path.join(data_folders_partition.folder_eval_m, file_name)
        )

    elif data_files_path.partition_dataframe.at[file_name, "partition"] == 2:
        shutil.move(
            folder_file, os.path.join(data_folders_partition.folder_test_m, file_name)
        )

    else:
        print("error")


#move all image to training folder because GANs are generative models and we don't need to split the data into training and testing sets 
def move_images_gans_trainning(file_name):

    folder_file = os.path.join(data_folders_partition.folder_source, file_name)

    shutil.move(
        folder_file, os.path.join(data_folders_partition.folder_train_m, file_name)
    )



def cosumer(queue, move_image=move_images_gans_trainning):
    # print(str(multiprocessing.current_process()),'cosumer started')
    while True:
        item = queue.get()
        # print(str(multiprocessing.current_process()),flush=True)
        # print(str(item))
        if item is None:
            break
        move_image(item)


def move_images_to_folders():

    import os
    from pathlib import Path

    import multiprocessing
    import psutil

    data_folders_partition = DataFoldersPartition()
    folder_source = data_folders_partition.folder_source

    folder_train = data_folders_partition.folder_train_m
    folder_eval = data_folders_partition.folder_eval_m
    folder_test = data_folders_partition.folder_test_m

    for folder in [folder_train, folder_eval, folder_test]:
        if not os.path.isdir(folder):
            Path(folder).mkdir(mode=0o007, parents=True, exist_ok=True)

    if not len(os.listdir(folder_source)) == 0:

        processes_count = len(psutil.Process().cpu_affinity())
        print("processes count used: ", processes_count, flush=True)
        queue = multiprocessing.Queue(maxsize=200)
        pool = multiprocessing.Pool(processes_count, cosumer, (queue,))

        with pool:
            for file_name in os.listdir(folder_source):
                if file_name.endswith(".jpg"):
                    queue.put(file_name, block=True, timeout=None)

            for _ in range(processes_count):
                queue.put(None)

            queue.close()
            queue.join_thread()

    print("number of training images: ", len(os.listdir(folder_train)))
    print("number of evaluations images: ", len(os.listdir(folder_eval)))
    print("number of testing images: ", len(os.listdir(folder_test)))

    print(
        "number of total images: ",
        len(os.listdir(folder_train))
        + len(os.listdir(folder_eval))
        + len(os.listdir(folder_test)),
    )


# uncomment to run from current file

if __name__ == "__main__":
    from timeit import default_timer as timer

    # use time to cronometer the execution time
    time_start = timer()

    move_images_to_folders()

    time_end = timer()

    print("time: ", (time_end - time_start) / 60)
