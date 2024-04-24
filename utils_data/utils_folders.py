#!/usr/bin/python
import os

from pathlib import Path


def remove_folder(folder):
    clean_folder(folder)

    if os.path.exists(folder):
        folder = Path(folder)
        folder.rmdir()


def clean_folder(folder):
    folder = Path(folder)

    if os.path.exists(folder):
        for item in folder.iterdir():
            if item.is_dir():
                remove_folder(item)
            else:
                item.unlink()

    else:
        print("Folder: " + folder.name + " does not exist")
        return


def remove_files_from_folder(folder_path):
    if os.path.exists(folder_path):
        for item in folder_path.iterdir():
            if not item.is_dir():
                item.unlink()


import cv2 as cv2
import numpy as np


# # read images from folder and stack them in the same image and save the final image
def get_one_stack_of_images(
    folder, image_name, save_dir, extension="png", axis=0, scale=1
):
    """
    stack images from a folder into a single image
    Args:
        folder     : path to the folder containing the images
        image_name : name of the image to be saved
        save_dir   : path to the directory where the stacked image will be saved
        extension  : extension of the images in the folder
        axis       : axis to stack the images
        scale      : scale of the stacked image
    """
    images = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            images.append(cv2.imread(os.path.join(folder, file)))
    stacked_image = np.concatenate(images, axis=axis)
    stacked_image = cv2.resize(stacked_image, (0, 0), fx=scale, fy=scale)

    file_path = os.path.join(save_dir, image_name)
    cv2.imwrite(file_path, stacked_image)

    return file_path


# # create a gif from a folder of images


# # read images from folder and stack them in the same image and save the final image
def stack_images_in_batches(
    folder,
    image_name,
    save_dir,
    number_of_images=5,
    extension="png",
    axis=0,
    scale=1,
):
    """
    stack images from a folder into a single image
    Args:
        folder (str): path to the folder containing the images
        image_name (str): name of the image to be saved
        save_dir (str): path to the directory where the stacked image will be saved
        extension (str): extension of the images in the folder
        axis (int): axis to stack the images
        scale (int): scale of the stacked image
    """

    images = []
    index_file = 1
    for file in os.listdir(folder):
        if file.endswith(extension):
            images.append(cv2.imread(os.path.join(folder, file)))

            if len(images) % number_of_images == 0:
                stacked_image = np.concatenate(images, axis=axis)
                stacked_image = cv2.resize(stacked_image, (0, 0), fx=scale, fy=scale)
                cv2.imwrite(
                    os.path.join(
                        save_dir,
                        str(len(images))
                        + "_"
                        + str(index_file)
                        + "_"
                        + "stack"
                        + "_"
                        + image_name,
                    ),
                    stacked_image,
                )

                images = []
                index_file = index_file + 1

    if images != []:
        stacked_image = np.concatenate(images, axis=axis)
        stacked_image = cv2.resize(stacked_image, (0, 0), fx=scale, fy=scale)

        cv2.imwrite(
            os.path.join(save_dir, str(len(images)) + "_" + "stack" + "_" + image_name),
            stacked_image,
        )


def write_to_disk_images_stacked(
    folder_source_imgs: Path,
    folder_output_batches_stacked: Path,
    folder_output_all_stacked: Path,
    num_epochs: int,
):
    """
    stack images from a folder into a single image
    Args:
        folder_source_imgs : path to the folder containing the images
        folder_output_batches_stacked : path to the folder containing the images batched and stacked
        folder_output_all_stacked : path to the folder containing all the images stacked
        num_epochs : identifier of the epoch to be used to name the stacked images

    """
    image_file_name = "images_generator_" + str(num_epochs) + "_epochs.png"

    stack_images_in_batches(
        folder_source_imgs, image_file_name, save_dir=folder_output_batches_stacked
    )

    resulting_images = get_one_stack_of_images(
        folder_source_imgs, image_file_name, save_dir=folder_output_all_stacked
    )

    return resulting_images
