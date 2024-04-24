import os
import torch


import torchvision
import numpy as np

from pathlib import Path

from PIL import Image
from IPython.display import Image as Image_IPython, display


# Suggested to use slerp instead of linear interp for GANs by https://github.com/soumith/ganhacks
# Spherical interpolation formula from https://en.wikipedia.org/wiki/Slerp
def spherical_interpolation(t, p0, p1):
    # t is the interpolation parameter in range [0,1]
    # p0 and p1 are the two latent vectors
    p0 = p0.detach().to("cpu")
    p1 = p1.detach().to("cpu")

    if t <= 0:
        return p0
    elif t >= 1:
        return p1
    elif np.allclose(p0, p1):
        return p0

    p0_norm = p0 / np.linalg.norm(p0)
    p1_norm = p1 / np.linalg.norm(p1)
    # Convert p0 and p1 to unit vectors and find the angle between them (omega)
    omega = np.arccos(np.dot(p0_norm, p1_norm))

    return (
        np.sin((1.0 - t) * omega) / np.sin(omega) * p0
        + np.sin(t * omega) / np.sin(omega) * p1
    )


def process_generated_image(generated_image_tensor):
    # Move the tensor from GPU to CPU, convert to numpy array, extract first batch
    generated_image = generated_image_tensor.detach().to("cpu")

    # Generator outputs pixel valeus in [-1,1] due to tanh activation on last layer
    # transform to [0,1] range for display: add (-1), divide by 2
    generated_image -= torch.min(generated_image)
    generated_image /= torch.max(generated_image)

    return generated_image


def create_gif(frame_folder: Path, gif_path):
    frames = [
        Image.open(frame_folder / image)
        for image in os.listdir(frame_folder)
        if image.endswith(".png")
    ]

    """ 
    frames =[]
    for image in os.listdir(frame_folder):
        # check if image is a png file        
        if image.endswith('.png'):
            frames.append(Image.open(frame_folder / image))
    """
    frame_one = frames[0]

    frame_one.save(
        gif_path,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=5000,
        loop=0,
    )

    print(f"\nGIF saved to {gif_path} ")

    return gif_path


def display_gif(gif_path: Path):
    with open(gif_path, "rb") as f:
        print(f"\n ------------- GIF interpolated------------ ")
        display(Image_IPython(data=f.read(), format="gif", width=900, height=300))


# interpolate images for GANs not ConditionalGAN or ConditionalVAE
def interpolate_images(
    folder_root: Path, DEVICE, generator, number_of_gifs=3, interpolation_resolution=200
):
    """
    prints/shows a stack images
    Args:
        interpolation_resolution : path to the folder containing the images
        number_of_gifs : number of gifs to stack
        DEVICE : device to run on
    Returns:
        stacked image

    """
    for j in range(number_of_gifs):

        max = 1
        min = -1

        # generate two random Gaussian latent vectors
        z1 = torch.rand(100, dtype=torch.float32).to(DEVICE)
        z1 = (max - min) * z1 + min
        z2 = torch.rand(100, dtype=torch.float32).to(DEVICE)
        z2 = (max - min) * z2 + min

        # number of images between the vectors a and b, including a and b
        a = z1
        b = z2

        folder_path: Path = folder_root / f"interpolated_imgs_ver_{j}"
        folder_gifs = folder_path / "gifs"

        if not folder_gifs.is_dir():
            folder_gifs.mkdir(parents=True, exist_ok=True)

        for i in range(interpolation_resolution):
            # t in range [0,1] i.e. fraction of total interpolation
            t = i / (interpolation_resolution - 1)
            # generate intermediate interpolated vector
            current_latent_vector = spherical_interpolation(t, a, b)
            # convert to tensor for compatibility with image processing functions previously defined

            tns = current_latent_vector.to(DEVICE)

            current_latent_vector = torch.unsqueeze(tns, dim=0)
            # generate image from latent vector and process for saving
            generated_img = process_generated_image(
                generator(current_latent_vector)
            )  # .detach())

            # save intermediate interpolated image to folder
            torchvision.utils.save_image(
                generated_img, folder_path / f"interpolated_img_{i}.png", normalize=True
            )

        gif_path = folder_gifs / f"interpolated_imgs.gif"
        # print(f'\nCreating gif from {folder_path} to {gif_path}')
        # make a GIF of the interpolation
        create_gif(folder_path, gif_path)

        # display gif in notebook
        display_gif(gif_path)


from utils_data.utils_folders import remove_files_from_folder
from utils_data.utils_data_load import unnormalize_FashionMNIST

# interpolate images  for ConditionalVAE
def interpolate_conditioned_vae_images(
    folder_root: Path,
    DEVICE,
    latent_dimensions,
    fixed_one_hot_label_list,
    cvae_model,
    number_of_gifs=3,
    interpolation_resolution=2000,
    unormalize=False,
):
    """
    prints/shows a stack images
    Args:
        interpolation_resolution : path to the folder containing the images
        number_of_gifs : number of gifs to stack
        DEVICE : device to run on
    Returns:
        stacked image

    """
    cvae_model.eval()

    for j in range(number_of_gifs):
        # generate two random Gaussian latent vectors
        z1 = torch.rand(*(latent_dimensions), dtype=torch.float32, device=DEVICE)
        z2 = torch.rand(*(latent_dimensions), dtype=torch.float32, device=DEVICE)

        # number of images between the vectors a and b, including a and b
        a = z1
        b = z2

        folder_path: Path = folder_root / f"interpolated_imgs_ver_{j}"
        folder_gifs = folder_path / "gifs"

        if not folder_gifs.is_dir():
            folder_gifs.mkdir(parents=True, exist_ok=True)

        for i in range(interpolation_resolution):
            # t in range [0,1] i.e. fraction of total interpolation
            t = i / (interpolation_resolution - 1)

            # conver to 1D to be easy to take the dot product of the two tensors
            a = a.view(-1)
            b = b.view(-1)

            # generate intermediate interpolated vector
            current_latent_vector = spherical_interpolation(t, a, b)
            current_latent_vector = current_latent_vector.view(*(latent_dimensions)).to(
                DEVICE
            )
            # generate image from latent vector
            decoded_images = cvae_model.decoder(
                current_latent_vector, fixed_one_hot_label_list
            )

            generated_img = decoded_images.detach().to("cpu")
            # use torch to normalize
            """ 
            if unormalize:
                generated_img = unnormalize_FashionMNIST(generated_img)
            """

            # save intermediate interpolated image to folder
            torchvision.utils.save_image(
                generated_img,
                folder_path / f"interpolated_img_{i}.png",
                normalize=unormalize,
            )

        gif_path = folder_gifs / f"interpolated_imgs.gif"

        # make a GIF of the interpolation
        create_gif(folder_path, gif_path)

        remove_files_from_folder(folder_path)

        # display gif in notebook
        display_gif(gif_path)
