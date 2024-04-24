from pathlib import Path
import torch
import torchvision

import os
from collections import OrderedDict


class GANStatisticsSaver(object):
    def __init__(self):

        # the length of the bellow dictionaries is assumed to be the batch size

        """
        key:epoch nubmer
        value:list of losses
        """
        self.discriminator_loss_real = OrderedDict()

        """
        key:epoch nubmer
        value:list of losses
        """
        self.discriminator_loss_fake = OrderedDict()

        """
        key:epoch nubmer
        value:list of losses
        """
        self.generator_loss_fake = OrderedDict()

        """
        key:epoch nubmer
        value:number of batches
        """
        self.number_batches = OrderedDict()

        self.trainned_model_generator = None
        self.trainned_model_discriminator = None

        self.trainned_optimizer_generator = None
        self.trainned_optimizer_discriminator = None

        self.train_generator_loss_per_batch = []
        self.train_discriminator_loss_per_batch = []
        self.train_discriminator_real_acc_per_batch = []
        self.train_discriminator_fake_acc_per_batch = []
        self.images_from_noise_per_epoch = 0

        self.total_traininng_time = -1

    def set_trainned_model(self, model_generator, model_discriminator):
        self.trainned_model_generator = model_generator
        self.trainned_model_discriminator = model_discriminator

    def set_trainned_optimizer(self, optimizer_generator, optimizer_discriminator):
        self.trainned_optimizer_generator = optimizer_generator
        self.trainned_optimizer_discriminator = optimizer_discriminator

    def save_trainned_model(self, save_model_file_name):
        torch.save(
            {
                "model_generator": self.trainned_model_generator,
                "optimizer_generator": self.trainned_optimizer_generator,
                "loss": self,
            },
            save_model_file_name + "_Generator_whole" + ".pth",
        )

        torch.save(
            {
                "model_discriminator": self.trainned_model_discriminator,
                "optimizer_discriminator": self.trainned_optimizer_generator,
                "loss": self,
            },
            save_model_file_name + "_Discriminator_whole" + ".pth",
        )

    from torch import nn
    import torch.optim as optim

    def load_trainned_model(
        self, model: nn.Module, optimizer: optim.Optimizer, file_path: Path
    ):
        dict_values = torch.load(file_path)

        model.load_state_dict(dict_values["model_state_dict"])

        optimizer.load_state_dict(dict_values["optimizer_state_dict"])

        return model, optimizer, dict_values["epoch"]

    def load_trainned_models(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_generator: optim.Optimizer,
        optimizer_discriminator: optim.Optimizer,
        saved_model_file_name,
    ):

        file_name = "file_sngan_model"

        file_path = saved_model_file_name / Path(file_name + "_Generator_params.pth")
        generator, optimizer_generator, epoch = self.load_trainned_model(
            generator, optimizer_generator, file_path
        )

        file_path = saved_model_file_name / Path(
            file_name + "_Discriminator_params.pth"
        )
        discriminator, optimizer_discriminator, last_epoch = self.load_trainned_model(
            discriminator, optimizer_discriminator, file_path
        )

        return last_epoch

    def save_model_params(self, save_model_file_name: Path, last_epoch):
        # last_epoch = self.number_batches.keys()[-1]
        file_name = "file_sngan_model"

        torch.save(
            {
                "epoch": last_epoch,
                "model_state_dict": self.trainned_model_generator.state_dict(),
                "optimizer_state_dict": self.trainned_optimizer_generator.state_dict(),
            },
            save_model_file_name / Path(file_name + "_Generator_params.pth"),
        )

        torch.save(
            {
                "epoch": last_epoch,
                "model_state_dict": self.trainned_model_discriminator.state_dict(),
                "optimizer_state_dict": self.trainned_optimizer_discriminator.state_dict(),
            },
            save_model_file_name / Path(file_name + "_Discriminator_params.pth"),
        )

    def save_checkpoint(model, optimizer, epoch, model_loss, checkpoint_file_name):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": model_loss,
            },
            os.path.join(
                "SNGAN_training_output", "SNGAN_checkpoints", checkpoint_file_name
            ),
        )

    def save_image_generated(generated_image, epoch):
        print("saving images ")
        torchvision.utils.save_image(
            generated_image,
            os.path.join(
                "SNGAN_training_output",
                "images_generated",
                f"generated_image_{epoch}.png",
            ),
        )

    def save_train_generator_loss_per_batch(self, generator_loss):
        self.train_generator_loss_per_batch.append(generator_loss)

    def save_train_discriminator_loss_per_batch(self, discriminator_loss):
        self.train_discriminator_loss_per_batch.append(discriminator_loss)

    def save_train_discriminator_real_acc_per_batch(self, acc_real):
        self.train_discriminator_real_acc_per_batch.append(acc_real)

    def save_train_discriminator_fake_acc_per_batch(self, acc_fake):
        self.train_discriminator_fake_acc_per_batch.append(acc_fake)

    def save_images_normalized_to_folder(self, images_generated, folder_path, epoch):
        # self.images_from_noise_per_epoch = self.images_from_noise_per_epoch + 1
        torchvision.utils.save_image(
            images_generated,
            os.path.join(folder_path, f"generated_image_{epoch}.png"),
            normalize=True,
        )  # value_range=(-1,1) producing unclear images

    def save_images_to_folder(self, images_generated, folder_path, epoch):
        # self.images_from_noise_per_epoch = self.images_from_noise_per_epoch + 1
        torchvision.utils.save_image(
            images_generated,
            os.path.join(folder_path, f"generated_image_{epoch}.png"),
            normalize=False,
        )

    def save_total_training_time(self, total_time):
        self.total_traininng_time = total_time
