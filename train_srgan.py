# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""File description: Initialize the SRResNet model."""
import os
import shutil
import time
from enum import Enum

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

import config
from dataset import CUDAPrefetcher, TrainValidImageDataset
from image_quality_assessment import PSNR, SSIM
from model import Discriminator, Generator, ContentLoss

import matplotlib.pyplot as plt

save_image_dir = 'drive/MyDrive/Thesis/'

def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    # save image dir
    while True: 
        print('Enter drive directory to save image:')
        save_image_dir = input()
        if os.path.exists(save_image_dir):
            print(save_image_dir)
            break
        print('Directory not found')

    train_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    discriminator, generator = build_model()
    print("Build SRGAN model successfully.")

    content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained discriminator model weights...")
    if config.pretrained_d_model_path:
        # Load checkpoint model
        checkpoint = torch.load(config.pretrained_d_model_path, map_location=lambda storage, loc: storage)
        # Load model state dict. Extract the fitted model weights
        model_state_dict = discriminator.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        discriminator.load_state_dict(model_state_dict)
        print(f"Loaded `{config.pretrained_d_model_path}` pretrained discriminator model weights successfully.")
    else:
        print("Pretrained discriminator model weights not found.")

    print("Check whether to load pretrained generator model weights...")
    if config.pretrained_g_model_path:
        # Load checkpoint model
        checkpoint = torch.load(config.pretrained_g_model_path, map_location=lambda storage, loc: storage)
        # Load model state dict. Extract the fitted model weights
        model_state_dict = generator.state_dict()
        state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the model weights to the current model
        model_state_dict.update(state_dict)
        generator.load_state_dict(model_state_dict)
        print(f"Loaded `{config.pretrained_g_model_path}` pretrained generator model weights successfully.")
    else:
        print("Pretrained generator model weights not found.")

    print("Check whether to resume training discriminator...")
    if config.resume_d:
        # Load checkpoint model
        checkpoint = torch.load(config.resume_d, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = discriminator.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                          k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        discriminator.load_state_dict(model_state_dict)
        # Load the optimizer model
        d_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        d_scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Loaded `{config.resume_d}` resume discriminator model weights successfully. "
              f"Resume training from epoch {start_epoch + 1}.")
    else:
        print("Resume training discriminator model not found. Start training from scratch.")

    print("Check whether to resume training generator...")
    if config.resume_g:
        # Load checkpoint model
        checkpoint = torch.load(config.resume_g, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = generator.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if
                          k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        generator.load_state_dict(model_state_dict)
        # Load the optimizer model
        g_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        g_scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Loaded `{config.resume_g}` resume generator model weights successfully. "
              f"Resume training from epoch {start_epoch + 1}.")
    else:
        print("Resume training generator model not found. Start training from scratch.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join(config.exp_name, "samples")
    results_dir = os.path.join(config.exp_name, "results")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

    for epoch in range(start_epoch, config.epochs):
        train(discriminator,
              generator,
              train_prefetcher,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              save_image_dir,
              writer)
        # _, _ = validate(generator, valid_prefetcher, epoch, writer, psnr_model, ssim_model, "Valid")
        # psnr, ssim = validate(generator, test_prefetcher, epoch, writer, psnr_model, ssim_model, "Test")
        print("\n")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        # is_best = psnr > best_psnr and ssim > best_ssim
        # best_psnr = max(psnr, best_psnr)
        # best_ssim = max(ssim, best_ssim)
        torch.save({"epoch": epoch + 1,
                    # "best_psnr": best_psnr,
                    # "best_ssim": best_ssim,
                    "state_dict": discriminator.state_dict(),
                    "optimizer": d_optimizer.state_dict(),
                    "scheduler": d_scheduler.state_dict()},
                   os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"))
        torch.save({"epoch": epoch + 1,
                    # "best_psnr": best_psnr,
                    # "best_ssim": best_ssim,
                    "state_dict": generator.state_dict(),
                    "optimizer": g_optimizer.state_dict(),
                    "scheduler": g_scheduler.state_dict()},
                   os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"))
        # if is_best:
        #     shutil.copyfile(os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
        #                     os.path.join(results_dir, "d_best.pth.tar"))
        #     shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
        #                     os.path.join(results_dir, "g_best.pth.tar"))
        # if (epoch + 1) == config.epochs:
        #     shutil.copyfile(os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
        #                     os.path.join(results_dir, "d_last.pth.tar"))
        #     shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
        #                     os.path.join(results_dir, "g_last.pth.tar"))


def load_dataset() -> [CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.clean_image_dir, config.noisy_image_dir, config.image_size, config.upscale_factor, "Train")
    # valid_datasets = TrainValidImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "Valid")
    # test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    # valid_dataloader = DataLoader(valid_datasets,
    #                               batch_size=1,
    #                               shuffle=False,
    #                               num_workers=1,
    #                               pin_memory=True,
    #                               drop_last=False,
    #                               persistent_workers=True)
    # test_dataloader = DataLoader(test_datasets,
    #                              batch_size=1,
    #                              shuffle=False,
    #                              num_workers=1,
    #                              pin_memory=True,
    #                              drop_last=False,
    #                              persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    # valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    # test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    # return train_prefetcher, valid_prefetcher, test_prefetcher
    return train_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    discriminator = Discriminator()
    generator = Generator(config.no_res_block)

    # Transfer to CUDA
    discriminator = discriminator.to(device=config.device, memory_format=torch.channels_last)
    generator = generator.to(device=config.device, memory_format=torch.channels_last)

    return discriminator, generator


def define_loss() -> [ContentLoss, nn.BCEWithLogitsLoss]:
    content_criterion = ContentLoss(config.feature_model_extractor_node,
                                    config.feature_model_normalize_mean,
                                    config.feature_model_normalize_std)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    content_criterion = content_criterion.to(device=config.device, memory_format=torch.channels_last)
    adversarial_criterion = adversarial_criterion.to(device=config.device, memory_format=torch.channels_last)

    return content_criterion, adversarial_criterion


def define_optimizer(discriminator: nn.Module, generator: nn.Module) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(discriminator.parameters(), config.model_lr, config.model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.model_lr, config.model_betas)

    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.StepLR, lr_scheduler.StepLR]:
    d_scheduler = lr_scheduler.StepLR(d_optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)
    g_scheduler = lr_scheduler.StepLR(g_optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)

    return d_scheduler, g_scheduler


def train(discriminator: nn.Module,
          generator: nn.Module,
          train_prefetcher: CUDAPrefetcher,
          content_criterion: ContentLoss,
          adversarial_criterion: nn.BCEWithLogitsLoss,
          d_optimizer: optim.Adam,
          g_optimizer: optim.Adam,
          epoch: int,
          scaler: amp.GradScaler,
          save_image_dir: str,
          writer: SummaryWriter) -> None:

    """Training main program

    Args:
        discriminator (nn.Module): discriminator model in adversarial networks
        generator (nn.Module): generator model in adversarial networks
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        content_criterion (ContentLoss): Calculate the feature difference between real samples and fake samples by the feature extraction model
        adversarial_criterion (nn.BCEWithLogitsLoss): Calculate the semantic difference between real samples and fake samples by the discriminator model
        d_optimizer (optim.Adam): an optimizer for optimizing discriminator models in adversarial networks
        g_optimizer (optim.Adam): an optimizer for optimizing generator models in adversarial networks
        epoch (int): number of training epochs during training the adversarial network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              content_losses, adversarial_losses,
                              d_hr_probabilities, d_sr_probabilities],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the adversarial network model in training mode
    discriminator.train()
    generator.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    # show image tensor
    def show_tensor_images(image_tensor, name, num_images=16, size=(1, 128, 128)):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        image_tensor = (image_tensor + 1) / 2
        image_unflat = image_tensor.detach().cpu()
        image_grid = make_grid(image_unflat[:num_images], nrow=8)
        plt.figure(figsize=(15, 15))
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        # plt.show()
        plt.savefig(name+'.pdf')

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Used for the output of the discriminator binary classification, the input sample is from the dataset (real sample) and marked as 1, and the input sample from the generator (fake sample) is marked as 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        discriminator.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the real sample
        scaler.scale(d_loss_hr).backward()

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            # Use the generator model to generate fake samples
            sr = generator(lr)
            sr_output = discriminator(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
            # Calculate the total discriminator loss value
            d_loss = d_loss_sr + d_loss_hr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        scaler.scale(d_loss_sr).backward()
        # Improve the discriminator model's ability to classify real and fake samples
        scaler.step(d_optimizer)
        scaler.update()
        # Finish training the discriminator model

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        generator.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        with amp.autocast():
            content_loss = config.content_weight * content_criterion(sr, hr)
            adversarial_loss = config.adversarial_weight * adversarial_criterion(discriminator(sr), real_label)
            # Calculate the generator total loss value
            g_loss = content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        scaler.scale(g_loss).backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        scaler.step(g_optimizer)
        scaler.update()
        # Finish training the generator model

        # Calculate the score of the discriminator on real samples and fake samples, the score of real samples is close to 1, and the score of fake samples is close to 0
        d_hr_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

        # Statistical accuracy and loss value for terminal data output
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # convert image tensor to range [0, 1]
        def post_process(image_tensor):
            # for input tensor range [-1, 1]
            # return (image_tensor + 1.0) / 2.0

            # for input tensor range [0, 1]
            return image_tensor

        # Write the data during training to the training log file

        if batch_index % config.print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            save_image(make_grid(post_process(sr), nrow=8), f'{iters}@Generated.jpg')
            save_image(make_grid(post_process(lr), nrow=8), f'{iters}@Noisy.jpg')
            save_image(make_grid(post_process(hr), nrow=8), f'{iters}@Clean.jpg')
            # save to drive
            save_image(make_grid(post_process(sr), nrow=8), os.path.join(save_image_dir, f'{iters}@Generated.jpg'))
            save_image(make_grid(post_process(lr), nrow=8), os.path.join(save_image_dir, f'{iters}@Noisy.jpg'))
            save_image(make_grid(post_process(hr), nrow=8), os.path.join(save_image_dir, f'{iters}@Clean.jpg'))
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            # writer.add_image("Generated", make_grid(post_process(sr), nrow=8), iters)
            # writer.add_image("Noisy", make_grid(post_process(lr), nrow=8), iters)
            # writer.add_image("Clean", make_grid(post_process(hr), nrow=8), iters)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
