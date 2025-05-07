# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file

from torch import nn
import torch

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from tqdm import trange
from torch.utils.data import DataLoader, Dataset
import matplotlib
from Siren_Network import *
from PIL import Image


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels

def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img



if __name__ == '__main__':

    device = 'cuda:0'
    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass

    cameraman = ImageFitting(256)
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                      hidden_layers=3, outermost_linear=True, first_omega_0=80, hidden_omega_0=80.)
    img_siren.cuda()

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in trange(total_steps):
        model_output, coords = img_siren(model_input)
        # model_output = gradient(model_output, coords)
        # model_output = laplace(model_output, coords)
        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            # img_grad = gradient(model_output, coords)
            # img_laplacian = laplace(model_output, coords)

            plt.imshow(model_output.cpu().view(256, 256).detach().numpy(), cmap='grey')
            # plt.imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy())
            # plt.imshow(img_laplacian.cpu().view(256, 256).detach().numpy())
            plt.show()
            # plt.savefig(f"tmp/output_{step}.png")
            # plt.close()

        optim.zero_grad()
        loss.backward()
        optim.step()