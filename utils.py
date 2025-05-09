import os
import random
import numpy as np
import torch

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def show_tensor_image(img_tensor, ax):
    img = img_tensor.clone().detach().cpu()
    img = img * 0.5 + 0.5  # convert back from [-1, 1] to [0, 1]
    img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
    img = img.numpy()
    ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax
