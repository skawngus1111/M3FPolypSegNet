import os

import numpy as np
import matplotlib.pyplot as plt

from utils.get_functions import get_save_path

def plot_image_per_epoch(args, image, epoch, count):
    model_dirs = get_save_path(args)

    plt.imshow(np.transpose(image.cpu().detach().numpy() * 255, (1, 2, 0)).astype("uint8"), vmin=0, vmax=255)
    plt.axis('off'); plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    if not os.path.exists(os.path.join(model_dirs, 'image_per_epoch', 'EPOCH {}'.format(epoch))):
        os.makedirs(os.path.join(model_dirs, 'image_per_epoch', 'EPOCH {}'.format(epoch)))

    plt.savefig(os.path.join(model_dirs, 'image_per_epoch', 'EPOCH {}/example_{}.png'.format(epoch, count)), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()