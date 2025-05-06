# coding: utf-8

from matplotlib import pyplot as plt, animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from re import fullmatch
import numpy as np
import os

NB_ANGLES = int(os.getenv("MICRO_ORIENT_NB_ANG", default="45"))


def sort_images(path: Path) -> int:
    """"""

    sec, = fullmatch(r'(\d+)_\d+\.npy', path.name).groups()
    return int(sec)


def make_plots(hdr_folder: Path,
               gabor_folder: Path,
               anim_folder: Path) -> None:
    """"""

    # Create the destination folder and parse the input folders
    anim_folder.mkdir(parents=False, exist_ok=True)
    images = tuple(sorted(gabor_folder.glob('*.npy'), key=sort_images))
    hdrs = tuple(sorted(hdr_folder.glob('*.npy'), key=sort_images))

    ang = np.linspace(0, 180, NB_ANGLES)

    # Create the first figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.axis('off')
    ax2.axis('off')

    # Display the dominant orientation in each pixel after Gabor filter
    img1 = ax1.imshow(ang[np.argmax(np.load(images[0]), axis=2)],
                      cmap='twilight', clim=(0, 180))
    divider_1 = make_axes_locatable(ax1)
    cax1 = divider_1.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(img1, cax=cax1, orientation='horizontal')

    # Display the HDR images
    img2 = ax2.imshow(np.load(hdrs[0]), cmap='grey', clim=(0, 1))
    divider_2 = make_axes_locatable(ax2)
    cax2 = divider_2.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(img2, cax=cax2, orientation='horizontal')

    def update(frame):
        """"""

        img1.set_array(ang[np.argmax(np.load(images[frame + 1]), axis=2)])
        img2.set_array(np.load(hdrs[frame + 1]))
        return img1, img2

    # Animate the figure over all the acquired time points
    ani = anim.FuncAnimation(fig=fig, func=update, frames=len(images) - 1,
                             interval=500, repeat=True, repeat_delay=2000)
    ani.save(anim_folder / "orientation_2.mkv", writer='ffmpeg', fps=2)
    plt.close()

    # Create the second figure showing the intensity of the Gabor response
    fig, ax = plt.subplots()
    img = plt.imshow(np.average(np.load(images[0]), axis=2),
                     cmap='plasma', clim=(0, 0.25))
    plt.colorbar()

    def update(frame):
        """"""

        img.set_array(np.average(np.load(images[frame + 1]), axis=2))
        img.set_clim(0, 0.25)
        return img

    # Animate the figure over all the acquired time points
    ani = anim.FuncAnimation(fig=fig, func=update, frames=len(images) - 1,
                             interval=500, repeat=False)
    ani.save(anim_folder / "intensity.gif", writer='imagemagick', fps=2)
    plt.close()
