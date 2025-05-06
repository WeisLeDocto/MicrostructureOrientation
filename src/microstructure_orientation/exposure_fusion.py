# coding: utf-8

import numpy as np
from pathlib import Path
from itertools import batched
from tqdm.auto import tqdm
import sys
import cv2
from re import fullmatch
from collections.abc import Sequence

# TODO: filter images before passing here


def exposure_fusion(images_paths: Sequence[Sequence[Path]],
                    dest_path: Path,
                    roi: tuple[slice, slice]) -> None:
    """"""

    # Create the folder containing the exposure fused images
    dest_path.mkdir(parents=False, exist_ok=True)

    # Iterate over batches of images from a same acquisition step
    for i, step in tqdm(enumerate(images_paths),
                        total=len(images_paths),
                        desc='Applying exposure fusion',
                        file=sys.stdout,
                        colour='green',
                        mininterval=0.01,
                        maxinterval=0.1):
        i: int
        step: Sequence[Path]

        # Apply the exposure fusion
        exp_fus = cv2.createMergeMertens(
          contrast_weight=0,
          saturation_weight=1,
          exposure_weight=1).process(tuple(np.load(img)[*roi] for img in step),
                                     tuple(int(fullmatch(r'\d+_(\d+)_\d+\.npy',
                                                         img.name).groups()[0])
                                           for img in step))

        # Save the output image at the indicated location
        name = fullmatch(r'(\d+)_\d+_\d+\.npy', step[0].name).groups()[0]
        np.save(dest_path / f'{i}_{name}.npy', exp_fus)
