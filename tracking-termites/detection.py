"""This module contains the termite detection functionalities."""

import cv2
import numpy as np


def show_frame(img):
    """Display a frame for template collection.

    Args:
        img (np.ndarray): array containing image values.
    Returns:
        None.

    """
    cv2.imshow('Initial frame', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def detect_sample_from_template(img, template, threshold):
    """Obtain a sample coordinates in a frame given a template.

    Args:
        img (np.ndarray): array containing grayscale image values.
        template (str): sample template path.
    Returns:
        point (tuple): tuple containg sample location coordinates (y,x).

    """
    template = cv2.imread(template, 0)

    results = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    matches = np.where(results >= threshold)
    for point in zip(*matches[::-1]):
        return point
