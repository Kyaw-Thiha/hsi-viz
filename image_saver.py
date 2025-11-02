from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from InquirerPy import inquirer


def get_save_choice(filetypes: List[str] = ["jpg", "png", "mat"]) -> str:
    # Building save choice
    save_choices: List[Dict[str, str]] = []
    for filetype in filetypes:
        save_obj = {"name": filetype.upper(), "value": filetype}
        save_choices.append(save_obj)
    save_choices.append({"name": "Don't Save", "value": ""})

    # Asking user input
    save_choice: str = inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message="Select which format you want to store your image in:",
        choices=save_choices,
        default="",
    ).execute()

    return save_choice


def change_image_shape(img: np.ndarray, message: str, transpose_map: Dict[str, tuple[int, int, int]]):
    # Dynamically build choice list
    choices = []
    for fmt, order in transpose_map.items():
        choices.append(
            {
                "name": f"({', '.join(fmt.upper())}) -> {img.shape}",
                "value": fmt,
            }
        )

    # Ask for user input
    format_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message=message,
        choices=choices,
        default="hwc",
    ).execute()

    # Transpose the image as wanted
    order = transpose_map[format_choice]
    if order is not None:
        img = img.transpose(order)

    return img
