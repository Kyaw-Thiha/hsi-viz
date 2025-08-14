from typing import List, Optional, Tuple
from scipy.io import loadmat
import os
import numpy as np
import tifffile

from InquirerPy import inquirer


FILE_PATH = "data"


def load_image(
    input_dir: str, select_multiple: bool = False
) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Interactively select one or more image files from a directory and load them.

    Supports `.mat`, `.npy`, and `.tif` files. For each selected file, the array is
    loaded and, if 3D, the user is prompted to choose its dimension order before
    returning it.

    Parameters
    ----------
    input_dir : str
        Path to the directory containing image files.
    select_multiple : bool, default=False
        If True, allows selection of multiple files. If False, only a single file can be selected.

    Returns
    -------
    List[Tuple[str, Optional[np.ndarray]]]
        A list of (file_path, loaded_array) pairs.
        The array is None if the file failed to load.
        Returns an empty list if no valid files are found or selected.


    Notes
    -----
    - Only files with extensions `.mat`, `.npy`, or `.tif` are considered valid.
    - For `.mat` files, the first valid 3D array found is used.
    - Dimension order is chosen interactively via `choose_image_shape`.
    """
    images: List[str] = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".mat", ".npy", ".tif")):
            images.append(os.path.join(input_dir, fname))

    if not images:
        print(f"[❌] Error: No valid images found in {input_dir}")
        return []

    choices = [{"name": f, "value": f} for f in images]
    selected_files = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
        message="Select which files you want to visualize",
        choices=choices,
        multiselect=select_multiple,
        # default=choices[0]["value"],
    ).execute()

    if not isinstance(selected_files, list):
        selected_files = [selected_files]

    loaded_images: List[Tuple[str, Optional[np.ndarray]]] = []
    for file_path in selected_files:
        img: Optional[np.ndarray] = None
        if file_path.endswith(".mat"):
            img = load_mat(file_path)
        elif file_path.endswith(".npy"):
            img = load_npy(file_path)
        elif file_path.endswith(".tif"):
            img = load_tiff(file_path)

        loaded_images.append((file_path, img))

    return loaded_images


def load_mat(img_path: str) -> Optional[np.ndarray]:
    """
    Load a MATLAB .mat file, find the first valid 3D NumPy array,
    and allow the user to choose its dimension order before returning it.

    Parameters
    ----------
    img_path : str
        Path to the `.mat` file to load.

    Returns
    -------
    Optional[np.ndarray]
        The selected and possibly transposed image array if found,
        otherwise `None`.
    """
    data = loadmat(img_path)

    img: Optional[np.ndarray] = None
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            img = choose_image_shape(value)
            return img

    print(f"[❌] Error: No valid key found in {img_path}")
    return None


def load_npy(img_path: str) -> Optional[np.ndarray]:
    """
    Load a `.npy` file and, if the array is 3D, prompt the user to choose
    its dimension order before returning it.

    Parameters
    ----------
    img_path : str
        Path to the `.npy` file.

    Returns
    -------
    Optional[np.ndarray]
        The selected and possibly transposed image array if it is 3D,
        otherwise `None`.
    """
    img = np.load(img_path)
    if img.ndim == 3:
        return choose_image_shape(img)

    print(f"[❌] Error: Array shape is wrong in {img_path}")
    return None


def load_tiff(img_path: str) -> Optional[np.ndarray]:
    """
    Load a `.tif` hyperspectral image (HSI) and, if the array is 3D,
    prompt the user to choose its dimension order before returning it.

    Parameters
    ----------
    img_path : str
        Path to the `.tif` file.

    Returns
    -------
    Optional[np.ndarray]
        The selected and possibly transposed HSI array if it is 3D,
        otherwise `None`.
    """
    img = tifffile.imread(img_path)
    if img.ndim == 3:
        return choose_image_shape(img)

    print(f"[❌] Error: Unexpected array shape {img.shape} in {img_path}")
    return None


def choose_image_shape(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Ask the user to choose how a 3D NumPy image array's dimensions should be interpreted,
    and transpose the array accordingly.

    Parameters
    ----------
    img : Optional[np.ndarray]
        Input image array, expected to have shape representing some permutation
        of (Height, Width, Channels). If None or not 3D, no changes are made.

    Returns
    -------
    Optional[np.ndarray]
        The image with dimensions rearranged to match the selected format,
        or None if no selection is made.
    """
    if img is None:
        print("[❌] No image data loaded.")
        return None
    if not hasattr(img, "shape") or img.ndim != 3:
        print(f"[❌] Unsupported image shape: {getattr(img, 'shape', None)}")
        return img

    # Mapping from choice value -> H, W, C
    transpose_map = {
        "hwc": None,
        "chw": (1, 2, 0),
        "wch": (2, 0, 1),
        "hcw": (0, 2, 1),
        "whc": (1, 0, 2),
        "cwh": (1, 0, 2),
    }

    # Dynamically build choice list
    choices = []
    for fmt, order in transpose_map.items():
        dims = img.shape if order is None else tuple(img.shape[i] for i in order)
        choices.append(
            {
                "name": f"({', '.join(fmt.upper())}) -> {img.shape}",
                "value": fmt,
            }
        )

    format_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message="Select which format your image is stored in:",
        choices=choices,
        default="hwc",
    ).execute()

    order = transpose_map[format_choice]
    if order is not None:
        img = img.transpose(order)

    print(
        f"[✅] Image transposed from ({', '.join(format_choice.upper())}) to (H, W, C) with shape {img.shape}"
    )
    return img


if __name__ == "__main__":
    print(f"Loading the images from {FILE_PATH}")
    load_image(f"{FILE_PATH}/raw")
    print("-------------------------------------")
