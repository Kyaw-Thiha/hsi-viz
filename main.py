import os
import typer
import numpy as np
from scipy.io import savemat
from InquirerPy import inquirer

from image_loader import load_image
from image_saver import change_image_shape, get_save_choice
from image_viewer import ImageViewer
from image_renderers import SingleBandRenderer, SingleAxisSliceRenderer, SinglePixelSpectrumRenderer, VolumeRenderer

app = typer.Typer()

INPUT_DIR = "input/test-5"
OUTPUT_DIR = "output"


@app.command()
def reshape(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    loaded_images = load_image(input_dir, select_multiple=False)
    print("\n")

    if not loaded_images:
        print("[❌] No images selected; aborting reshape.")
        return

    file_path, img = loaded_images[0]
    if img is None:
        print(f"[❌] Unable to reshape {file_path}; image data missing.")
        return

    transpose_map = {
        "hwc": (0, 1, 2),
        "chw": (2, 0, 1),
        "wch": (1, 2, 0),
        "hcw": (0, 2, 1),
        "whc": (2, 0, 1),
        "cwh": (1, 0, 2),
    }
    img = change_image_shape(img, "Select which format you want to store your image in:", transpose_map)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    default_save_name = os.path.join(output_dir, f"{base_name}_reshaped.mat")
    save_name: str = inquirer.text(  # type: ignore[reportPrivateImportUsage]
        message="Enter the filename to save as:", default=default_save_name
    ).execute()

    if not save_name.lower().endswith(".mat"):
        save_name = f"{save_name}.mat"

    savemat(save_name, {"image": img})
    print(f"[✅] Saved reshaped image to {save_name}")


@app.command()
def viz(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR, show_fig: bool = True):
    loaded_images = load_image(input_dir, select_multiple=True)
    print("\n")

    while True:
        image_choices = [{"name": img[0], "value": img[0]} for img in loaded_images]
        image_choices.append({"name": "Exit", "value": ""})
        chosen_images = inquirer.checkbox(  # type: ignore[reportPrivateImportUsage]
            message="Press [TAB] to select and [ENTER] to confirm. \nSelect which images you want to visualize:",
            choices=image_choices,
            default=image_choices[0]["value"],
        ).execute()

        if "" in chosen_images:
            break
        if len(chosen_images) == 0:
            print("Select an image by pressing [TAB].")
            continue

        image_tuples = [t for t in loaded_images if t[0] in chosen_images]
        images: list[tuple[str, np.ndarray]] = [(name, arr) for name, arr in image_tuples if arr is not None]

        visualization_strategies = {
            "spatial_monogray": ("[Spatial]: GreyScale", SingleBandRenderer("gray")),
            "spatial_monocolor": ("[Spatial]: MonoColor", SingleBandRenderer()),
            "spectral_single_axis": ("[Spectral]: Single-Axis Slice", SingleAxisSliceRenderer()),
            "spectral_single_pixel": ("[Spectral]: Single-Pixel Spectrum", SinglePixelSpectrumRenderer()),
            "3d_cube": ("[3D]: Cube", VolumeRenderer(max_voxels=500_000, debug=True)),
            "spatial_rgb": ("[Spatial]: RGB", SingleBandRenderer()),
        }
        visualization_choices = [{"name": label, "value": key} for key, (label, _factory) in visualization_strategies.items()]
        visualization_choice = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
            message="Select what strategy you want to use to visualize: ",
            choices=visualization_choices,
        ).execute()

        stratgy = visualization_strategies.get(visualization_choice)
        if stratgy is not None:
            strategy_name, renderer = stratgy
            image_viewer = ImageViewer(renderer, images, output_dir)
            fig = image_viewer.render()
            if show_fig and fig is not None:
                fig.show()

            save_choice = get_save_choice(["jpg", "png", "svg", "pdf", "html"])
            if save_choice != "" and fig is not None:
                os.makedirs("output", exist_ok=True)
                default_save_name = f"{output_dir}/plot.{save_choice}"
                save_name: str = inquirer.text(message="Enter the filename to save as:", default=default_save_name).execute()  # type: ignore[reportPrivateImportUsage]
                if save_choice == "html":
                    fig.write_html(save_name)
                else:
                    fig.write_image(save_name)

        print("\n")


if __name__ == "__main__":
    app()
