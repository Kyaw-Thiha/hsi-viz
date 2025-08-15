import typer
import numpy as np
from InquirerPy import inquirer

from image_loader import load_image
from image_viewer import ImageViewer
from image_renderers import SingleBandRenderer, SingleAxisSliceRenderer, SinglePixelSpectrumRenderer

app = typer.Typer()

INPUT_DIR = "input"
OUTPUT_DIR = "output"


@app.command()
def viz_three():
    pass


@app.command()
def viz_two(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
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

            save_choices = [
                {"name": "JPG", "value": "jpg"},
                {"name": "PNG", "value": "png"},
                {"name": "SVG", "value": "svg"},
                {"name": "PDF", "value": "pdf"},
                {"name": "HTML", "value": "html"},
                {"name": "Don't Save", "value": ""},
            ]
            save_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
                message="Select which format your image is stored in:",
                choices=save_choices,
                default="",
            ).execute()
            if save_choice != "" and fig is not None:
                default_save_name = f"{output_dir}/plot.{save_choice}"
                save_name: str = inquirer.text(message="Enter the filename to save as:", default=default_save_name).execute()  # type: ignore[reportPrivateImportUsage]
                if save_choice == "html":
                    fig.write_html(save_name)
                else:
                    fig.write_image(save_name)

        print("\n")


if __name__ == "__main__":
    app()
