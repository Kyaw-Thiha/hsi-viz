import typer
from InquirerPy import inquirer

from image_loader import load_image
from image_viewer import ImageViewer
from image_renderers import SingleBandRenderer

app = typer.Typer()

INPUT_DIR = "input"
OUTPUT_DIR = "output"


@app.command()
def viz_three():
    pass


@app.command()
def viz_two(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    loaded_images = load_image(input_dir)

    while True:
        image_choices = [{"name": img[0], "value": img[0]} for img in loaded_images]
        image_choices.append({"name": "Exit", "value": ""})
        image_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
            message="Select which image you want to visualize:",
            choices=image_choices,
        ).execute()

        if image_choice == "":
            break
        image_tuple = next((t for t in loaded_images if t[0] == image_choice), None)
        if image_tuple is not None:
            image_name, image = image_tuple
            if image is not None:
                visualization_strategies = {
                    "spatial_monogray": ("[Spatial]: GreyScale", SingleBandRenderer("gray")),
                    "spatial_monocolor": ("[Spatial]: MonoColor", SingleBandRenderer()),
                    "spatial_rgb": ("[Spatial]: RGB", SingleBandRenderer()),
                }
                visualization_choices = [
                    {"name": label, "value": key} for key, (label, _factory) in visualization_strategies.items()
                ]
                visualization_choice = inquirer.fuzzy(  # type: ignore[reportPrivateImportUsage]
                    message="Select what strategy you want to use to visualize: ",
                    choices=visualization_choices,
                ).execute()

                stratgy = visualization_strategies.get(visualization_choice)
                if stratgy is not None:
                    strategy_name, renderer = stratgy
                    image_viewer = ImageViewer(renderer, image, image_name, output_dir)
                    image_viewer.render()


if __name__ == "__main__":
    app()
