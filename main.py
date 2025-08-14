import typer
from InquirerPy import inquirer

from image_loader import load_image

app = typer.Typer()

INPUT_DIR = "input"
OUTPUT_DIR = "output"


@app.command()
def viz_three():
    pass


@app.command()
def viz_two(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    image = load_image(input_dir)

    color_action = inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message="Select how you to visualize the 2D Image:",
        choices=[
            {"name": "RGB", "value": "rgb"},
            {"name": "BW", "value": "bw"},
            {"name": "Exit", "value": None},
        ],
        default="rgb",
    ).execute()

    if color_action is None:
        print("Bye")
    elif color_action == "rgb":
        print("RGB path")
    elif color_action == "bw":
        print("BW path")


if __name__ == "__main__":
    app()
