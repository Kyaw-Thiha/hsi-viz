from InquirerPy import inquirer

# Base colorscales
_BASE_COLOR_SCALES = {
    "gray": ("Grayscale", "gray"),
    "viridis": ("Viridis (balanced, colorblind-friendly)", "viridis"),
    "magma": ("Magma (dark-to-bright purple/orange)", "magma"),
    "inferno": ("Inferno (fiery, high contrast)", "inferno"),
    "plasma": ("Plasma (purple→yellow)", "plasma"),
    "cividis": ("Cividis (perceptually uniform, CVD-friendly)", "cividis"),
    "ice": ("Ice (cool blues)", "ice"),
    "thermal": ("Thermal (cool→hot)", "thermal"),
    "aggrnyl": ("Aggrnyl (aqua→green)", "aggrnyl"),
    "sunset": ("Sunset (pink→orange)", "sunset"),
}

# Expanded with inverted versions
COLOR_SCALES = {}
for key, (label, scale) in _BASE_COLOR_SCALES.items():
    COLOR_SCALES[key] = (label, scale)
    COLOR_SCALES[key + "_r"] = (label + " (Inverted)", scale + "_r")


def pick_mono_color():
    color_choices = [{"name": label, "value": key} for key, (label, _) in COLOR_SCALES.items()]
    color_choice = inquirer.select(  # type: ignore[reportPrivateImportUsage]
        message="Select a monochrome/gradient palette:",
        choices=color_choices,
        default="gray",
    ).execute()
    return color_choice
