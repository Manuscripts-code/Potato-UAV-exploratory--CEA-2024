import pandas as pd

from configs import configs

"""
This script is used to search for indices that are appropriate for our multispectral sensor.
"""

replacements = {
    "RE1": "RE",
    "RE2": "RE",
    "RE3": "RE",
    "N2": "N",
    "G1": "G",
}


def replace_bands(formula: str) -> str:
    """
    Replace the bands in the formula with the bands that are available in our sensor.
    Also change it to format suitable for formatting in word document.
    """
    for old_band, new_band in replacements.items():
        formula = formula.replace(old_band, new_band)
        formula = formula.replace(" ", "")
        formula = formula.replace("**", "^")
    return formula


if __name__ == "__main__":
    # Github: https://github.com/awesome-spectral-indices/awesome-spectral-indices
    # read json from github link
    gh_link = "https://raw.githubusercontent.com/awesome-spectral-indices/awesome-spectral-indices/main/output/spectral-indices-table.csv"
    df = pd.read_csv(gh_link)
    all_available_names = []
    for name, bands in zip(df["short_name"], df["bands"]):
        bands = eval(bands)
        if set(bands).issubset(set(["B", "G", "G1", "R", "RE1", "RE2", "RE3", "N", "N2"])):
            print(f"{name}:    {bands}")
            all_available_names.append(name)
    print(f"all_available_names: {all_available_names}")

    # display table of indices for manuscript
    df_available = df[df["short_name"].isin(all_available_names)].reset_index(drop=True)
    df_available["formula"] = df_available["formula"].apply(replace_bands)
    df_available[["short_name", "long_name", "formula", "reference"]].to_excel(
        configs.SAVE_RESULTS_DIR / "spectral_indices_table.xlsx", index=False
    )
