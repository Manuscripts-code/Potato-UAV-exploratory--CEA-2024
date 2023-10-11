import pandas as pd

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
