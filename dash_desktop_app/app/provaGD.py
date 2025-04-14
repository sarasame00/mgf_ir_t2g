import os
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
from matplotlib.colors import LinearSegmentedColormap, to_hex
from .drive_utils import get_file_id_by_name, download_txt_file, creds
import requests, io

# === CONFIG ===
CSV_FILENAME = "simulated_values_ss.csv"
FOLDER_ID = "1qroO12tPkKu6c3w5Xy-2Reys5XFcbX5L"  # Shared Drive folder with all your files
shape = (101, 101)  # Shape of your simulation data files



def load_csv_from_drive_by_name(filename):
    file_id = get_file_id_by_name(filename)
    if not file_id:
        raise FileNotFoundError(f"❌ CSV file '{filename}' not found in Google Drive folder")

    if not creds.valid:
        creds.refresh(Request())

    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    headers = {"Authorization": f"Bearer {creds.token}"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    return pd.read_csv(io.StringIO(response.text))


# === Load main CSV ===
df = load_csv_from_drive_by_name(CSV_FILENAME)

# === Prepare sliders ===
param_values = {
    "N": sorted([int(v) for v in df["N"].unique()]),
    "U": sorted([float(v) for v in df["U"].unique()]),
    "J": sorted([float(v) for v in df["J"].unique()]),
    "g": sorted([float(v) for v in df["g"].unique()]),
    "lbd": sorted([float(v) for v in df["lbd"].unique()]),
    "B": sorted([float(v) for v in df["B"].unique()])
}

# === Custom colormap ===
colors = [
    (0.0, "black"), (0.15, "yellow"), (0.25, "orange"), (0.35, "red"),
    (0.5, "magenta"), (0.65, "blue"), (0.85, "cyan"), (1.0, "white")
]
custom_cmap = LinearSegmentedColormap.from_list("custom_map", colors)
plotly_colorscale = [(i / 254, to_hex(custom_cmap(i / 254))) for i in range(255)]


# === Dash App ===
app = Dash(__name__)

app.layout = html.Div(
    [
        html.Div([
            html.Label(param),
            dcc.Slider(
                min=min(values),
                max=max(values),
                step=None,
                marks={v: str(v) for v in values},
                value=values[0],
                id=f"slider-{param}"
            )
        ], style={'margin': '20px'}) for param, values in param_values.items()
    ] + [
        dcc.Graph(id='energy-map')
    ]
)


@callback(
    Output("energy-map", "figure"),
    [Input(f"slider-{param}", "value") for param in param_values]
)
def update_figure(N, U, J, g, lbd, B):
    match = df[(df["N"] == N) & (df["U"] == U) & (df["J"] == J) &
               (df["g"] == g) & (df["B"] == B) & (df["lbd"] == lbd)]

    if match.empty:
        fig = px.imshow(np.zeros(shape), color_continuous_scale=plotly_colorscale)
        fig.update_layout(title="No matching simulation found")
        return fig

    timestamp = str(match.iloc[0]["timestamp"])
    filename = f"{timestamp}"

    file_id = get_file_id_by_name(filename)

    if not file_id:
        fig = px.imshow(np.zeros(shape), color_continuous_scale=plotly_colorscale)
        fig.update_layout(title=f"Missing file: {filename}")
        return fig

    try:
        data = download_txt_file(file_id)
        emap = data[:, 2].reshape(shape)
        zmin = np.min(emap)
        zmax = zmin + (np.max(emap) - zmin) * 1.0
    except Exception as e:
        print(f"❌ Error loading or reshaping data: {e}")
        emap = np.zeros(shape)
        zmin, zmax = 0, 1

    fig = px.imshow(
        emap,
        color_continuous_scale=plotly_colorscale,
        zmin=zmin,
        zmax=zmax
    )
    fig.update_layout(
        title=f"Ground State Energy — N={N}, U={U}, J={J}, g={g}, B={B}, λ={lbd}",
        transition_duration=300
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)