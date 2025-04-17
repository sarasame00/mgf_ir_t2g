import os
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
from matplotlib.colors import LinearSegmentedColormap, to_hex

# === Load parameter log ===
df = pd.read_csv("/Users/sarasalesmerino/Desktop/ICMAB/mgf_ir_t2g/single_site/ss_data/simulated_values_ss.csv")  

# === Simulation output folder ===
data_dir = "/Users/sarasalesmerino/Desktop/ICMAB/mgf_ir_t2g/single_site/ss_data/ss_results/"
shape = (101, 101)  # Update if needed

# === Prepare parameter sliders ===
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

# === Precompute global contrast range ===
vmin = df["timestamp"].apply(
    lambda ts: np.min(np.loadtxt(os.path.join(data_dir, str(ts))))
).min()
vmax = vmin + (
    df["timestamp"].apply(
        lambda ts: np.max(np.loadtxt(os.path.join(data_dir, str(ts))))
    ).max() - vmin
) * 0.15

# === Build Dash layout ===
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

# === Dynamic plot callback ===
@callback(
    Output("energy-map", "figure"),
    [Input(f"slider-{param}", "value") for param in param_values]
)
def update_figure(N, U, J, g, lbd, B):
    print(f"\n▶ Selected params: N={N}, U={U}, J={J}, g={g}, lbd={lbd}, B={B}")

    match = df[(df["N"] == N) & (df["U"] == U) & (df["J"] == J) &
               (df["g"] == g) & (df["B"] == B) & (df["lbd"] == lbd)]

    if match.empty:
        print(f"❌ No matching row found in CSV.")
        fig = px.imshow(np.zeros(shape), color_continuous_scale=plotly_colorscale)
        fig.update_layout(title="No matching simulation found")
        return fig

    timestamp = str(match.iloc[0]["timestamp"])
    filepath = os.path.join(data_dir, timestamp)

    print(f"✔ Found timestamp: {timestamp}")
    print(f"📂 Looking for file: {filepath}")

    if not os.path.exists(filepath):
        print("❌ File not found.")
        fig = px.imshow(np.zeros(shape), color_continuous_scale=plotly_colorscale)
        fig.update_layout(title=f"Data file missing: {timestamp}")
        return fig

    data = np.loadtxt(filepath)
    print(f"📈 Loaded data shape: {data.shape}")

    try:
        emap = data[:, 2].reshape(shape)
        zmin = np.min(emap)
        zmax = zmin + (np.max(emap) - zmin) * 1.0  # Use full range just to be sure
    except Exception as e:
        print(f"❌ Failed to reshape or normalize data: {e}")
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
