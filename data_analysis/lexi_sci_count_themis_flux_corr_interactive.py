import datetime
import importlib
import warnings
from pathlib import Path

import lexi_data_analysis_functions as lexi_functions
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

importlib.reload(lexi_functions)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
np.seterr(divide="ignore", invalid="ignore")


def load_and_process_themis_flux(sc: str, date_range: str) -> pd.DataFrame:
    fname = f"../data/themis_data/csv/themis_{sc}_esa_parameters_{date_range}_flux.csv"
    df = pd.read_csv(fname)
    df.rename(columns={df.columns[0]: "epoch"}, inplace=True)
    df.set_index("epoch", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    flux_key = f"th{sc}_peir_flux"
    df = df[[flux_key]].dropna()
    return df.rolling("1s").mean()


date_range = "2025-03-16_to_2025-03-17"
sc_list = ["b", "c"]
themis_flux = {sc: load_and_process_themis_flux(sc, date_range) for sc in sc_list}

# Load Lexi counts
lexi_counts_df = pd.read_csv(
    "/home/cephandrius/Desktop/git/Lexi-BU/lexi_data_pipeline/data/counts_per_second.csv"
)
lexi_counts_df.rename(columns={lexi_counts_df.columns[0]: "epoch"}, inplace=True)
lexi_counts_df.set_index("epoch", inplace=True)
lexi_counts_df.index = pd.to_datetime(lexi_counts_df.index, utc=True)

start_time = datetime.datetime(2025, 3, 16, 19, 0, 0, tzinfo=datetime.timezone.utc)
end_time = datetime.datetime(2025, 3, 16, 21, 15, 0, tzinfo=datetime.timezone.utc)
delta_time_list = [1, 5, 10, 15, 30, 60]  # in minutes

for delta_time in delta_time_list:
    delta_time = datetime.timedelta(minutes=delta_time)

    # Calculate correlations
    correlation_results = {}
    for sc in sc_list:
        results = []
        current_time = start_time
        while current_time + delta_time <= end_time:
            t_start, t_end = current_time, current_time + delta_time
            themis_filtered = themis_flux[sc].loc[t_start:t_end]
            lexi_filtered = lexi_counts_df.loc[t_start:t_end]
            min_len = min(len(themis_filtered), len(lexi_filtered))
            if min_len == 0:
                current_time += delta_time
                continue
            corr = stats.pearsonr(themis_filtered.iloc[:min_len, 0], lexi_filtered.iloc[:min_len, 0])[0]
            # Compute the spearman correlation
            # corr = stats.spearmanr(themis_filtered.iloc[:min_len, 0], lexi_filtered.iloc[:min_len, 0])[
            #     0
            # ]
            results.append({"middle_time": t_start + delta_time / 2, "correlation": corr})
            current_time += delta_time
        correlation_results[sc] = pd.DataFrame(results).set_index("middle_time")

    # === Matplotlib Plot ===
    mpl.style.use("dark_background")
    plt.rcParams.update(
        {
            "axes.facecolor": "#000000",
            "axes.edgecolor": "#ffffff",
            "figure.facecolor": "#020202",
            "figure.edgecolor": "#ffffff",
            "grid.color": "#8E8B8B",
            "text.color": "#ffffff",
            "xtick.color": "#ffffff",
            "ytick.color": "#ffffff",
        }
    )

    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.subplots_adjust(hspace=0)

    colors = {"b": "cyan", "c": "blue"}
    for sc in sc_list:
        axs[0].plot(
            themis_flux[sc].index,
            themis_flux[sc].iloc[:, 0],
            label=f"THEMIS {sc.upper()} Flux",
            color=colors[sc],
            alpha=0.5,
        )
    axs[0].set_ylabel("Flux (particles/s/m^2)")
    twinx = axs[0].twinx()
    twinx.plot(lexi_counts_df.index, lexi_counts_df.iloc[:, 0], color="orange", label="Lexi Counts")
    twinx.set_ylabel("Counts (particles/s)")
    axs[0].legend(loc="upper left")
    twinx.legend(loc="upper right")

    for sc in sc_list:
        axs[1].plot(
            correlation_results[sc].index,
            correlation_results[sc]["correlation"],
            label=f"Corr THEMIS {sc.upper()}",
            lw=1,
        )
    axs[1].set_ylabel("Correlation Coefficient")
    axs[1].set_xlabel("Time [UTC]")
    axs[1].legend()

    axs[1].xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    axs[1].set_xlim(start_time, end_time)
    axs[1].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))

    for ax in axs:
        for i in range(int((end_time - start_time) / delta_time)):
            ax.axvline(start_time + i * delta_time, color="gray", linestyle="--", linewidth=0.5)
        sunset_start = datetime.datetime(2025, 3, 16, 19, 38, 0, tzinfo=datetime.timezone.utc)
        sunset_end = datetime.datetime(2025, 3, 16, 20, 44, 0, tzinfo=datetime.timezone.utc)
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(
            gradient,
            extent=[sunset_start, sunset_end, ax.get_ylim()[0], ax.get_ylim()[1]],
            aspect="auto",
            cmap="binary_r",
            alpha=0.3,
            zorder=1,
        )
        if ax is axs[0]:
            ax.text(
                sunset_start + datetime.timedelta(minutes=1),
                ax.get_ylim()[1] * 0.8,
                f"Sunset Start\n{sunset_start.strftime('%H:%M')}",
                color="white",
                ha="left",
            )
            ax.text(
                sunset_end - datetime.timedelta(minutes=1),
                ax.get_ylim()[1] * 0.8,
                f"Sunset End\n{sunset_end.strftime('%H:%M')}",
                color="grey",
                ha="right",
            )

    plt.tight_layout()
    fig_folder = Path("../figures/themis_lexi_correlation")
    fig_folder.mkdir(parents=True, exist_ok=True)
    fig_name = f"themis_lexi_correlation_{delta_time.total_seconds()}s_bc.png"
    fig.savefig(fig_folder / fig_name, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # === Plotly Plot ===

    fig_plotly = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("THEMIS Flux and Lexi Counts", "Correlation Coefficients"),
        specs=[[{"secondary_y": True}], [{}]],
    )

    # Plot THEMIS fluxes (row 1, primary y-axis)
    for sc in sc_list:
        fig_plotly.add_trace(
            go.Scatter(
                x=themis_flux[sc].index,
                y=themis_flux[sc].iloc[:, 0],
                name=f"THEMIS {sc.upper()} Flux",
                line=dict(color=colors[sc]),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # Plot Lexi counts (row 1, secondary y-axis)
    fig_plotly.add_trace(
        go.Scatter(
            x=lexi_counts_df.index,
            y=lexi_counts_df.iloc[:, 0],
            name="Lexi Counts",
            line=dict(color="orange"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Plot correlation values (row 2)
    for sc in sc_list:
        fig_plotly.add_trace(
            go.Scatter(
                x=correlation_results[sc].index,
                y=correlation_results[sc]["correlation"],
                name=f"Corr THEMIS {sc.upper()}",
                mode="lines+markers",
                line=dict(dash="dash"),
            ),
            row=2,
            col=1,
        )

    # Define sunset markers
    sunset_start = datetime.datetime(2025, 3, 16, 19, 38, 0, tzinfo=datetime.timezone.utc)
    sunset_end = datetime.datetime(2025, 3, 16, 20, 44, 0, tzinfo=datetime.timezone.utc)

    # Add shaded rectangle for sunset in both subplots
    for row in [1, 2]:
        fig_plotly.add_vrect(
            x0=sunset_start,
            x1=sunset_end,
            fillcolor="grey",
            opacity=0.2,
            layer="below",
            line_width=0,
            row=row,
            col=1,
        )

    fig_plotly.update_layout(
        title="THEMIS B/C Flux, Lexi Counts, and Correlation with Sunset Highlighted",
        template="plotly_dark",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig_plotly.update_yaxes(title_text="Flux (particles/s/m²)", row=1, col=1, secondary_y=False)
    fig_plotly.update_yaxes(title_text="Lexi Counts (particles/s)", row=1, col=1, secondary_y=True)
    fig_plotly.update_yaxes(title_text="Correlation Coefficient", row=2, col=1)

    fig_plotly.update_xaxes(title_text="Time [UTC]", row=2, col=1)
    # Set the x-axis range to the start and end time
    fig_plotly.update_xaxes(range=[start_time, end_time], row=2, col=1)

    # Save as HTML
    html_path = fig_folder / f"themis_lexi_correlation_subplots_{delta_time.total_seconds()}s_bc.html"
    fig_plotly.write_html(html_path)
