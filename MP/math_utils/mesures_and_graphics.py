import os
import platform
import subprocess
import webbrowser
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MP_PATH = Path(__file__).resolve().parent.parent

def graph_performance(historic, config):
    """
    Generates the final interactive training report using Plotly and displays
    it in a standalone browser window using Chrome application mode.
    """

    # Define output paths
    graph_dir = MP_PATH.parent / "graph" / config.model_name
    os.makedirs(graph_dir, exist_ok=True)
    html_path = graph_dir / "training_report.html"

    # Determine the actual number of executed epochs
    total_epochs = len(historic["train_loss"])
    x_axis = list(range(1, total_epochs + 1))

    # Identify the best epoch based on the minimum validation loss
    best_epoch_loss = int(np.argmin(historic["val_loss"])) + 1
    best_val_loss = historic["val_loss"][best_epoch_loss - 1]

    # Identify the best epoch based in the maximum validation accuracy
    best_epoch_accuracy  = int(np.argmax(historic["val_accuracy"])) + 1
    best_val_accuracy = historic["val_accuracy"][best_epoch_accuracy - 1]

    # Configure the subplot layout (2 rows, 1 column)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=(
            f"Loss Evolution (Min Validation Loss: {best_val_loss:.5f})",
            f"Accuracy Evolution (Max Validation accuracy: {best_val_accuracy:.5f})"
        )
    )

    # Loss subplot (training vs validation)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=historic["train_loss"],
            name="Train Loss",
            line=dict(color="#1f77b4", width=2.5)
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=historic["val_loss"],
            name="Validation Loss",
            line=dict(color="#ff7f0e", width=2.5, dash="dash")
        ),
        row=1,
        col=1
    )

    # Accuracy subplot (training vs validation)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=historic["train_accuracy"],
            name="Train Accuracy",
            line=dict(color="#2ca02c", width=2.5)
        ),
        row=2,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=historic["val_accuracy"],
            name="Validation Accuracy",
            line=dict(color="#d62728", width=2.5, dash="dash")
        ),
        row=2,
        col=1
    )

    fig.add_vline(
        x=best_epoch_loss,
        line_width=1.5,
        line_dash="dot",
        line_color="white",
        row=1,
        col=1
    )

    fig.add_vline(
        x=best_epoch_accuracy,
        line_width=1.5,
        line_dash="dot",
        line_color="white",
        row=2,
        col=1
    )

    fig.add_annotation(
        x=best_epoch_loss,
        y=best_val_loss,
        text=f"best model by loss (Epoch {best_epoch_loss})",
        showarrow=True,
        arrowhead=1,
        row=1,
        col=1
    )

    fig.add_annotation(
        x=best_epoch_accuracy,
        y=best_val_accuracy,
        text=f"Best model by accuracy (Epoch {best_epoch_accuracy})",
        showarrow=True,
        arrowhead=1,
        row=2,
        col=1
    )

    # Configure the visual appearance
    fig.update_layout(
        title=dict(
            text=f"TRAINING REPORT: {config.model_name.upper()} ({config.optimizer_type.upper()})",
            x=0.5,
            font=dict(size=18, color="#ffffff")
        ),
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=50, r=35, t=80, b=50)
    )

    fig.update_xaxes(
        title_text="Epochs",
        range=[1, total_epochs],
        row=2,
        col=1
    )

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=2, col=1)

    # Save the report as a self-contained HTML file
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        # include_plotlyjs="cdn",
        full_html=True,
        auto_open=False
    )

    print(f"Training report saved to: {html_path}")
    print("Opening interactive visualization...")

    # Open the report in a standalone Chrome application window
    url = f"file://{html_path.resolve()}"
    sistema = platform.system()

    try:
        if sistema == "Linux":
            subprocess.Popen(
                ["google-chrome", f"--app={url}", "--window-size=950,650"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        elif sistema == "Darwin":
            subprocess.Popen(
                [
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                    f"--app={url}",
                    "--window-size=950,650",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        elif sistema == "Windows":
            subprocess.Popen(
                ["cmd", "/c", "start", "chrome", f"--app={url}", "--window-size=950,650"],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        else:
            # Fallback to the default browser on unsupported systems
            webbrowser.open(url)

    except Exception:
        # Fallback to the default browser if Chrome launch fails
        webbrowser.open(url)