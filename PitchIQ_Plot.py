import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
from PIL import Image
from highlight_text import fig_text

class pitchiq_plot:
    def __init__(self):
        """
        Initialize the class without any dataset.
        The dataset will be passed directly to the `create_scatter_plot` method.
        """
        pass

    def create_scatter_plot(self, data, metric, x_var, y_var, title, top_n=10, minutes_col='90s', min_minutes=4.5):
        """
        Create a scatter plot for the top N players based on a given metric.

        Parameters:
        - data: The dataset (DataFrame) containing player stats.
        - metric: The metric to rank players by (e.g., 'xAG').
        - x_var: The x-axis variable (e.g., 'Key Passes per 90').
        - y_var: The y-axis variable (e.g., 'Expected Assists per 90').
        - title: The title of the plot.
        - top_n: The number of top players to highlight (default is 10).
        - minutes_col: The column name for minutes played (default is '90s').
        - min_minutes: The minimum minutes played to filter players (default is 4.5).
        """
        # Ensure the dataset has the required columns
        required_columns = ['Player', metric, x_var, y_var, minutes_col]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"The dataset must contain the following columns: {required_columns}")

        # Filter players based on minimum minutes played
        data = data[data[minutes_col] >= min_minutes]

        # Get the top N players based on the given metric
        top_players = data.nlargest(top_n, metric)
        players = top_players['Player'].tolist()

        # Split the data into highlighted and non-highlighted players
        df_main = data[~data["Player"].isin(players)].reset_index(drop=True)
        df_highlight = data[data["Player"].isin(players)].reset_index(drop=True)

        # Apply FiveThirtyEight style
        plt.style.use("fivethirtyeight")

        # Create the scatter plot
        fig = plt.figure(figsize=(8, 8), dpi=300)
        ax = plt.subplot()

        # Customize the plot appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Plot non-highlighted players
        ax.scatter(
            df_main[x_var],
            df_main[y_var],
            s=100,  # Increased dot size
            alpha=0.75,
            color="#264653",
            zorder=3
        )

        # Plot highlighted players
        ax.scatter(
            df_highlight[x_var],
            df_highlight[y_var],
            s=100,  # Increased dot size
            alpha=0.95,
            color="#F64740",
            zorder=3,
            ec="#000000",
        )

        # Add median lines
        ax.plot(
            [data[x_var].median(), data[x_var].median()],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            ls=":",
            color="gray",
            zorder=2
        )

        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [data[y_var].median(), data[y_var].median()],
            ls=":",
            color="gray",
            zorder=2
        )

        # Add grid
        ax.grid(True, ls=":", color="lightgray")

        # Annotate highlighted players
        for index, row in df_highlight.iterrows():
            X = row[x_var]
            Y = row[y_var]
            name = row["Player"]

            # Use a consistent offset for all players
            x_pos = 5  # Horizontal offset
            y_pos = -10  # Vertical offset (negative to place text below the point)

            # Add annotation
            text_ = ax.annotate(
                xy=(X, Y),
                text = name.split()[-1] if len(name.split()) > 1 else name ,  # Use the player's last name
                ha="right",
                va="bottom",
                xytext=(x_pos, y_pos),
                textcoords="offset points",
                fontsize=8,  # Adjust font size for annotations
            )

            # Add white stroke effect to annotations
            text_.set_path_effects(
                [path_effects.Stroke(linewidth=2.5, foreground="white"),
                path_effects.Normal()]
            )

        # Add labels and title
        ax.set_xlabel(x_var.replace("_", " "), fontsize=10)  # Remove underscores
        ax.set_ylabel(y_var.replace("_", " "), fontsize=10)  # Remove underscores

        ax.tick_params(axis='both', which='major', labelsize=8)  # Smaller tick labels

        # Add league icon (top-left corner)
        try:
            league_icon = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images/premier-league-2-logo.png")
            league_ax = fig.add_axes([0.02, 0.88, 0.10, 0.10], zorder=1)  # Top-left corner
            league_ax.imshow(league_icon)
            league_ax.axis("off")
        except FileNotFoundError:
            print("League icon not found. Skipping...")

        # Add custom logo (top-right corner)
        try:
            ax3 = fig.add_axes([0.88, 0.88, 0.10, 0.10], zorder=1)  # Top-right corner
            ax3.axis('off')
            img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
            ax3.imshow(img)
        except FileNotFoundError:
            print("Custom logo not found. Skipping...")

        # Add title (right of the league icon)
        fig_text(
            x=0.15, y=0.93,  # Adjusted to be right of the league icon
            s=title,
            highlight_textprops=[{"color": "#228B22", "style": "italic"}],
            va="bottom", ha="left",
            fontsize=13, color="black", weight="bold"  # Increased font size for title
        )

        # Add subtitle (below the title)
        fig_text(
            x=0.15, y=0.86,  # Adjusted to prevent overlap with title
            s=f"{x_var.replace('_', ' ')} vs {y_var.replace('_', ' ')}\nSeason 2024/2025\nPlayers with more than {min_minutes * 90} minutes are considered. Viz by @stephenaq7.",
            va="bottom", ha="left",
            fontsize=7, color="#4E616C"  # Increased font size for subtitle
        )

        plt.show()