import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from PIL import Image
import urllib.request
import matplotlib.image as image
from matplotlib import style
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
from mplsoccer import Radar
from mplsoccer import PyPizza, add_image, FontManager, Radar
from mplsoccer.utils import add_image
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from highlight_text import fig_text, ax_text


class GoalkeeperPlotter:
    def __init__(self, gk_role_templates, image_dir="/Users/stephenahiabah/Desktop/Code/cannoniq/Images/"):
        self.gk_role_templates = gk_role_templates
        self.image_dir = image_dir
        
        # Font properties (you may need to adjust paths or use system fonts)
        self.font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf')
        self.font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Italic.ttf')
        self.font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf')


        
        # GK-specific stat mapping for cleaner display names
        self.gk_stat_mapping = {
            'Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90': 'PSxG+/- per 90',
            'Goals_Against': 'Goals Against per 90',
            'Launched_Pass_Completion_Percentage': 'Long Pass %',
            'Average_Pass_Length': 'Avg Pass Length',
            'Crosses_Stopped_Percentage': 'Cross Stop %',
            'Defensive_Actions_Outside_Penalty_Area_Per_90': 'Sweeping per 90',
            'Average_Distance_Of_Defensive_Actions': 'Avg Sweep Distance',
            'Crosses_Stopped': 'Crosses Stopped per 90',
            'Crosses_Faced': 'Crosses Faced per 90',
            'Launched_Passes_Completed': 'Long Passes Completed per 90',
            'Launched_Passes_Attempted': 'Long Passes Attempted per 90',
            'Passes_Attempted_Excluding_Goal_Kicks': 'Short Passes per 90',
            'Throws_Attempted': 'Throws per 90',
            'Shot_Stopping_Score': 'Shot Stopping',
            'Distribution_Score': 'Distribution',
            'Claiming_Score': 'Claiming',
            'Sweeping_Score': 'Sweeping'
        }

    def plot_gk_role_comparison(self, player_name, gk_style, df=None, comparative_list=None):
        """Create radar chart for goalkeeper role comparison"""
        plt.style.use("fivethirtyeight")

        if gk_style not in self.gk_role_templates:
            raise ValueError(f"Unknown GK style '{gk_style}'.")

        if df is None:
            raise ValueError("No dataframe provided.")

        player_dir = f"GK_profiles/{player_name}"
        os.makedirs(player_dir, exist_ok=True)

        # Get role-specific parameters
        params = self.gk_role_templates[gk_style]['key_stats']
        readable_params = [self.gk_stat_mapping.get(p, p) for p in params]

        main_player_row = df[df['Player'] == player_name]
        if main_player_row.empty:
            raise ValueError(f"{player_name} not found in dataset")

        # Filter to goalkeepers with sufficient playing time
        scaled_df = df[df['Minutes_90s'] > 10]  # Minimum 10 * 90 minutes played

        def get_player_data(name):
            row = df[df['Player'] == name]
            if row.empty:
                raise ValueError(f"{name} not found in dataset")
            return row[params].values.flatten().tolist()

        main_values = [float(x) for x in get_player_data(player_name)]
        comp_values_list = [[float(x) for x in get_player_data(name)] for name in comparative_list] if comparative_list else []

        # Single radar chart
        if not comp_values_list:
            low = [0] * len(params)
            high = scaled_df[params].max().tolist()

            radar = Radar(readable_params, low, high, round_int=[False]*len(params),
                          num_rings=5, ring_width=1, center_circle_radius=1)
            fig, ax = radar.setup_axis()
            fig.patch.set_facecolor('#ededed')
            ax.set_facecolor('#ededed')
            radar.draw_circles(ax=ax, facecolor='#dddddd')
            radar.draw_radar(main_values, ax=ax, kwargs_radar={'facecolor': '#228B22', 'alpha': 0.65})
            radar.draw_range_labels(ax=ax, fontsize=15)
            radar.draw_param_labels(ax=ax, fontsize=15)
            ax.legend([player_name], loc='upper right', fontsize=12)

            club = main_player_row['Squad'].values[0]
            age = float(main_player_row['Age'].values[0])
            nineties = float(main_player_row["Minutes_90s"].values[0])
            season = "2024–2025"

            fig.savefig(f"{player_dir}/{player_name}_{gk_style}_radar.png", dpi=300, bbox_inches="tight")

            self._add_gk_title_text(fig, player_name, club, age, nineties, season, gk_style, "Goalkeepers")

        # Comparison radar charts
        else:
            for i, comp_values in enumerate(comp_values_list):
                low = [min(m, c) * 0.5 for m, c in zip(main_values, comp_values)]
                high = [max(m, c) * 1.05 for m, c in zip(main_values, comp_values)]

                radar = Radar(readable_params, low, high, round_int=[False]*len(params),
                              num_rings=5, ring_width=1, center_circle_radius=1)

                fig, ax = radar.setup_axis()
                fig.patch.set_facecolor('#f0f0f0')
                ax.set_facecolor('#f0f0f0')

                radar.draw_circles(ax=ax, facecolor='#ffb2b2')
                radar.draw_radar_compare(main_values, comp_values, ax=ax,
                                         kwargs_radar={'facecolor': '#228B22', 'alpha': 0.6},
                                         kwargs_compare={'facecolor': '#FF6B35', 'alpha': 0.6})
                radar.draw_range_labels(ax=ax, fontsize=15)
                radar.draw_param_labels(ax=ax, fontsize=15)
                ax.legend([player_name, comparative_list[i]], loc='upper right', fontsize=12)

                comp_name = comparative_list[i]
                fig.savefig(f"{player_dir}/{player_name}_vs_{comp_name}_radar.png", dpi=300, bbox_inches="tight")

                self._add_gk_comparison_text(fig, player_name, comp_name, main_player_row, gk_style)

        self._add_gk_badge_overlay(fig)

    def plot_gk_role_kde(self, player_name, gk_style, df=None):
        """Create KDE distribution plots for GK role statistics"""
        if gk_style not in self.gk_role_templates:
            raise ValueError(f"Invalid GK style '{gk_style}'.")

        if df is None:
            raise ValueError("No dataframe provided.")

        params = self.gk_role_templates[gk_style]['key_stats']
        
        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        # Filter to goalkeepers with sufficient playing time
        filtered_df = df[df['Minutes_90s'] > 10]

        num_params = len(params)
        fig, axs = plt.subplots(num_params, 1, figsize=(7, num_params * 1.2))
        plt.subplots_adjust(hspace=0.8)

        if num_params == 1:
            axs = [axs]

        for i, param in enumerate(params):
            ax = axs[i]

            # Get player value first
            player_val = float(player_row[param].values[0])
            
            # Skip if player value is NaN
            if pd.isna(player_val):
                continue
            
            # Get data and calculate percentile
            data = pd.to_numeric(filtered_df[param], errors='coerce').dropna().astype(np.float64)
            
            # Calculate percentile based on stat type
            if param in ['Goals_Against']:  # Stats where lower is better
                percentile_val = (data > player_val).mean() * 100
            else:  # Stats where higher is better
                percentile_val = (data < player_val).mean() * 100
            
            # Handle NaN percentile values
            if pd.isna(percentile_val):
                percentile = 0
            else:
                percentile = round(percentile_val)

            # Pad data for KDE
            data = np.concatenate(([0], data))

            # KDE plots
            sns.kdeplot(data, color="gray", ax=ax, linewidth=1)
            sns.kdeplot(data, fill=True, alpha=0.35, color="green", ax=ax, linewidth=0,
                        clip=(data.min(), player_val))

            # Player marker
            ax.axvline(player_val, color='red', linestyle='-', lw=1)
            ax.plot(player_val, 0, '^', color='red', markersize=6)

            # Axes cleanup
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelsize=6)
            ax.set_xlim(left=0)

            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.grid(True, linestyle=':', linewidth=0.5, color='gray')

            # Labels
            clean_label = self.gk_stat_mapping.get(param, param)
            ax.text(
                ax.get_xlim()[0],
                ax.get_ylim()[1] * 1.2,
                f"{clean_label}: {player_val:.2f}",
                fontsize=10, ha='left', va='center', fontweight='book', color='black'
            )
            ax.text(
                ax.get_xlim()[1],
                ax.get_ylim()[1] * 1.2,
                f"Rk: {percentile:.1f}%",
                fontsize=10, ha='right', va='center', fontweight='book', color='red'
            )

        # Save KDE plot
        player_dir = f"GK_profiles/{player_name}"
        os.makedirs(player_dir, exist_ok=True)
        fig.savefig(f"{player_dir}/{player_name}_{gk_style}_kde.png", dpi=300, bbox_inches="tight")

    def combine_gk_profile_plots(self, player_name, gk_style, df=None, include_logos=True):
        """Combine radar and KDE plots into a single profile"""
        if df is None:
            raise ValueError("No dataframe provided.")

        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        club = player_row['Squad'].values[0]
        age = float(player_row['Age'].values[0])
        nineties = float(player_row["Minutes_90s"].values[0])
        season = "2024–2025"
        competition = player_row['Competition'].values[0]

        # Paths
        radar_path = f"GK_profiles/{player_name}/{player_name}_{gk_style}_radar.png"
        kde_path = f"GK_profiles/{player_name}/{player_name}_{gk_style}_kde.png"
        output_path = f"GK_profiles/{player_name}/{player_name}_{gk_style}_combined_profile.png"

        radar_img = Image.open(radar_path)
        kde_img = Image.open(kde_path)

        # Match radar height
        radar_w, radar_h = radar_img.size
        kde_img = kde_img.resize((int(kde_img.width), radar_h))

        fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
        fig.patch.set_facecolor('#ededed')
        ax.axis('off')

        combined_img = Image.new("RGB", (radar_w + kde_img.width, radar_h), (237, 237, 237))
        combined_img.paste(radar_img, (0, 50))
        combined_img.paste(kde_img, (radar_w, 0))

        ax.imshow(combined_img)

        # Add logos if available
        if include_logos:
            try:
                badge_img = Image.open(os.path.join(self.image_dir, "piqmain.png"))
                ax_badge = fig.add_axes([0.75, 0.88, 0.12, 0.18])
                ax_badge.imshow(badge_img)
                ax_badge.axis('off')
            except FileNotFoundError:
                print("Logo not found – skipping logo.")

        # Title and credits
        title = (
            f"{club} | {player_name} (GK)\n"
            f"90's Played: {nineties:.1f} | Age: {age:.1f}\n"
            f"Season: {season} | Competition: {competition}\n"
            f"{gk_style} Template"
        )

        CREDIT_1 = "viz by @pitchiq.bsky.social\ndata via FBREF / Opta"
        CREDIT_2 = "inspired by: @cannonstats.com, @FootballSlices"

        fig.text(
            0.88, 0.00, f"{CREDIT_1}\n{CREDIT_2}", size=9,
            fontproperties=self.font_italic.prop, color="#000000", ha="right"
        )

        plt.suptitle(title, fontsize=13, fontweight='book', ha='left', x=0.25, y=1.02)

        plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()

        return output_path

    def create_gk_percentile_pizza(self, player_name, gk_style, df=None, output_dir=None):
        """Create percentile pizza chart for goalkeepers"""
        if df is None:
            raise ValueError("No dataframe provided.")

        if gk_style not in self.gk_role_templates:
            raise ValueError(f"Invalid GK style '{gk_style}'.")

        params = self.gk_role_templates[gk_style]['key_stats']
        readable_params = [self.gk_stat_mapping.get(p, p) for p in params]

        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        filtered_df = df[df['Minutes_90s'] > 10]

        percentiles = []
        for param in params:
            data = pd.to_numeric(filtered_df[param], errors='coerce').dropna().astype(np.float64)
            player_val = float(player_row[param].values[0])
            percentile = round((data < player_val).mean() * 100)
            percentiles.append(percentile)

        # GK-specific color scheme (green tones for goalkeepers)
        slice_colors = ["#228B22"] * 2 + ["#32CD32"] * 2 + ["#90EE90"] * 2 + ["#006400"] * 2
        if len(params) > 8:
            slice_colors.extend(["#7CFC00"] * (len(params) - 8))
        slice_colors = slice_colors[:len(params)]
        
        text_colors = ["#FFFFFF"] * len(params)

        try:
            from mplsoccer import PyPizza
            
            baker = PyPizza(
                params=readable_params,
                background_color="#EBEBE9",
                straight_line_color="#EBEBE9",
                straight_line_lw=1,
                last_circle_lw=0,
                other_circle_lw=0,
                inner_circle_size=20
            )

            fig, ax = baker.make_pizza(
                percentiles,
                figsize=(8, 8.5),
                color_blank_space="same",
                slice_colors=slice_colors,
                value_colors=text_colors,
                value_bck_colors=slice_colors,
                blank_alpha=0.4,
                kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
                kwargs_params=dict(color="#000000", fontsize=9, va="center", fontproperties=self.font_normal.prop),
                kwargs_values=dict(
                    color="#000000", fontsize=11, fontproperties=self.font_normal.prop, zorder=3,
                    bbox=dict(edgecolor="#000000", facecolor="lightgreen", boxstyle="round,pad=0.2", lw=1)
                )
            )

            club = player_row['Squad'].values[0]
            competition = player_row['Competition'].values[0]

            fig.text(0.05, 0.985, f"{player_name} (GK) - {club} - {gk_style}", size=14,
                    ha="left", fontproperties=self.font_bold.prop, color="#000000")
            fig.text(0.05, 0.963, f"Percentile Rank vs {competition} Goalkeepers | Season 2024-25", size=10,
                    ha="left", fontproperties=self.font_bold.prop, color="#000000")

            # Credits
            CREDIT_1 = "@pitchiq.bsky.social\ndata via FBREF / Opta"
            CREDIT_2 = "inspired by: @Worville, @FootballSlices"
            fig.text(0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
                    fontproperties=self.font_italic.prop, color="#000000", ha="right")

            # Logo
            try:
                logo_path = os.path.join(self.image_dir, "piqmain.png")
                ax3 = fig.add_axes([0.80, 0.075, 0.15, 1.75])
                ax3.axis('off')
                img = image.imread(logo_path)
                ax3.imshow(img)
            except FileNotFoundError:
                print("Logo image not found.")

            output_dir = output_dir or f"GK_profiles/{player_name}"
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{player_name}_{gk_style}_percentile_pizza.png")

            plt.savefig(save_path, dpi=500, facecolor="#EBEBE9", bbox_inches="tight", edgecolor="none", transparent=False)

            return save_path
            
        except ImportError:
            print("PyPizza not available. Install mplsoccer for pizza charts.")
            return None

    def generate_full_gk_profile(self, player_name, gk_style, df=None, include_logos=True):
        """Generate complete goalkeeper profile with all visualizations"""
        if df is None:
            raise ValueError("No dataframe provided.")

        self.plot_gk_role_comparison(player_name, gk_style, df)
        self.plot_gk_role_kde(player_name, gk_style, df)
        self.create_gk_percentile_pizza(player_name, gk_style, df)
        return self.combine_gk_profile_plots(player_name, gk_style, df, include_logos)

    def _add_gk_title_text(self, fig, player_name, club, age, nineties, season, gk_style, comparison_group):
        """Add title text to GK plots"""
        from matplotlib import pyplot as plt
        
        plt.figtext(
            x=0.66, y=0.93,
            s=f"{club} | {player_name} (GK)\n"
              f"90's Played: {nineties:.1f} | Age: {age:.1f}\n"
              f"Season: {season}\n"
              f"{gk_style} Template compared to {comparison_group}",
            va="bottom", ha="right",
            fontsize=14, color="black", weight="book"
        )

    def _add_gk_comparison_text(self, fig, player_name, comp_name, main_player_row, gk_style):
        """Add comparison text to GK radar plots"""
        from matplotlib import pyplot as plt
        
        plt.figtext(
            x=0.65, y=0.93,
            s=f"{main_player_row['Squad'].values[0]} | {player_name} vs {comp_name}\n"
              f"Season: 2024–2025\n"
              f"{gk_style} Template - Goalkeeper Comparison",
            va="bottom", ha="right",
            fontsize=14, color="black", weight="book"
        )

    def _add_gk_badge_overlay(self, fig):
        """Add badge overlay to GK plots"""
        try:
            badge_path = os.path.join(self.image_dir, "piqmain.png")
            badge_img = image.imread(badge_path)
            ax3 = fig.add_axes([0.002, 0.89, 0.20, 0.15], zorder=1)
            ax3.axis('off')
            ax3.imshow(badge_img)
        except FileNotFoundError:
            print("Logo or badge image not found, skipping visual extras.")

    def create_gk_kmeans_df(self, df):
        """
        Create K-means clustering for goalkeepers using GK-specific metrics
        """
        print(f"Input dataframe shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")
        
        # GK-specific clustering columns
        GK_KMeans_cols = [
            'Player',
            'Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90',
            'Goals_Against', 
            'Launched_Pass_Completion_Percentage',
            'Average_Pass_Length',
            'Crosses_Stopped_Percentage',
            'Defensive_Actions_Outside_Penalty_Area_Per_90',
            'Average_Distance_Of_Defensive_Actions',
            'Crosses_Stopped',
            'Crosses_Faced',
            'Launched_Passes_Completed',
            'Passes_Attempted_Excluding_Goal_Kicks',
            'Throws_Attempted'
        ]
        
        # Check which columns actually exist
        available_cols = ['Player']
        missing_cols = []
        
        for col in GK_KMeans_cols[1:]:  # Skip 'Player'
            if col in df.columns:
                available_cols.append(col)
            else:
                missing_cols.append(col)
        
        print(f"Available GK columns: {len(available_cols)-1} out of {len(GK_KMeans_cols)-1}")
        print(f"Missing columns: {missing_cols}")
        
        if len(available_cols) < 5:  # Need at least 4 numeric columns + Player
            raise ValueError(f"Not enough columns available for clustering. Available: {available_cols}")
        
        # Filter to only include columns that exist in the dataframe
        df_kmeans = df[available_cols].copy()
        print(f"Filtered dataframe shape: {df_kmeans.shape}")
        
        # Check for non-numeric data
        numeric_cols = available_cols[1:]  # Exclude 'Player'
        for col in numeric_cols:
            non_numeric_count = df_kmeans[col].apply(lambda x: not isinstance(x, (int, float, np.number))).sum()
            if non_numeric_count > 0:
                print(f"Warning: {col} has {non_numeric_count} non-numeric values")
        
        # Store player names
        player_names = df_kmeans['Player'].tolist()
        print(f"Number of players: {len(player_names)}")
        
        # Drop Player column for clustering
        df_features = df_kmeans.drop(['Player'], axis=1)
        
        # Convert to numeric and handle missing values
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
        
        print(f"Features shape before cleaning: {df_features.shape}")
        print(f"Missing values per column:\n{df_features.isnull().sum()}")
        
        # Handle missing values by filling with median
        df_features = df_features.fillna(df_features.median())
        
        # Remove rows with all NaN values (if any remain)
        df_features = df_features.dropna(how='all')
        
        print(f"Features shape after cleaning: {df_features.shape}")
        
        if df_features.empty:
            raise ValueError("No valid data remaining after cleaning. Check your data quality.")
        
        # Update player names list to match cleaned data
        if len(player_names) != len(df_features):
            # Get indices of rows that weren't dropped
            valid_indices = df_features.index
            player_names = [player_names[i] for i in valid_indices if i < len(player_names)]
        
        # Normalize features
        x = df_features.values
        scaler = preprocessing.MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        X_norm = pd.DataFrame(x_scaled)
        
        # PCA reduction
        pca = PCA(n_components=2)
        reduced = pd.DataFrame(pca.fit_transform(X_norm))
        
        # Determine optimal clusters using elbow method
        n_samples = len(reduced)
        max_clusters = min(8, n_samples)  # Don't try more clusters than we have samples
        
        if n_samples < 3:
            raise ValueError(f"Not enough samples for clustering: {n_samples}. Need at least 3.")
        
        wcss = []
        for i in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            kmeans.fit(reduced)
            wcss.append(kmeans.inertia_)
        
        # Use fewer clusters for small datasets
        optimal_clusters = min(5, max(2, n_samples // 5))  # At least 2, at most 5 clusters
        print(f"Using {optimal_clusters} clusters for {n_samples} samples")
        
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        kmeans = kmeans.fit(reduced)
        
        labels = kmeans.predict(reduced)
        clusters = kmeans.labels_.tolist()
        
        reduced['cluster'] = clusters
        reduced['name'] = player_names[:len(reduced)]  # Ensure matching length
        reduced.columns = ['x', 'y', 'cluster', 'name']
        
        print(f"Final kmeans dataframe shape: {reduced.shape}")
        print(f"Cluster distribution:\n{reduced['cluster'].value_counts()}")
        
        return reduced

    def plot_gk_kmeans_clusters(self, df, player_name):
        """
        Create K-means cluster plot for goalkeepers
        """
        print(f"Starting K-means clustering for {player_name}")
        print(f"Input dataframe shape: {df.shape}")
        
        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")
        
        print(f"Found player: {player_name}")
        
        # Filter to goalkeepers with sufficient playing time
        print(f"Filtering for Minutes_90s > 10...")
        df_filtered = df[df['Minutes_90s'] > 10]
        print(f"After filtering: {df_filtered.shape[0]} players remain")
        
        if df_filtered.empty:
            print("No players meet the minimum playing time requirement!")
            print("Trying with lower threshold...")
            df_filtered = df[df['Minutes_90s'] > 5]
            print(f"With Minutes_90s > 5: {df_filtered.shape[0]} players")
            
            if df_filtered.empty:
                print("Still no players! Using all available data...")
                df_filtered = df.copy()
        
        # Check if target player is in filtered data
        target_in_filtered = player_name in df_filtered['Player'].values
        print(f"Target player {player_name} in filtered data: {target_in_filtered}")
        
        if not target_in_filtered:
            print(f"Adding {player_name} back to filtered data...")
            target_row = df[df['Player'] == player_name]
            df_filtered = pd.concat([df_filtered, target_row]).drop_duplicates()
        
        print(f"Final filtered dataframe shape: {df_filtered.shape}")
        
        try:
            filtered_kmeans = self.create_gk_kmeans_df(df_filtered)
        except Exception as e:
            print(f"Error in create_gk_kmeans_df: {e}")
            raise
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create scatter plot with clusters
        unique_clusters = filtered_kmeans['cluster'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster in enumerate(unique_clusters):
            cluster_data = filtered_kmeans[filtered_kmeans['cluster'] == cluster]
            ax.scatter(cluster_data['x'], cluster_data['y'], 
                      c=[colors[i]], s=250, alpha=0.7, 
                      label=f'Cluster {cluster}')
        
        # Add player names as labels
        texts = []
        for x, y, s in zip(filtered_kmeans.x, filtered_kmeans.y, filtered_kmeans.name):
            texts.append(ax.text(x, y, s, fontweight='light', fontsize=8))
        
        # Highlight the target player
        target_player_data = filtered_kmeans[filtered_kmeans['name'] == player_name]
        if not target_player_data.empty:
            ax.scatter(target_player_data['x'], target_player_data['y'], 
                      c='red', s=400, marker='*', 
                      label=f'{player_name} (Target)', zorder=5)
        else:
            print(f"Warning: {player_name} not found in clustering results")
        
        # Styling
        ax.set_title(f'KMeans Clustering - Goalkeepers\n{player_name} Analysis', 
                    fontsize=20, pad=20)
        ax.set_xlabel('PC 1', fontsize=16)
        ax.set_ylabel('PC 2', fontsize=16)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        
        # Add logos and credits
        try:
            ax2 = fig.add_axes([0.01, 0.9, 0.08, 0.12])
            ax2.axis('off')
            img1 = image.imread(os.path.join(self.image_dir, "premier-league-2-logo.png"))
            ax2.imshow(img1)
            
            ax3 = fig.add_axes([0.85, 0.9, 0.08, 0.12])
            ax3.axis('off')
            img2 = image.imread(os.path.join(self.image_dir, "piqmain.png"))
            ax3.imshow(img2)
        except FileNotFoundError:
            print("Logo images not found.")
        
        fig.text(0.02, 0.02, '24/25 Season | Viz by @pitchiq | Data via FBREF', 
                size=12, style='italic')
        
        plt.tight_layout()
        
        # Save the plot
        player_dir = f"GK_profiles/{player_name}"
        os.makedirs(player_dir, exist_ok=True)
        plt.savefig(f"{player_dir}/{player_name}_kmeans_clusters.png", 
                   dpi=300, bbox_inches='tight')
        
        return filtered_kmeans

    def debug_gk_data(self, gk_df, player_name="Kepa Arrizabalaga"):
        """
        Debug function to check data quality before running similarity analysis
        """
        print("=== GOALKEEPER DATA DEBUG ===")
        print(f"Total records: {len(gk_df)}")
        print(f"Columns: {len(gk_df.columns)}")
        print(f"\nLooking for player: {player_name}")
        
        # Check if player exists
        player_exists = player_name in gk_df['Player'].values
        print(f"Player found: {player_exists}")
        
        if player_exists:
            player_data = gk_df[gk_df['Player'] == player_name]
            print(f"Player Minutes_90s: {player_data['Minutes_90s'].values[0]}")
            print(f"Player Squad: {player_data['Squad'].values[0]}")
        
        # Check Minutes_90s distribution
        print(f"\nMinutes_90s distribution:")
        print(f"  > 10: {len(gk_df[gk_df['Minutes_90s'] > 10])} players")
        print(f"  > 5: {len(gk_df[gk_df['Minutes_90s'] > 5])} players")
        print(f"  > 1: {len(gk_df[gk_df['Minutes_90s'] > 1])} players")
        print(f"  Min: {gk_df['Minutes_90s'].min()}")
        print(f"  Max: {gk_df['Minutes_90s'].max()}")
        print(f"  Mean: {gk_df['Minutes_90s'].mean():.2f}")
        
        # Check required columns for K-means
        required_cols = [
            'Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90',
            'Goals_Against', 
            'Launched_Pass_Completion_Percentage',
            'Average_Pass_Length',
            'Crosses_Stopped_Percentage',
            'Defensive_Actions_Outside_Penalty_Area_Per_90'
        ]
        
        print(f"\nChecking required columns:")
        missing_cols = []
        for col in required_cols:
            exists = col in gk_df.columns
            print(f"  {col}: {'✓' if exists else '✗'}")
            if not exists:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"\nMissing columns: {missing_cols}")
            print("Available columns that might work:")
            for col in gk_df.columns:
                if any(keyword in col.lower() for keyword in ['goal', 'pass', 'cross', 'defensive']):
                    print(f"  - {col}")
        
        # Check data quality
        print(f"\nData quality check:")
        numeric_cols = gk_df.select_dtypes(include=[np.number]).columns
        print(f"Numeric columns: {len(numeric_cols)}")
        
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            null_count = gk_df[col].isnull().sum()
            print(f"  {col}: {null_count} nulls out of {len(gk_df)}")
        
        return gk_df[gk_df['Minutes_90s'] > 10] if len(gk_df[gk_df['Minutes_90s'] > 10]) > 0 else gk_df

    def find_similar_gks_and_scores(self, player_name, kmeans_df, gk_scores_df, max_age=35, top_n=20):
        """
        Find similar goalkeepers based on K-means clustering
        """
        player = kmeans_df[kmeans_df['name'] == player_name].iloc[0]
        kmeans_df['distance'] = np.sqrt((kmeans_df['x'] - player['x'])**2 + 
                                       (kmeans_df['y'] - player['y'])**2)
        max_distance = kmeans_df['distance'].max()
        kmeans_df['perc_similarity'] = (((max_distance - kmeans_df['distance']) / max_distance) * 100) * 0.90
        
        similar_players = kmeans_df.sort_values('distance').head(top_n + 1)[1:]  # Exclude the player themselves
        similarity_table = similar_players[['name', 'perc_similarity']].rename(columns={'name': 'Player'})
        
        metrics_similarity = pd.merge(similarity_table, gk_scores_df, on='Player', how='left')
        metrics_similarity = metrics_similarity.drop_duplicates(subset=['Player'])
        
        # Handle Age column safely
        if 'Age' in metrics_similarity.columns:
            metrics_similarity['Age'] = pd.to_numeric(metrics_similarity['Age'], errors='coerce')
            metrics_similarity = metrics_similarity[metrics_similarity['Age'] < max_age]
        else:
            print("Warning: 'Age' column not found, skipping age filter")
        
        comparative_list = list(metrics_similarity.Player.unique())
        sim_index = [round(item, 2) for item in metrics_similarity.perc_similarity.unique()]
        
        return metrics_similarity, comparative_list, sim_index

    def plot_gk_comparison_radars(self, player_name, gk_style, comparative_list, gk_df, metrics_similarity, sim_index):
        """
        Plot comparison radar charts for similar goalkeepers
        """
        params = self.gk_role_templates[gk_style]['key_stats']
        readable_params = [self.gk_stat_mapping.get(p, p) for p in params]
        
        def get_gk_data(df, player_name, params):
            player_data = df[df['Player'] == player_name][params].values.tolist()
            return [val for sublist in player_data for val in sublist]
        
        main_player = get_gk_data(gk_df, player_name, params)
        comp_players = [get_gk_data(metrics_similarity, comp, params) for comp in comparative_list]
        
        def convert_to_numeric(input_list):
            return [float(x) for x in input_list]
        
        numeric_main_player = convert_to_numeric(main_player)
        
        for idx, comp_player in enumerate(comp_players):
            if idx >= len(sim_index):  # Safety check
                break
                
            numeric_comp_player = convert_to_numeric(comp_player)
            
            low = [min(value, value_2) * 0.5 for value, value_2 in zip(numeric_main_player, numeric_comp_player)]
            high = [max(value, value_2) * 1.05 for value, value_2 in zip(numeric_main_player, numeric_comp_player)]
            
            radar = Radar(readable_params, low, high,
                         round_int=[False]*len(params),
                         num_rings=5,
                         ring_width=1, center_circle_radius=1)
            
            fig, ax = radar.setup_axis()
            fig.patch.set_facecolor('#f0f0f0')
            ax.set_facecolor('#f0f0f0')
            
            radar.draw_circles(ax=ax, facecolor='#ffb2b2')
            radar.draw_radar_compare(numeric_main_player, numeric_comp_player, ax=ax,
                                   kwargs_radar={'facecolor': '#228B22', 'alpha': 0.6},
                                   kwargs_compare={'facecolor': '#FF6B35', 'alpha': 0.6})
            
            radar.draw_range_labels(ax=ax, fontsize=15)
            radar.draw_param_labels(ax=ax, fontsize=15)
            
            ax.legend([f'{player_name} (GK)', f'{comparative_list[idx]} (GK)'], 
                     loc='upper right', fontsize=12)
            
            # Add title with similarity score
            plt.figtext(
                x=0.57, y=0.90,
                s=f"{player_name} vs {comparative_list[idx]} (GKs)\n"
                  f"Season 2024/2025\n"
                  f"GK Similarity Score: {sim_index[idx]}%\n"
                  f"Style: {gk_style}",
                va="bottom", ha="right",
                fontsize=15, color="black", weight="book"
            )
            
            # Add logos
            try:
                ax3 = fig.add_axes([0.80, 0.09, 0.13, 0.15])
                ax3.axis('off')
                img = image.imread(os.path.join(self.image_dir, "piqmain.png"))
                ax3.imshow(img)
            except FileNotFoundError:
                pass
            
            # Save comparison radar
            output_dir = f"GK_profiles/{player_name}/compare_radars/"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/{player_name}_vs_{comparative_list[idx]}_gk.png', 
                       bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
            
            plt.close()  # Close to prevent memory issues



    def generate_gk_similarity_score_card(self, player_name, mertrics_similarity, Cannoniq_DB):
        mertrics_similarity = mertrics_similarity.rename(columns={'Squad': 'team'})
        fm_ids = pd.read_csv("/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/Top6_leagues_fotmob_ids.csv")
        fm_ids = fm_ids[["team", "team_id"]]

        mertrics_similarity = mertrics_similarity.merge(fm_ids, on='team', how='left')
        mertrics_similarity = mertrics_similarity.dropna(subset=['team_id'])
        mertrics_similarity['team_id'] = mertrics_similarity['team_id'].astype(float).astype(int)

        mertrics_similarity[['perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']] = \
            mertrics_similarity[['perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']].round(2)

        mertrics_similarity["Pos"] = "GK"  
        df_final = mertrics_similarity[['Player', 'Pos','team_id','perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']]
        metric_scores =['Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']

        sim_player_vals = Cannoniq_DB[Cannoniq_DB['Player'] == player_name][metric_scores].values.tolist()
        sim_player_vals = [val for sublist in sim_player_vals for val in sublist]

        df_final['Δ% Shot Stopping'] = ((df_final['Shot_Stopping_Score'] - sim_player_vals[0]) / sim_player_vals[0]).round(1) * 100
        df_final['Δ% Distribution'] = ((df_final['Distribution_Score'] - sim_player_vals[1]) / sim_player_vals[1]).round(1) * 100
        df_final['Δ% Claiming'] = ((df_final['Claiming_Score'] - sim_player_vals[2]) / sim_player_vals[2]).round(1) * 100
        df_final['Δ% Sweeping'] = ((df_final['Sweeping_Score'] - sim_player_vals[3]) / sim_player_vals[3]).round(1) * 100
        df_final = df_final[::-1]

        def perc_battery(perc_similarity, ax):
            '''
            This function takes an integer and an axes and 
            plots a battery chart.
            '''
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.barh([0.5], [1], fc = 'white', ec='black', height=.35)
            ax.barh([0.5], [perc_similarity/100], fc = '#00529F', height=.35)
            text_ = ax.annotate(
                xy=((perc_similarity/100), .5),
                text=f'{(perc_similarity/100):.0%}',
                xytext=(-8,0),
                textcoords='offset points',
                weight='bold',
                color='#EFE9E6',
                va='center',
                ha='center',
                size=8
            )
            ax.set_axis_off()
            return ax

        def ax_logo(team_id, ax):
            '''
            Plots the logo of the team at a specific axes.
            Args:
                team_id (int): the id of the team according to Fotmob. You can find it in the url of the team page.
                ax (object): the matplotlib axes where we'll draw the image.
            '''
            fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
            club_icon = Image.open(urllib.request.urlopen(f'{fotmob_url}{team_id}.png'))
            ax.imshow(club_icon)
            ax.axis('off')
            return ax

        fig = plt.figure(figsize=(17,17), dpi=400)
        ax = plt.subplot()

        ncols = 12
        nrows = df_final.shape[0]

        ax.set_xlim(0, ncols + 1)
        ax.set_ylim(0, nrows + 1)

        positions = [0.25, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
        columns = ['Player', 'Pos', 'perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score', 'Δ% Shot Stopping', 'Δ% Distribution', 'Δ% Claiming', 'Δ% Sweeping']

        # -- Add table's main text
        for i in range(nrows):
            for j, column in enumerate(columns):
                if j == 0:
                    ha = 'left'
                else:
                    ha = 'center'
                if column == 'perc_similarity':
                    continue
                else:
                    text_label = f'{df_final[column].iloc[i]}'
                    weight = 'normal'
                ax.annotate(
                    xy=(positions[j], i + .5),
                    text=text_label,
                    ha=ha,
                    va='center',
                    size = 10,
                    weight=weight
                )

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
        # -- Add nation axes
        ax_point_1 = DC_to_NFC([2.25, 0.25])
        ax_point_2 = DC_to_NFC([2.75, 0.75])
        ax_width = abs(ax_point_1[0] - ax_point_2[0])
        ax_height = abs(ax_point_1[1] - ax_point_2[1])
        for x in range(0, nrows):
            ax_coords = DC_to_NFC([2.25, x + .25])
            flag_ax = fig.add_axes(
                [ax_coords[0], ax_coords[1], ax_width, ax_height]
            )
            ax_logo(df_final['team_id'].iloc[x], flag_ax)

        ax_point_1 = DC_to_NFC([4, 0.05])
        ax_point_2 = DC_to_NFC([5, 0.95])
        ax_width = abs(ax_point_1[0] - ax_point_2[0])
        ax_height = abs(ax_point_1[1] - ax_point_2[1])
        for x in range(0, nrows):
            ax_coords = DC_to_NFC([4, x + .025])
            bar_ax = fig.add_axes(
                [ax_coords[0], ax_coords[1], ax_width, ax_height]
            )
            perc_battery(df_final['perc_similarity'].iloc[x], bar_ax)

        # -- Add column names
        column_names = ['Player', 'Position', 'Percent\nSimilarity','Shot\nStopping', 'Distribution', 'Claiming', 'Sweeping','Δ%\nShot Stopping','Δ%\nDistribution','Δ%\nClaiming','Δ%\nSweeping']
        for index, c in enumerate(column_names):
                if index == 0:
                    ha = 'left'
                else:
                    ha = 'center'
                ax.annotate(
                    xy=(positions[index], nrows + .25),
                    text=column_names[index],
                    ha=ha,
                    va='bottom',
                    size = 12,
                    weight='book'
                )

        # Add dividing lines
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
        for x in range(1, nrows):
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

        ax.fill_between(
            x=[0,2],
            y1=nrows,
            y2=0,
            color='lightgrey',
            alpha=0.5,
            ec='None'
        )

        # Custom colormap and normalization
        cmap = mcolors.LinearSegmentedColormap.from_list('red_green', ['red', 'white', 'green'])
        norm = mcolors.Normalize(vmin=-50, vmax=50)

        # Example of Δ% columns to be visualized
        delta_columns = ['Δ% Shot Stopping', 'Δ% Distribution', 'Δ% Claiming', 'Δ% Sweeping']

        # Loop through delta columns and fill between with corresponding color
        for idx, col in enumerate(delta_columns):
            for i in range(nrows):
                value = df_final[col].iloc[i]
                ax.fill_between(
                    x=[9 + idx, 10 + idx],
                    y1=i + 1,
                    y2=i,
                    color=cmap(norm(value)),
                    alpha=0.6,
                    ec='None'
                )

        ax.set_axis_off()

        # -- Final details
        league_icon = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images//premier-league-2-logo.png")
        league_ax = fig.add_axes([0.06, 0.88, 0.10, 0.10], zorder=1)
        league_ax.imshow(league_icon)
        league_ax.axis("off")

        ax.tick_params(axis='both', which='major', labelsize=8)

        fig_text(
            x = 0.5, y = 0.95, 
            s = f'{player_name} - PIQ Similarity Model Results 24/25',
            va = "bottom", ha = "center",
            fontsize = 17, color = "black", weight = "bold"
        )

        fig_text(
            x = 0.5, y = 0.92, 
            s = f'Shot Stopping, Distribution, Claiming & Sweeping scores generated via weighted aggregated metrics from FBREF (Out of 10)',
            va = "bottom", ha = "center",
            fontsize = 12, color = "black", weight = "book"
        )
        fig_text(
            x = 0.5, y = 0.90, 
            s = f'Δ% columns are PIQ score percentage change, -ve% means {player_name} has a better score and vice versa',
            va = "bottom", ha = "center",
            fontsize = 12, color = "black", weight = "book"
        )

        ### Add Stats by Steve logo
        ax3 = fig.add_axes([0.83, 0.13, 0.09, 1.60])
        ax3.axis('off')
        img = plt.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
        ax3.imshow(img)
  
        output_path = f"GK_profiles/{player_name}/scorecards/"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_path = os.path.join(output_path, f"{player_name}_similarity_score_card.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
        
        plt.show()

        return df_final



    def generate_full_gk_similarity_analysis(self, player_name, gk_style, gk_df, max_age=35, top_n=25):
        """
        Generate complete goalkeeper similarity analysis
        """
        # Step 1: K-means clustering
        kmeans_df = self.plot_gk_kmeans_clusters(gk_df, player_name)
        
        # Step 2: Find similar goalkeepers
        metrics_similarity, comparative_list, sim_index = self.find_similar_gks_and_scores(
            player_name, kmeans_df, gk_df, max_age=max_age, top_n=top_n
        )
        
        # Step 3: Plot comparison radars
        self.plot_gk_comparison_radars(
            player_name, gk_style, comparative_list, gk_df, metrics_similarity, sim_index
        )
        
        # Step 4: Generate similarity score card
        df_final = self.generate_gk_similarity_score_card(
            player_name, metrics_similarity, gk_df
        )
        
        print(f"✓ Complete GK similarity analysis generated for {player_name}")
        print(f"  - K-means cluster plot")
        print(f"  - {len(comparative_list)} comparison radar charts")
        print(f"  - Similarity score card")
        
        return df_final


# Usage example:
def setup_gk_plotter_with_templates():
    """Setup the GK plotter with role templates"""
    
    # Define GK role templates (using your existing templates)
    gk_role_templates = {
        "Classic Shot-Stopper": {
            "description": "Traditional goalkeeper focused on shot-stopping and basic distribution",
            "key_stats": [
                "Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90",
                "Goals_Against",
                "Crosses_Stopped_Percentage",
                "Launched_Pass_Completion_Percentage",
                "Crosses_Stopped",
                "Crosses_Faced",
                "Average_Pass_Length"
            ]
        },
        
        "Sweeper-Keeper": {
            "description": "Modern goalkeeper who acts as an extra defender",
            "key_stats": [
                "Defensive_Actions_Outside_Penalty_Area_Per_90",
                "Average_Distance_Of_Defensive_Actions",
                "Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90",
                "Launched_Pass_Completion_Percentage",
                "Average_Pass_Length",
                "Passes_Attempted_Excluding_Goal_Kicks",
                "Goals_Against"
            ]
        },
        
        "Ball-Playing Goalkeeper": {
            "description": "Goalkeeper excellent at distribution and building play",
            "key_stats": [
                "Launched_Pass_Completion_Percentage",
                "Average_Pass_Length",
                "Passes_Attempted_Excluding_Goal_Kicks",
                "Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90",
                "Throws_Attempted",
                "Launched_Passes_Completed",
                "Launched_Passes_Attempted"
            ]
        },
        
        "Complete Goalkeeper": {
            "description": "Well-rounded goalkeeper excelling in all areas",
            "key_stats": [
                "Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90",
                "Launched_Pass_Completion_Percentage",
                "Crosses_Stopped_Percentage",
                "Defensive_Actions_Outside_Penalty_Area_Per_90",
                "Goals_Against",
                "Average_Pass_Length",
                "Crosses_Stopped"
            ]
        },
        
        "Penalty Box Guardian": {
            "description": "Goalkeeper who dominates the penalty area",
            "key_stats": [
                "Crosses_Stopped_Percentage",
                "Crosses_Stopped",
                "Post_Shot_Expected_Goals_Minus_Goals_Allowed_Per_90",
                "Goals_Against",
                "Crosses_Faced",
                "Launched_Pass_Completion_Percentage",
                "Average_Pass_Length"
            ]
        }
    }
    
    return GoalkeeperPlotter(gk_role_templates)

# Example usage:
"""
# Setup the plotter
gk_plotter = setup_gk_plotter_with_templates()

# Generate full profile for a goalkeeper
gk_plotter.generate_full_gk_profile(
    player_name="Alisson", 
    gk_style="Complete Goalkeeper", 
    df=your_gk_dataframe,
    include_logos=True
)

# Or create individual plots
gk_plotter.plot_gk_role_comparison(
    player_name="Alisson",
    gk_style="Complete Goalkeeper", 
    df=your_gk_dataframe,
    comparative_list=["Ederson", "Hugo Lloris"]
)
"""