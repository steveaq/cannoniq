# Standard libraries
import os
import sys
import math
import warnings
import unicodedata
from math import pi
from urllib.request import urlopen
import time

# Data handling
import pandas as pd
import numpy as np

# Web scraping
import requests
from bs4 import BeautifulSoup

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.image as image
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
from matplotlib import cm
from matplotlib.patheffects import withStroke
import seaborn as sns
import matplotlib.style as style

# Highlighting and annotation
from highlight_text import fig_text
from adjustText import adjust_text

# Image handling
from PIL import Image
import urllib

# Stats & clustering
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# String matching
from fuzzywuzzy import fuzz, process

# Charting
from mplsoccer import PyPizza, add_image, FontManager, Radar

# Tabular display
from tabulate import tabulate

# Suppress warnings
warnings.filterwarnings('ignore')

# Plot style
style.use('fivethirtyeight')

# Ensure pandas doesn't truncate wide tables
pd.options.display.max_columns = None


# Local import
module_path = os.path.abspath(os.path.join('/Users/stephenahiabah/Desktop/Code/cannoniq'))
if module_path not in sys.path:
    sys.path.append(module_path)

import FBREF_Aggregations as fbref


class CannoniqCharts:
    def __init__(self, data_dir='/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/',
                 image_dir='/Users/stephenahiabah/Desktop/Code/cannoniq/Images/',
                 player_role_templates=None,
                 clean_stat_mapping=None):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.player_role_templates = player_role_templates or {}
        self.clean_stat_mapping = clean_stat_mapping or {}
        self.db = None

        self.font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf')
        self.font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Italic.ttf')
        self.font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf')


    def create_cannoniq_database(self, phonebook_filename='24-25player_phonebook.csv',
                                  output_filename='CannonIQ_DB.csv'):
        phonebook_path = os.path.join(self.data_dir, phonebook_filename)
        output_path = os.path.join(self.data_dir, output_filename)

        fb_ref_db = fbref.CreateFBRefDatabase()
        CannonIQ_DB = fb_ref_db.generate_pitch_iq_scores()

        player_phonebook = pd.read_csv(phonebook_path)
        CannonIQ_DB['player_name_match'] = CannonIQ_DB['Player'].apply(fb_ref_db.remove_accents)
        player_phonebook = player_phonebook.rename(columns={'Player': 'player_name_match'})
        player_phonebook = player_phonebook.drop_duplicates(subset='player_name_match', keep='first')

        CannonIQ_DB = pd.merge(
            CannonIQ_DB,
            player_phonebook[['player_name_match', 'scouting_url', 'match_logs']],
            on='player_name_match',
            how='left'
        )
        time.sleep(100)
        full_stats = fb_ref_db.create_full_stats_db()
        full_stats = fb_ref_db.per_90fi(full_stats)

        CannonIQ_DB = CannonIQ_DB.drop_duplicates(subset=['Player'], keep='first')
        CannonIQ_DB = CannonIQ_DB[[
            'Player', 'Nation', 'Pos', 'Squad', 'position_group',
            'Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score',
            'player_name_match', 'scouting_url', 'match_logs'
        ]]

        CannonIQ_DB = pd.merge(
            CannonIQ_DB,
            full_stats,
            on=['Player', 'Nation', 'Pos', 'Squad'],
            how='inner'
        )

        columns_to_drop = [col for col in CannonIQ_DB.columns if col.endswith('_y')]
        CannonIQ_DB = CannonIQ_DB.drop(columns=columns_to_drop)
        CannonIQ_DB.columns = [col.replace('_x', '') for col in CannonIQ_DB.columns]
        CannonIQ_DB = CannonIQ_DB.drop(columns=['TO'], errors='ignore')

        CannonIQ_DB.to_csv(output_path, index=False)
        print(f'CannonIQ_DB saved to {output_path}')

        self.db = CannonIQ_DB
        return self.db
    
    def plot_role_based_comparison(self, player_name, role, df=None, comparative_list=None):
        plt.style.use("fivethirtyeight")

        if role not in self.player_role_templates:
            raise ValueError(f"Unknown role '{role}'.")

        # df = df or self.db
        if df is None:
            raise ValueError("No dataframe provided or available in the class.")

        player_dir = f"Player_profiles/{player_name}"
        os.makedirs(player_dir, exist_ok=True)

        params = self.player_role_templates[role]
        readable_params = [self.clean_stat_mapping.get(p, p) for p in params]

        main_player_row = df[df['Player'] == player_name]
        if main_player_row.empty:
            raise ValueError(f"{player_name} not found in dataset")

        pos_group = main_player_row['position_group'].values[0]
        scaled_df = df[(df['position_group'] == pos_group) & (df['90s'] > 10)]

        def get_player_data(name):
            row = df[df['Player'] == name]
            if row.empty:
                raise ValueError(f"{name} not found in dataset")
            return row[params].values.flatten().tolist()

        main_values = [float(x) for x in get_player_data(player_name)]
        comp_values_list = [[float(x) for x in get_player_data(name)] for name in comparative_list] if comparative_list else []

        # --- Single Radar ---
        if not comp_values_list:
            low = [0] * len(params)
            high = scaled_df[params].max().tolist()

            radar = Radar(readable_params, low, high, round_int=[False]*len(params),
                          num_rings=5, ring_width=1, center_circle_radius=1)
            fig, ax = radar.setup_axis()
            fig.patch.set_facecolor('#ededed')
            ax.set_facecolor('#ededed')
            radar.draw_circles(ax=ax, facecolor='#dddddd')
            radar.draw_radar(main_values, ax=ax, kwargs_radar={'facecolor': '#aaaaaa', 'alpha': 0.65})
            radar.draw_range_labels(ax=ax, fontsize=15)
            radar.draw_param_labels(ax=ax, fontsize=15)
            ax.legend([player_name], loc='upper right', fontsize=12)

            club = main_player_row['Squad'].values[0]
            age = float(main_player_row['Age'].values[0])
            nineties = float(main_player_row["90s"].values[0])
            season = "2024–2025"
            role_label = role.replace("_", " ").title()

            fig.savefig(f"{player_dir}/{player_name}_radar.png", dpi=300, bbox_inches="tight")

            fig_text(
                x=0.66, y=0.93,
                s=f"{club} | {player_name}\n"
                  f"90's Played: {nineties:.1f} | Age: {age:.1f}\n"
                  f"Season: {season}\n"
                  f"{role} Template compared to {pos_group}",
                va="bottom", ha="right",
                fontsize=14, color="black", weight="book"
            )

        # --- Comparison Radar ---
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
                                         kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                         kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
                radar.draw_range_labels(ax=ax, fontsize=15)
                radar.draw_param_labels(ax=ax, fontsize=15)
                ax.legend([player_name, comparative_list[i]], loc='upper right', fontsize=12)
                club = main_player_row['Squad'].values[0]
                age = float(main_player_row['Age'].values[0])
                nineties = float(main_player_row["90s"].values[0])
                season = "2024–2025"
                role_label = role.replace("_", " ").title()

                comp_name = comparative_list[i]
                fig.savefig(f"{player_dir}/{player_name}_vs_{comp_name}_radar.png", dpi=300, bbox_inches="tight")

                fig_text(
                    x=0.65, y=0.93,
                    s=f"{main_player_row['Squad'].values[0]} | {player_name} vs {comp_name}\n"
                      f"Season: 2024–2025\n"
                      f"{role} Template compared to {pos_group}s",
                    va="bottom", ha="right",
                    fontsize=14, color="black", weight="book"
                )

        # --- Badge Overlay ---
        try:
            badge_path = os.path.join(self.image_dir, "piqmain.png")
            badge_img = image.imread(badge_path)
            ax3 = fig.add_axes([0.002, 0.89, 0.20, 0.15], zorder=1)
            ax3.axis('off')
            ax3.imshow(badge_img)
        except FileNotFoundError:
            print("Logo or badge image not found, skipping visual extras.")

        plt.show()

    def plot_role_based_comparison(self, player_name, role, df=None, comparative_list=None):
        plt.style.use("fivethirtyeight")

        if role not in self.player_role_templates:
            raise ValueError(f"Unknown role '{role}'.")

        # df = df or self.db
        if df is None:
            raise ValueError("No dataframe provided or available in the class.")

        player_dir = f"Player_profiles/{player_name}"
        os.makedirs(player_dir, exist_ok=True)

        params = self.player_role_templates[role]
        readable_params = [self.clean_stat_mapping.get(p, p) for p in params]

        main_player_row = df[df['Player'] == player_name]
        if main_player_row.empty:
            raise ValueError(f"{player_name} not found in dataset")

        pos_group = main_player_row['position_group'].values[0]
        scaled_df = df[(df['position_group'] == pos_group) & (df['90s'] > 10)]

        def get_player_data(name):
            row = df[df['Player'] == name]
            if row.empty:
                raise ValueError(f"{name} not found in dataset")
            return row[params].values.flatten().tolist()

        main_values = [float(x) for x in get_player_data(player_name)]
        comp_values_list = [[float(x) for x in get_player_data(name)] for name in comparative_list] if comparative_list else []

        # --- Single Radar ---
        if not comp_values_list:
            low = [0] * len(params)
            high = scaled_df[params].max().tolist()

            radar = Radar(readable_params, low, high, round_int=[False]*len(params),
                          num_rings=5, ring_width=1, center_circle_radius=1)
            fig, ax = radar.setup_axis()
            fig.patch.set_facecolor('#ededed')
            ax.set_facecolor('#ededed')
            radar.draw_circles(ax=ax, facecolor='#dddddd')
            radar.draw_radar(main_values, ax=ax, kwargs_radar={'facecolor': '#aaaaaa', 'alpha': 0.65})
            radar.draw_range_labels(ax=ax, fontsize=15)
            radar.draw_param_labels(ax=ax, fontsize=15)
            ax.legend([player_name], loc='upper right', fontsize=12)

            club = main_player_row['Squad'].values[0]
            age = float(main_player_row['Age'].values[0])
            nineties = float(main_player_row["90s"].values[0])
            season = "2024–2025"
            role_label = role.replace("_", " ").title()

            fig.savefig(f"{player_dir}/{player_name}_radar.png", dpi=300, bbox_inches="tight")

            fig_text(
                x=0.66, y=0.93,
                s=f"{club} | {player_name}\n"
                  f"90's Played: {nineties:.1f} | Age: {age:.1f}\n"
                  f"Season: {season}\n"
                  f"{role_label} Template compared to {pos_group}",
                va="bottom", ha="right",
                fontsize=14, color="black", weight="book"
            )

        # --- Comparison Radar ---
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
                                         kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                         kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
                radar.draw_range_labels(ax=ax, fontsize=15)
                radar.draw_param_labels(ax=ax, fontsize=15)
                ax.legend([player_name, comparative_list[i]], loc='upper right', fontsize=12)

                comp_name = comparative_list[i]
                club = main_player_row['Squad'].values[0]
                age = float(main_player_row['Age'].values[0])
                nineties = float(main_player_row["90s"].values[0])
                season = "2024–2025"
                role_label = role.replace("_", " ").title()

                fig.savefig(f"{player_dir}/{player_name}_vs_{comp_name}_radar.png", dpi=300, bbox_inches="tight")

                fig_text(
                    x=0.65, y=0.93,
                    s=f"{main_player_row['Squad'].values[0]} | {player_name} vs {comp_name}\n"
                      f"Season: 2024–2025\n"
                      f"{role_label} Template compared to {pos_group}s",
                    va="bottom", ha="right",
                    fontsize=14, color="black", weight="book"
                )

        # --- Badge Overlay ---
        try:
            badge_path = os.path.join(self.image_dir, "piqmain.png")
            badge_img = image.imread(badge_path)
            ax3 = fig.add_axes([0.002, 0.89, 0.20, 0.15], zorder=1)
            ax3.axis('off')
            ax3.imshow(badge_img)
        except FileNotFoundError:
            print("Logo or badge image not found, skipping visual extras.")

        plt.show()

    def plot_role_based_kde(self, player_name, role, df=None):
        if role not in self.player_role_templates:
            raise ValueError(f"Invalid role '{role}'.")

        # df = df or self.db
        if df is None:
            raise ValueError("No dataframe provided or available in the class.")

        params = self.player_role_templates[role]
        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        pos_group = player_row['position_group'].values[0]
        filtered_df = df[(df['position_group'] == pos_group) & (df['90s'] > 10)]

        num_params = len(params)
        fig, axs = plt.subplots(num_params, 1, figsize=(7, num_params * 1.2))
        plt.subplots_adjust(hspace=0.8)

        for i, param in enumerate(params):
            ax = axs[i]

            data = pd.to_numeric(filtered_df[param], errors='coerce').dropna().astype(np.float64)
            player_val = float(player_row[param].values[0])
            percentile = (data < player_val).mean() * 100

            # Pad left to start KDE from 0
            data = np.concatenate(([0], data))

            # KDE lines
            sns.kdeplot(data, color="gray", ax=ax, linewidth=1)
            sns.kdeplot(data, fill=True, alpha=0.35, color="black", ax=ax, linewidth=0,
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
            clean_label = self.clean_stat_mapping.get(param, param)
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

        # Directory setup
        player_dir = f"Player_profiles/{player_name}"
        os.makedirs(player_dir, exist_ok=True)

        # Save KDE plot
        fig.savefig(f"{player_dir}/{player_name}_kde.png", dpi=300, bbox_inches="tight")

        plt.show()



    def combine_player_profile_plots(self, player_name, df=None, include_logos=True):
        # df = df or self.db
        if df is None:
            raise ValueError("No dataframe provided or available in the class.")

        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        club = player_row['Squad'].values[0]
        age = float(player_row['Age'].values[0])
        nineties = float(player_row["90s"].values[0])
        season = "2024–2025"
        pos_group = player_row['position_group'].values[0]

        # Paths
        radar_path = f"Player_profiles/{player_name}/{player_name}_radar.png"
        kde_path = f"Player_profiles/{player_name}/{player_name}_kde.png"
        output_path = f"Player_profiles/{player_name}/{player_name}_combined_profile.png"

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

        # Logos
        if include_logos:
            try:
                league_icon = Image.open(os.path.join(self.image_dir, "premier-league-2-logo.png"))
                badge_img = Image.open(os.path.join(self.image_dir, "piqmain.png"))

                ax_league = fig.add_axes([0.15, 0.88, 0.12, 0.18])
                ax_league.imshow(league_icon)
                ax_league.axis('off')

                ax_badge = fig.add_axes([0.75, 0.88, 0.12, 0.18])
                ax_badge.imshow(badge_img)
                ax_badge.axis('off')
            except FileNotFoundError:
                print("Logos not found – skipping logos.")

        # Title + credits
        title = (
            f"{club} | {player_name}\n"
            f"90's Played: {nineties:.1f} | Age: {age:.1f}\n"
            f"Season: {season}\n"
            f"Template compared to {pos_group}s"
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

    def generate_full_player_profile(self, player_name, role, df=None, include_logos=True):
        # df = df or self.db
        if df is None:
            raise ValueError("No dataframe provided or available in the class.")

        self.plot_role_based_comparison(player_name, role, df)
        self.plot_role_based_kde(player_name, role, df)
        return self.combine_player_profile_plots(player_name, df, include_logos)

    def create_percentile_pizza(self, player_name, role, df=None, output_dir=None):
        # df = df or self.db
        if df is None:
            raise ValueError("No dataframe provided or available in the class.")

        if role not in self.player_role_templates:
            raise ValueError(f"Invalid role '{role}'.")

        params = self.player_role_templates[role]
        readable_params = [self.clean_stat_mapping.get(p, p) for p in params]

        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        pos_group = player_row['position_group'].values[0]
        filtered_df = df[(df['position_group'] == pos_group) & (df['90s'] > 10)]

        percentiles = []
        for param in params:
            data = pd.to_numeric(filtered_df[param], errors='coerce').dropna().astype(np.float64)
            player_val = float(player_row[param].values[0])
            percentile = round((data < player_val).mean() * 100)
            percentiles.append(percentile)

        # Slice colors by role type
        if "cb" in role.lower():
            slice_colors = ["#8B0000"] * 5 + ["#B22222"] * 5 + ["#DC143C"] * 5
        elif any(pos in role.lower() for pos in ["cm", "dm"]):
            slice_colors = ["#097969"] * 5 + ["#AFE1AF"] * 5 + ["#088F8F"] * 5
        elif any(pos in role.lower() for pos in ["am", "winger"]):
            slice_colors = ["#00008B"] * 5 + ["#4169E1"] * 5 + ["#87CEFA"] * 5
        elif any(pos in role.lower() for pos in ["f9", "forward"]):
            slice_colors = ["#B0B0B0"] * 5 + ["#808080"] * 5 + ["#404040"] * 5
        else:
            slice_colors = ["#D70232"] * 5 + ["#FF9300"] * 5 + ["#1A78CF"] * 5

        text_colors = ["#000000"] * 15

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
                bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1)
            )
        )

        club = player_row['Squad'].values[0]
        age = float(player_row['Age'].values[0])
        nineties = float(player_row['90s'].values[0])

        fig.text(0.05, 0.985, f"{player_name} - {club} - {role} Template", size=14,
                ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.05, 0.963, f"Percentile Rank vs Top-Five League {pos_group} | Season 2024-25", size=10,
                ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.08, 0.925, "Attacking          Possession        Defending", size=12,
                fontproperties=self.font_bold.prop, color="#000000")

        # Role legend patches
        if "CB" in role:
            att_color, pos_color, def_color = "#DC143C", "#B22222", "#8B0000"
        elif "CM" in role or "DM" in role:
            att_color, pos_color, def_color = "#088F8F", "#AFE1AF", "#097969"
        elif "AM" in role or "Winger" in role:
            att_color, pos_color, def_color = "#87CEFA", "#4169E1", "#00008B"
        elif "F9" in role or "Forward" in role:
            att_color, pos_color, def_color = "#696969", "#A9A9A9", "#D3D3D3"
        else:
            att_color, pos_color, def_color = "#1A78CF", "#FF9300", "#D70232"

        fig.patches.extend([
            plt.Rectangle((0.05, 0.9225), 0.025, 0.021, fill=True, color=att_color, transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.2, 0.9225), 0.025, 0.021, fill=True, color=pos_color, transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.351, 0.9225), 0.025, 0.021, fill=True, color=def_color, transform=fig.transFigure, figure=fig),
        ])

        CREDIT_1 = "@cannoniq.bsky.com\ndata via FBREF / Opta"
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

        output_dir = output_dir or f"Player_profiles/{player_name}"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{player_name}_percentile_pizza.png")

        plt.savefig(save_path, dpi=500, facecolor="#EBEBE9", bbox_inches="tight", edgecolor="none", transparent=False)
        plt.show()

        return save_path

    def generate_full_player_profile_with_pizza(self, player_name, role, df=None):
        # df = df or self.db
        self.plot_role_based_comparison(player_name, role, df)
        self.plot_role_based_kde(player_name, role, df)
        self.create_percentile_pizza(player_name, role, df)
        return self.combine_player_profile_plots(player_name, df, include_logos=True)


    def scrape_scouting_reports(self, scout_links, delay=10):
        """
        Scrape advanced per-90 and percentile stats from FBref scouting reports.
        
        Args:
            scout_links (list): List of scouting report URLs.
            delay (int): Seconds to wait between requests (default: 10).
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (per90_df, percentile_df)
        """
        appended_data_per90 = []
        appended_data_percent = []

        for url in scout_links:
            try:
                warnings.filterwarnings("ignore")

                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'html.parser')
                name_elements = soup.find_all("span")
                name = name_elements[7].text if len(name_elements) > 7 else "Unknown"

                html_content = page.text
                tables = pd.read_html(html_content)
                scout_table = tables[-1]
                scout_table.columns = scout_table.columns.droplevel(0)  # drop top header row

                advanced_stats = scout_table.loc[
                    (scout_table['Statistic'] != "Statistic") & 
                    (scout_table['Statistic'].notna())
                ]

                advanced_stats = advanced_stats.dropna(subset=["Statistic", "Per 90", "Percentile"])

                per90 = advanced_stats[["Statistic", "Per 90"]].set_index("Statistic").T
                per90["Name"] = name

                percent = advanced_stats[["Statistic", "Percentile"]].set_index("Statistic").T
                percent["Name"] = name

                appended_data_per90.append(per90)
                appended_data_percent.append(percent)

                print(f"✔ Scraped {name}")
                time.sleep(delay)

            except Exception as e:
                print(f"✘ Failed to scrape {url}: {e}")
                continue

        # Combine per90
        per90_df = pd.concat(appended_data_per90, ignore_index=True)
        per90_df = per90_df[['Name'] + [col for col in per90_df.columns if col != 'Name']]
        per90_df = per90_df.loc[:, ~per90_df.columns.duplicated()]

        # Combine percentiles
        percentile_df = pd.concat(appended_data_percent, ignore_index=True)
        percentile_df = percentile_df[['Name'] + [col for col in percentile_df.columns if col != 'Name']]
        percentile_df = percentile_df.loc[:, ~percentile_df.columns.duplicated()]

        return per90_df, percentile_df


    def create_scout_percentile_pizza(self, df_in, player_name, output_subdir="ML_DOF_DCAM"):
        params = [
            "Non-Penalty Goals", "npxG + xAG", "Assists",
            "Shot-Creating Actions", "Carries into Penalty Area",
            "Touches", "Progressive Passes", "Progressive Carries",
            "Passes into Penalty Area", "Crosses",
            "Interceptions", "Tackles Won",
            "Passes Blocked", "Ball Recoveries", "Aerials Won"
        ]

        subset_of_data = df_in.query('Player == @player_name')
        if subset_of_data.empty:
            raise ValueError(f"{player_name} not found in the provided dataframe.")

        scout_links = list(subset_of_data.scouting_url.unique())
        if not scout_links:
            raise ValueError(f"No scouting URL available for {player_name}.")

        # Scrape and get percentile data
        _, appended_data_percentile = self.scrape_scouting_reports(scout_links, delay=10)
        appended_data_percentile = appended_data_percentile[params]
        appended_data_percentile = appended_data_percentile.apply(pd.to_numeric)
        values = appended_data_percentile.iloc[0].values.tolist()

        team = subset_of_data['Squad'].unique()[0]
        pos_group = subset_of_data['position_group'].unique()[0]

        style.use('fivethirtyeight')

        # Colors
        slice_colors = ["#1A78CF"] * 5 + ["#FF9300"] * 5 + ["#D70232"] * 5
        text_colors = ["#000000"] * 10 + ["#F2F2F2"] * 5

        baker = PyPizza(
            params=params,
            background_color="#EBEBE9",
            straight_line_color="#EBEBE9",
            straight_line_lw=1,
            last_circle_lw=0,
            other_circle_lw=0,
            inner_circle_size=20
        )

        fig, ax = baker.make_pizza(
            values,
            figsize=(8, 8.5),
            color_blank_space="same",
            slice_colors=slice_colors,
            value_colors=text_colors,
            value_bck_colors=slice_colors,
            blank_alpha=0.4,
            kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
            kwargs_params=dict(
                color="#000000", fontsize=9,
                fontproperties=self.font_normal.prop, va="center"
            ),
            kwargs_values=dict(
                color="#000000", fontsize=11,
                fontproperties=self.font_normal.prop, zorder=3,
                bbox=dict(edgecolor="#000000", facecolor="cornflowerblue",
                        boxstyle="round,pad=0.2", lw=1)
            )
        )

        # Title, subtitle, legend
        fig.text(0.05, 0.985, f"{player_name} - {team}", size=14,
                ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.05, 0.963,
                f"Percentile Rank vs Top-Five League {pos_group} | Season 2024–25",
                size=10, ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.08, 0.925, "Attacking          Possession        Defending",
                size=12, fontproperties=self.font_bold.prop, color="#000000")

        # Legend blocks
        fig.patches.extend([
            plt.Rectangle((0.05, 0.9225), 0.025, 0.021, fill=True, color="#1a78cf", transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.2, 0.9225), 0.025, 0.021, fill=True, color="#ff9300", transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.351, 0.9225), 0.025, 0.021, fill=True, color="#d70232", transform=fig.transFigure, figure=fig),
        ])

        # Credits
        CREDIT_1 = "@stephenaq7\ndata via FBREF / Opta"
        CREDIT_2 = "inspired by: @Worville, @FootballSlices"
        fig.text(0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
                fontproperties=self.font_italic.prop, color="#000000", ha="right")

        # Logo
        try:
            ax3 = fig.add_axes([0.80, 0.075, 0.15, 1.75])
            ax3.axis('off')
            img = image.imread(os.path.join(self.image_dir, "piqmain.png"))
            ax3.imshow(img)
        except FileNotFoundError:
            print("Logo not found.")

        # Save
        save_dir = os.path.join(self.data_dir, "Substack_Images", output_subdir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{player_name} - plot.png")

        plt.savefig(save_path, dpi=500, facecolor="#EFE9E6", bbox_inches="tight", edgecolor="none", transparent=False)
        plt.show()

        return save_path


    def plot_kmeans_clusters_for_player(self, df, player_name):
        fb_ref_db = fbref.CreateFBRefDatabase()

        player_row = df[df['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        position_group = player_row['position_group'].values[0]
        df = df[df['position_group'] == position_group]

        filtered_kmeans = fb_ref_db.create_kmeans_df(df)
                

        # Create the scatter plot using lmplot
        ax = sns.lmplot(x="x", y="y", hue='cluster', data=filtered_kmeans, legend=True, legend_out=True,
                        fit_reg=False, height=20, scatter_kws={"s": 250})

        texts = []
        for x, y, s in zip(filtered_kmeans.x, filtered_kmeans.y, filtered_kmeans.name):
            texts.append(plt.text(x, y, s, fontweight='light'))

        # Additional axes for logos and titles
        fig = plt.gcf()
        ax1 = plt.gca()

        # Add title and logos to the current figure
        fig.text(.1, 1.08, f'KMeans clustering - {position_group}', size=20)
        fig.text(.1, 1.03, '24/25 Season | Viz by @stephenaq7 | Data via FBREF', size=14)

        try:
            ax2 = fig.add_axes([0.01, 0.175, 0.07, 1.75])
            ax2.axis('off')
            img1 = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/premier-league-2-logo.png')
            ax2.imshow(img1)

            ax3 = fig.add_axes([0.85, 0.175, 0.1, 1.75])
            ax3.axis('off')
            img2 = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
            ax3.imshow(img2)
        except FileNotFoundError:
            print("One or more logo images not found.")

        # Set axis limits and labels for the lmplot
        ax1.set(ylim=(-2, 2))
        plt.tick_params(labelsize=15)
        plt.xlabel("PC 1", fontsize=20)
        plt.ylabel("PC 2", fontsize=20)

        plt.tight_layout()
        plt.show()

        return filtered_kmeans

    def find_similar_players_and_scores(self, player_name, kmeans_df, pitch_iq_scores, max_age=32, top_n=35):
        player = kmeans_df[kmeans_df['name'] == player_name].iloc[0]
        kmeans_df['distance'] = np.sqrt((kmeans_df['x'] - player['x'])**2 + (kmeans_df['y'] - player['y'])**2)
        max_distance = kmeans_df['distance'].max()
        kmeans_df['perc_similarity'] = (((max_distance - kmeans_df['distance']) / max_distance) * 100) * 0.90
        similar_players = kmeans_df.sort_values('distance').head(top_n + 1)[1:]
        similarity_table = similar_players[['name', 'perc_similarity']].rename(columns={'name': 'Player'})
        metrics_similarity = pd.merge(similarity_table, pitch_iq_scores, on='Player', how='left')
        metrics_similarity = metrics_similarity.drop_duplicates(subset=['Player'])
        metrics_similarity = metrics_similarity[metrics_similarity['Age'] < max_age]
        comparative_list = list(metrics_similarity.Player.unique())
        sim_index = [round(item, 2) for item in metrics_similarity.perc_similarity.unique()]
        return metrics_similarity, comparative_list, sim_index


    def plot_comparison_radars(self, player_name, role, comparative_list, Cannoniq_DB, metrics_similarity, sim_index):
        params = self.player_role_templates[role]

        def get_player_data(df, player_name, params):
            player_data = df[df['Player'] == player_name][params].values.tolist()
            return [val for sublist in player_data for val in sublist]

        main_player = get_player_data(Cannoniq_DB, player_name, params)
        comp_players = [get_player_data(metrics_similarity, comp, params) for comp in comparative_list]

        def convert_to_numeric(input_list):
            return [float(x) for x in input_list]

        numeric_main_player = convert_to_numeric(main_player)

        for idx, comp_player in enumerate(comp_players):
            numeric_comp_player = convert_to_numeric(comp_player)

            low = [min(value, value_2) * 0.5 for value, value_2 in zip(numeric_main_player, numeric_comp_player)]
            high = [max(value, value_2) * 1.05 for value, value_2 in zip(numeric_main_player, numeric_comp_player)]

            radar = Radar(params, low, high,
                            round_int=[False]*len(params),
                            num_rings=5,
                            ring_width=1, center_circle_radius=1)

            fig, ax = radar.setup_axis()
            fig.patch.set_facecolor('#f0f0f0')
            ax.set_facecolor('#f0f0f0')

            radar.draw_circles(ax=ax, facecolor='#ffb2b2')
            radar.draw_radar_compare(numeric_main_player, numeric_comp_player, ax=ax,
                                        kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
                                        kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})

            radar.draw_range_labels(ax=ax, fontsize=15)
            radar.draw_param_labels(ax=ax, fontsize=15)

            ax.legend([player_name, comparative_list[idx]], loc='upper right', fontsize=12)

            league_icon = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images/premier-league-2-logo.png")
            league_ax = fig.add_axes([0.002, 0.89, 0.20, 0.15], zorder=1)
            league_ax.imshow(league_icon)
            league_ax.axis("off")

            fig_text(
                x=0.57, y=0.90,
                s=f"{player_name} vs {comparative_list[idx]}\nSeason 2023/2024\nPitch IQ Similarity Score: {sim_index[idx]}%\nViz by @stephenaq7.",
                va="bottom", ha="right",
                fontsize=17, color="black", weight="book"
            )

            ax3 = fig.add_axes([0.80, 0.09, 0.13, 1.75])
            ax3.axis('off')
            img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
            ax3.imshow(img)

            plt.show()


    def generate_similarity_score_card(self, player_name, mertrics_similarity, Cannoniq_DB):
        mertrics_similarity = mertrics_similarity.rename(columns={'Squad': 'team'})
        fm_ids = pd.read_csv("/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/Top6_leagues_fotmob_ids.csv")
        fm_ids = fm_ids[["team", "team_id"]]

        mertrics_similarity = mertrics_similarity.merge(fm_ids, on='team', how='left')
        mertrics_similarity = mertrics_similarity.dropna(subset=['team_id'])
        mertrics_similarity['team_id'] = mertrics_similarity['team_id'].astype(float).astype(int)

        mertrics_similarity[['perc_similarity', 'Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']] = \
            mertrics_similarity[['perc_similarity', 'Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']].round(2)

        df_final = mertrics_similarity[['Player', 'Pos','team_id','perc_similarity', 'Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']]
        metric_scores =['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']

        sim_player_vals = Cannoniq_DB[Cannoniq_DB['Player'] == player_name][metric_scores].values.tolist()
        sim_player_vals = [val for sublist in sim_player_vals for val in sublist]

        df_final['Δ% Passing'] = ((df_final['Passing_Score'] - sim_player_vals[0]) / sim_player_vals[0]).round(1) * 100
        df_final['Δ% Defending'] = ((df_final['Defending_Score'] - sim_player_vals[1]) / sim_player_vals[1]).round(1) * 100
        df_final['Δ% Creation'] = ((df_final['Creation_Score'] - sim_player_vals[2]) / sim_player_vals[2]).round(1) * 100
        df_final['Δ% Shooting'] = ((df_final['Shooting_Score'] - sim_player_vals[3]) / sim_player_vals[3]).round(1) * 100
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
        columns = ['Player', 'Pos', 'perc_similarity', 'Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score', 'Δ% Passing', 'Δ% Defending', 'Δ% Creation', 'Δ% Shooting']

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
        column_names = ['Player', 'Position', 'Percent\nSimilarity','Passing', 'Defending', 'Creation', 'Shooting','Δ%\nPassing','Δ%\nDefending','Δ%\nCreation','Δ%\nShooting']
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
        delta_columns = ['Δ% Passing', 'Δ% Defending', 'Δ% Creation', 'Δ% Shooting']

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
            x = 0.6, y = 0.95, 
            s = f'{player_name} - PIQ Similarity Model Results 24/25',
            va = "bottom", ha = "right",
            fontsize = 17, color = "black", weight = "bold"
        )

        fig_text(
            x = 0.74, y = 0.92, 
            s = f'Passing, Defending, Creation & Shooting scores generated via weighted aggregated metrics from FBREF (Out of 10)',
            va = "bottom", ha = "right",
            fontsize = 12, color = "black", weight = "book"
        )
        fig_text(
            x = 0.72, y = 0.90, 
            s = f'Δ% columns are PIQ score percentage change, -ve% means {player_name} has a better score and vice versa',
            va = "bottom", ha = "right",
            fontsize = 12, color = "black", weight = "book"
        )

        ### Add Stats by Steve logo
        ax3 = fig.add_axes([0.83, 0.13, 0.09, 1.60])
        ax3.axis('off')
        img = plt.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
        ax3.imshow(img)
        plt.show()
        output_path = os.path.join(self.data_dir, "Player_profiles", player_name, f"{player_name}_similarity_score_card.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())   

        return 
    def generate_full_similarity_analysis(self, player_name, role, Cannoniq_DB, max_age=32, top_n=35):
        # Step 1: Get KMeans cluster plot and reduced dataset
        kmeans_df = self.plot_kmeans_clusters_for_player(Cannoniq_DB, player_name=player_name)

        # Step 2: Find similar players and similarity scores
        metrics_similarity, comparative_list, sim_index = self.find_similar_players_and_scores(
            player_name, kmeans_df, Cannoniq_DB, max_age=max_age, top_n=top_n
        )

        # Step 3: Plot role-based radar comparisons
        self.plot_comparison_radars(
            player_name, role, comparative_list, Cannoniq_DB, metrics_similarity, sim_index
        )

        # Step 4: Generate and return similarity score table
        df_final = self.generate_similarity_score_card(
            player_name, metrics_similarity, Cannoniq_DB
        )
        
        return df_final