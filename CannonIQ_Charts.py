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
        scaled_df = df[(df['position_group'] == pos_group) & (df['90s'] > 15)]

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

        # plt.show()

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
        scaled_df = df[(df['position_group'] == pos_group) & (df['90s'] > 15)]

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

            fig.savefig(f"{player_dir}/{player_name}_{role}_radar.png", dpi=300, bbox_inches="tight")

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

        # plt.show()

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
        filtered_df = df[(df['position_group'] == pos_group) & (df['90s'] > 15)]

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
        fig.savefig(f"{player_dir}/{player_name}_{role}_kde.png", dpi=300, bbox_inches="tight")

        # plt.show()



    def combine_player_profile_plots(self, player_name, role ,df=None, include_logos=True):
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
        radar_path = f"Player_profiles/{player_name}/{player_name}_{role}_radar.png"
        kde_path = f"Player_profiles/{player_name}/{player_name}_{role}_kde.png"
        output_path = f"Player_profiles/{player_name}/{player_name}_{role}_combined_profile.png"

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
        return self.combine_player_profile_plots(player_name, role, df, include_logos)

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
        filtered_df = df[(df['position_group'] == pos_group) & (df['90s'] > 15)]

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
        save_path = os.path.join(output_dir, f"{player_name}_{role}_percentile_pizza.png")

        plt.savefig(save_path, dpi=500, facecolor="#EBEBE9", bbox_inches="tight", edgecolor="none", transparent=False)
        # plt.show()

        return save_path

    def generate_full_player_profile_with_pizza(self, player_name, role, df=None):
        # df = df or self.db
        self.plot_role_based_comparison(player_name, role, df)
        self.plot_role_based_kde(player_name, role, df)
        self.create_percentile_pizza(player_name, role, df)
        self.create_scout_percentile_pizza(df, player_name)
        return self.combine_player_profile_plots(player_name,  role, df, include_logos=True)


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


    def create_scout_percentile_pizza(self, df_in, player_name):
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
        output_dir = f"Player_profiles/{player_name}"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{player_name}_basic_percentile_pizza.png")

        plt.savefig(save_path, dpi=500, facecolor="#EBEBE9", bbox_inches="tight", edgecolor="none", transparent=False)

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
        # plt.show()

        return filtered_kmeans

    def find_similar_players_and_scores(self, player_name, kmeans_df, pitch_iq_scores, max_age=30, top_n=20):
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
            output_dir = f"Player_profiles/{player_name}/compare_radars/"
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}_{player_name} vs {comparative_list[idx]}.png', bbox_inches='tight', facecolor=fig.get_facecolor())


            # plt.show()


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
            x = 0.5, y = 0.95, 
            s = f'{player_name} - PIQ Similarity Model Results 24/25',
            va = "bottom", ha = "center",
            fontsize = 17, color = "black", weight = "bold"
        )

        fig_text(
            x = 0.5, y = 0.92, 
            s = f'Passing, Defending, Creation & Shooting scores generated via weighted aggregated metrics from FBREF (Out of 10)',
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
  
        output_path = f"Player_profiles/{player_name}/scorecards/"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_path = os.path.join(output_path, f"{player_name}_similarity_score_card.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
        
        plt.show()

        # ---------------------- 
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
    
    def create_metrics_scores(self, key_stats_df):
        # Define the key_stats grouped by the metrics
        core_stats = ['90s','Total - Cmp%','KP', 'TB','Sw','PPA', 'PrgP','Tkl%','Blocks', 'Tkl+Int','Clr', 'Carries - PrgDist','SCA90','GCA90','CrsPA','xA', 'Rec','PrgR','xG', 'Sh','SoT']
        passing_metrics = ['Total - Cmp%', 'KP', 'TB', 'Sw', 'PPA', 'PrgP']
        defending_metrics = ['Tkl%', 'Blocks', 'Tkl+Int', 'Clr']
        creation_metrics = ['Carries - PrgDist', 'SCA90', 'GCA90', 'CrsPA', 'xA', 'Rec', 'PrgR']
        shooting_metrics = ['xG', 'Sh', 'SoT']

        # Create a MinMaxScaler instance
        scaler = MinMaxScaler()

        # Normalize the metrics
        stats_normalized = key_stats_df.copy()  # Create a copy of the DataFrame
        stats_normalized[core_stats] = scaler.fit_transform(stats_normalized[core_stats])

        # Calculate scores for each metric grouping and scale to 0-10
        stats_normalized['Passing_Score'] = stats_normalized[passing_metrics].mean(axis=1) * 10
        stats_normalized['Defending_Score'] = stats_normalized[defending_metrics].mean(axis=1) * 10
        stats_normalized['Creation_Score'] = stats_normalized[creation_metrics].mean(axis=1) * 10
        stats_normalized['Shooting_Score'] = stats_normalized[shooting_metrics].mean(axis=1) * 10

        # Add a small offset to ensure unique scores
        stats_normalized['Passing_Score'] += stats_normalized.index * 0.001
        stats_normalized['Defending_Score'] += stats_normalized.index * 0.001
        stats_normalized['Creation_Score'] += stats_normalized.index * 0.001
        stats_normalized['Shooting_Score'] += stats_normalized.index * 0.001

        # Clip scores to ensure they are within the 0-10 range
        stats_normalized[['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']] = stats_normalized[['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']].clip(lower=0, upper=10)
        return stats_normalized

    def adjust_player_rating_range(self, dataframe):
        # Get the 'total player rating' column
        player_ratings = dataframe[['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']]
        
        # Define the desired range for the ratings
        min_rating = 4.5
        max_rating = 9.5
        
        # Normalize the ratings to be within the desired range (5 to 9.5) for each column
        for col in player_ratings.columns:
            normalized_ratings = min_rating + (max_rating - min_rating) * ((player_ratings[col] - player_ratings[col].min()) / (player_ratings[col].max() - player_ratings[col].min()))
            dataframe[col] = normalized_ratings
        
        return dataframe
    
    def update_pitch_iq_scores(self, player_name, Cannoniq_DB):

        # Step 1: Find the player's row and position group
        player_row = Cannoniq_DB[Cannoniq_DB['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in Cannoniq_DB.")

        position_group = player_row['position_group'].values[0]

        # Step 2: Filter to players in the same position group with >10 90s
        filtered_df = Cannoniq_DB[
            (Cannoniq_DB['position_group'] == position_group) & 
            (Cannoniq_DB['90s'] > 15)
        ].copy()

        # Step 3: Score and adjust
        pitch_iq_scoring = self.create_metrics_scores(filtered_df)
        pitch_iq_scoring = self.adjust_player_rating_range(pitch_iq_scoring)

        # Step 4: Update existing columns in filtered_df
        for col in ['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']:
            filtered_df[col] = filtered_df['Player'].map(
                pitch_iq_scoring.set_index('Player')[col]
            )

        return filtered_df
    
    def create_player_scorecard_plot(self, player_name, role, Cannoniq_DB):
        from matplotlib import style
        from mplsoccer import PyPizza
        from PIL import Image
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import pandas as pd
        from scipy.stats import percentileofscore

        Cannoniq_DB = self.update_pitch_iq_scores(player_name, Cannoniq_DB)

        player_row = Cannoniq_DB[Cannoniq_DB['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        position_group = player_row['position_group'].values[0]
        Cannoniq_DB = Cannoniq_DB[Cannoniq_DB['position_group'] == position_group]

        style.use('fivethirtyeight')
        background = '#F0F0F0'  # FiveThirtyEight background
        text_color = 'black'
        grey_text = '#4E616C'

        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['xtick.color'] = grey_text
        mpl.rcParams['ytick.color'] = grey_text
        mpl.rcParams.update({'font.size': 14})

        lst_cols_pr = ['Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']
        sim_player = Cannoniq_DB[Cannoniq_DB['Player'] == player_name][lst_cols_pr]

        percentiles = {
            col: percentileofscore(Cannoniq_DB[col], sim_player[col].values[0], kind='rank')
            for col in lst_cols_pr
        }
        sim_player_perc = pd.DataFrame([percentiles])

        df_player_pr_t = sim_player_perc[lst_cols_pr].T.reset_index(drop=False)
        df_player_pr_t.columns = ['Metric', 'PR']
        dict_metrics = {
            'Passing_Score': 'Passing',
            'Defending_Score': 'Defending',
            'Creation_Score': 'Creating',
            'Shooting_Score': 'Shooting'
        }
        df_player_pr_t['Metric'] = df_player_pr_t['Metric'].map(dict_metrics)

        metric = df_player_pr_t['Metric']
        pr = df_player_pr_t['PR']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig = plt.figure(figsize=(18, 18))
        gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1.5], width_ratios=[1, 0.8])
        fig.set_facecolor(background)

        # Bar chart
        ax1 = fig.add_subplot(gs[0:2, 0])
        ax1.set_facecolor(background)
        ax1.barh(metric, pr / 100, color=colors, alpha=0.75)
        for spine in ['top', 'bottom', 'left', 'right']:
            ax1.spines[spine].set_visible(False)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.tick_params(axis='y', labelsize=10)
        ax1.xaxis.set_tick_params(pad=2)
        ax1.yaxis.set_tick_params(pad=20)
        ax1.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax1.invert_yaxis()
        for i in ax1.patches:
            ax1.text(i.get_width() + 0.015, i.get_y() + 0.4, f"{round((100 * i.get_width()), 1)}%", fontsize=14,
                     fontweight='regular', color='black')
        vals = ax1.get_xticks()
        ax1.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
        ax1.set_yticklabels(df_player_pr_t['Metric'], size=9, rotation=90, color=grey_text)
        ax1.set_title('PitchIQ Score Percentile', size=18, color=grey_text)

        ax1.axvline(0.5, 0, 0.952, color='#FFD200', linestyle='--', linewidth=3)

        # Pizza chart
        params = list(dict_metrics.values())
        values = [round(val, 1) for val in sim_player[lst_cols_pr].values.flatten().tolist()]
        baker = PyPizza(params=params, straight_line_color="#F2F2F2", straight_line_lw=1, straight_line_limit=11.0,
                        last_circle_lw=0, other_circle_lw=0, inner_circle_size=0.4)
        fig_pizza, ax_pizza = baker.make_pizza(values, figsize=(6, 6), color_blank_space="same", blank_alpha=0.4,
                                               param_location=5.5, slice_colors=colors, kwargs_slices=dict(facecolor="cornflowerblue",
                                                                                                 edgecolor="#F2F2F2", zorder=2,
                                                                                                 linewidth=1),
                                               kwargs_params=dict(color="#000000", fontsize=12, va="center"),
                                               kwargs_values=dict(color="#000000", fontsize=12, zorder=3,
                                                                  bbox=dict(edgecolor="#000000",
                                                                            facecolor="cornflowerblue",
                                                                            boxstyle="round,pad=0.2", lw=1)))
        fig_pizza.savefig("temp_img/pizza_plot.png", bbox_inches="tight", dpi=300)
        pizza_img = Image.open("temp_img/pizza_plot.png")
        ax2 = fig.add_subplot(gs[:2, 1])
        ax2.imshow(pizza_img)
        ax2.axis('off')
        ax2.set_title('PitchIQ Score Radar', size=18, color=grey_text)

        # Swarm plots
        s_params = ['KP', 'PPA', 'PrgP', 'Tkl+Int', 'Carries - PrgDist', 'SCA90', 'xA', 'xG']
        s_params_dict = {
            'KP': 'Key Passes',
            'PPA': 'Passes into Penalty Area',
            'PrgP': 'Progressive Passes',
            'Tkl+Int': 'Tackles + Interceptions',
            'Carries - PrgDist': 'Progressive Carry Distance',
            'SCA90': 'Shot-Creating Actions per 90',
            'xA': 'Expected Assists',
            'xG': 'Expected Goals'
        }

        for idx, param in enumerate(s_params):
            ax = fig.add_subplot(gs[2 + idx // 2, idx % 2])
            ax.set_facecolor(background)
            ax.grid(ls='dotted', lw=.5, color='lightgrey', axis='y', zorder=1)

            sns.swarmplot(x=param, data=Cannoniq_DB, ax=ax, zorder=1, color='#64645e')

            ax.set_xlabel(s_params_dict[param], size=9, color=grey_text)
            ax.tick_params(axis='x', labelsize=9, colors=grey_text)
            ax.tick_params(axis='y', length=0, labelleft=False)

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - 0.1, ymax + 0.1)

            for spine in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine].set_visible(False)

            if player_name in Cannoniq_DB['Player'].values:
                val = Cannoniq_DB.loc[Cannoniq_DB['Player'] == player_name, param].values[0]
                ax.scatter(x=val, y=0, s=200, c='#6CABDD', zorder=2)

        fig.suptitle(f'How Does {player_name} Compare\nAgainst EU Top 5 League {role}s?',
                    fontsize=22, fontweight='bold', color=text_color, x=0.55, y=1)

        fig.text(0.2, 0.935,
                f'A comparison of the {role}s Passing, Defending, Creation & Shooting scores\nScores generated via weighted aggregated metrics from FBREF (Out of 10) in the \'Big 5\' European Leagues.',
                fontsize=14, fontweight='regular', color=text_color)

        league_icon = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images/premier-league-2-logo.png")
        league_ax = fig.add_axes([0.1, 0.87, 0.08, 0.14], zorder=1)
        league_ax.imshow(league_icon)
        league_ax.axis("off")

        ax3 = fig.add_axes([0.80, 0.87, 0.11, 0.14])
        ax3.axis('off')
        img = plt.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
        ax3.imshow(img)

        plt.subplots_adjust(hspace=0.9, bottom=0.08)

        output_path = f"Player_profiles/{player_name}/scorecards/"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_path = os.path.join(output_path, f"{player_name}_general_score_card.png")

        # Save BEFORE plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()

    def generate_full_analysis_suite(self, player_name, role, Cannoniq_DB, comparative_list=None):
        # Check if player has at least 15 90s
        player_row = Cannoniq_DB[Cannoniq_DB['Player'] == player_name]
        if player_row.empty or player_row['90s'].values[0] < 15:
            print(f"Skipping {player_name} – insufficient minutes played.")
            return

        self.generate_full_player_profile_with_pizza(player_name, role, Cannoniq_DB)
        self.plot_role_based_comparison(player_name, role, Cannoniq_DB, comparative_list)
        _ = self.generate_full_similarity_analysis(player_name, role, Cannoniq_DB)
        self.create_player_scorecard_plot(player_name, role, Cannoniq_DB)


    def get_player_league_timeseries(self, Cannoniq_DB, player_name):
        if 'match_logs' not in Cannoniq_DB.columns:
            raise ValueError("'match_logs' column not found in Cannoniq_DB.")

        player_row = Cannoniq_DB[Cannoniq_DB['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"Player '{player_name}' not found in Cannoniq_DB.")

        raw_links = player_row['match_logs'].values[0]

        if isinstance(raw_links, str):
            match_data_urls = [link.strip() for link in raw_links.split(',') if link.strip()]
        elif isinstance(raw_links, list):
            match_data_urls = raw_links
        else:
            raise ValueError("Unsupported format in 'match_logs'. Must be string or list of URLs.")

        return self.league_performance_df(match_data_urls, player_name)



    def league_performance_df(self, match_links, player_name):

        data_append = []

        for x in match_links:
            print(f"Fetching: {x}")
            warnings.filterwarnings("ignore")

            page = requests.get(x)
            soup = BeautifulSoup(page.content, 'html.parser')

            html_content = page.text.replace('<!--', '').replace('-->', '')
            df = pd.read_html(html_content)

            try:
                df[0].columns = df[0].columns.droplevel(0)
            except ValueError:
                pass

            stats = df[0]
            stats = stats[
                (stats['Comp'].isin(['La Liga', 'Premier League', 'Bundesliga', 'Serie A', 'Ligue 1','Primeira Liga'])) &
                (stats['Pos'] != "On matchday squad, but did not play")
            ]

            keep_cols = [
                'Opponent', 'Start', 'Pos', 'Date', 'Gls', 'Ast', 'xG', 'npxG', 'xAG', 'SoT',
                'Squad', 'CrdY', 'CrdR', 'Result', 'SCA', 'GCA', 'Cmp', 'Cmp%', 'Min',
                'PrgP', 'Carries', 'PrgC', 'Succ', 'Tkl', 'Int', 'Blocks', 'Touches'
            ]
            season = stats[keep_cols].copy()

            if season['Cmp%'].dtype == 'object':
                season['Cmp%'] = season['Cmp%'].str.replace('%', '', regex=False).astype(float)

            columns_to_convert = [
                'Gls', 'Ast', 'xG', 'npxG', 'xAG', 'SoT', 'SCA', 'GCA', 'Cmp', 'Min',
                'Cmp%', 'PrgP', 'Carries', 'PrgC', 'Succ', 'Tkl', 'Int', 'Blocks', 'Touches'
            ]
            for col in columns_to_convert:
                if col in season.columns and isinstance(season[col], pd.Series):
                    season[col] = pd.to_numeric(season[col], errors='coerce').fillna(0).astype(float)
                else:
                    season[col] = 0.0

            season = season.rename({'Squad': 'team'}, axis=1)
            season['Player'] = player_name

            data_append.append(season)
            time.sleep(1)

        df_total = pd.concat(data_append, ignore_index=True)

        df_total['CrdY'] = pd.to_numeric(df_total['CrdY'], errors='coerce').fillna(0).astype(int)
        df_total['CrdR'] = pd.to_numeric(df_total['CrdR'], errors='coerce').fillna(0).astype(int)
        df_total['Min'] = pd.to_numeric(df_total['Min'], errors='coerce').fillna(0).astype(int)

        return df_total

    def plot_per90_summary(self, per90_df, player_name):

        style.use('fivethirtyeight')
        # -- Derived columns
        per90_df["npxG + xAG"] = per90_df["npxG"] + per90_df["xAG"]
        per90_df["G+A"] = per90_df["Gls"] + per90_df["Ast"]
        per90_df["Tkl + Int"] = per90_df["Tkl"] + per90_df["Int"]
        per90_df["PrgP + PrgC"] = per90_df["PrgP"] + per90_df["PrgC"]

        # -- Metrics to plot with readable labels
        plot_data = {
            "Non-Penalty xG + xAG": per90_df["npxG + xAG"],
            "Goals + Assists": per90_df["G+A"],
            "Shot-Creating Actions": per90_df["SCA"],
            "Tackles + Interceptions": per90_df["Tkl + Int"],
            "Pass Completion %": per90_df["Cmp%"],
            "Progressive Pass + Carry": per90_df["PrgP + PrgC"]
        }

        # -- Define opponent type colors
        opponent_colors = {
            'big_teams': '#003f5c',
            'european_2nd_tier': '#58508d',
            'mid_table': '#bc5090',
            'relegation_teams': '#ffa600'
        }

        fig, axs = plt.subplots(3, 2, figsize=(12, 12), dpi=300)
        plt.rcParams['hatch.linewidth'] = 0.2
        axs = axs.flatten()

        for idx, (title, values) in enumerate(plot_data.items()):
            ax = axs[idx]
            ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

            bars_ = ax.bar(
                per90_df["opponent_type"].str.replace('_', ' ').str.title(),
                values,
                hatch='//////',
                ec=ax.get_facecolor(),
                width=0.6
            )

            for i, b in enumerate(bars_):
                opponent = per90_df["opponent_type"].iloc[i]
                color = opponent_colors.get(opponent, '#333333')
                b.set_facecolor(color)
                ax.annotate(f'{b.get_height():.2f}', xy=(i, b.get_height()),
                            xytext=(0, 6), textcoords='offset points',
                            ha='center', va='center', color=color, fontsize=9)

            ax.set_title(title, fontsize=10)
            ax.set_xticks(range(len(per90_df["opponent_type"])))
            ax.set_xticklabels(per90_df["opponent_type"].str.replace('_', ' ').str.title(), rotation=30)
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8)

        # -- Final layout
        fig.tight_layout()
        
        output_path = f"Player_profiles/{player_name}/attacking_performance/"

        save_path = os.path.join(output_path, f"{player_name}_performance_types.png")
        # Save BEFORE plt.show()
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

        plt.show()

    def plot_cumulative_chart(self, df_player_series, df_timeseries, player_name, plot_type='goals'):

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.patheffects as path_effects
        from highlight_text import fig_text, ax_text
        from matplotlib import style
        import numpy as np

        style.use('fivethirtyeight')

        def get_cumulative(player_name, data):
            df = data.copy()
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df = df[(df['Player'] == player_name)][['Player', 'Gls', 'xG', 'Ast', 'xAG', 'Date']].reset_index(drop=True)
            df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
            df['cum_goals'] = df['Gls'].cumsum()
            df['cum_xgoals'] = df['xG'].cumsum()
            df['cum_assists'] = df['Ast'].cumsum()
            df['cum_xassists'] = df['xAG'].cumsum()
            return df

        player = df_player_series['Player'].iloc[0]
        x_val = df_player_series['xG'].iloc[0] if plot_type == 'goals' else df_player_series['xAG'].iloc[0]
        y_val = df_player_series['Gls'].iloc[0] if plot_type == 'goals' else df_player_series['Ast'].iloc[0]
        test_df = get_cumulative(player, df_timeseries)

        n_matches = test_df.shape[0]
        final_match_idx = n_matches - 1

        fig = plt.figure(figsize=(10, 6), dpi=200)
        ax_main = plt.subplot2grid((3, 4), (1, 0), colspan=4, rowspan=2)

        ax_main.grid(ls='--', color='lightgrey')
        for spine in ax_main.spines.values():
            spine.set_visible(False)
        ax_main.tick_params(color='lightgrey', labelsize=6, labelcolor='grey')

        if plot_type == 'goals':
            ax_main.plot(test_df.index + 1, test_df['cum_goals'], marker='o', mfc='white', ms=1, linewidth=1, color='#287271')
            ax_main.plot(test_df.index + 1, test_df['cum_xgoals'], marker='o', mfc='white', ms=1, linewidth=1, color='#D81159')
        else:
            ax_main.plot(test_df.index + 1, test_df['cum_assists'], marker='o', mfc='white', ms=1, linewidth=1, color='#287271')
            ax_main.plot(test_df.index + 1, test_df['cum_xassists'], marker='o', mfc='white', ms=1, linewidth=1, color='#D81159')

        ax_main.set_ylim(-0.5, max(y_val, x_val) + 5)
        ax_main.set_xlim(-1, max(38, n_matches + 1))
        ax_main.set_xlabel("Match", fontsize=8, color='#4E616C')

        xtick_range = np.arange(0, max(38, n_matches + 1), 5)
        ax_main.set_xticks(xtick_range)
        ax_main.set_xticklabels([str(int(x)) for x in xtick_range], fontsize=8, color='#4E616C')

        # Scatter points
        ax_main.scatter(final_match_idx + 1, y_val, color='#287271', zorder=5)
        ax_main.annotate(f'{"Goals" if plot_type == "goals" else "Assists"}: {y_val:.0f}',
                        xy=(final_match_idx + 1, y_val),
                        xytext=(-50, 6), textcoords='offset points',
                        ha='right', va='center',
                        color='#287271', fontsize=9, weight='bold')

        ax_main.scatter(final_match_idx + 1, x_val, color='#D81159', zorder=5)
        ax_main.annotate(f'{"xG" if plot_type == "goals" else "xAG"}: {x_val:.1f}',
                        xy=(final_match_idx + 1, x_val),
                        xytext=(-50, -10), textcoords='offset points',
                        ha='right', va='center',
                        color='#D81159', fontsize=9, weight='bold')

        # Player title and footer
        ax_text(
            x=0, y=1.15,
            s=f'<Exp. Goals: {x_val:.1f}> <|> <Goals: {y_val:.0f}>' if plot_type == 'goals' else f'<Exp. Assists: {x_val:.1f}> <|> <Assists: {y_val:.0f}>',
            ax=ax_main,
            highlight_textprops=[
                # {'weight': 'bold', 'color': 'black'},
                {'size': '7', 'color': '#D81159'},
                {'size': '7', 'color': 'grey'},
                {'size': '7', 'color': '#287271'}
            ],
            ha='left', size=10, annotationbbox_kw={'xycoords': 'axes fraction'}
        )

        fig_text(
            x=0.08, y=0.61,
            s=f'<Expected Goals> (in probabilistic terms) vs. <actual Goals>' if plot_type == 'goals' else f'<Expected Assists> (in probabilistic terms) vs. <actual Assists>',
            highlight_textprops=[
                {'weight': 'bold', 'color': '#D81159'},
                {'weight': 'bold', 'color': '#287271'}
            ],
            va='bottom', ha='left',
            fontsize=8, color='#4E616C', font='DejaVu Sans'
        )
        output_path = f"Player_profiles/{player_name}/attacking_performance/"
        os.makedirs(output_path, exist_ok=True)  # Ensure directory exists

        # Set save path based on plot type
        if plot_type == 'goals':
            save_path = os.path.join(output_path, f"{player_name}_goals_xg.png")
        else:
            save_path = os.path.join(output_path, f"{player_name}_assist_xag.png")

        fig.savefig(save_path, dpi=400, bbox_inches='tight', facecolor=fig.get_facecolor())


        # Save BEFORE plt.show()
        plt.show()

    # Main consolidated function
    def process_and_plot_player_summary(self, Cannoniq_DB, team_categories, player_name):

        df_timeseries = self.get_player_league_timeseries(Cannoniq_DB, player_name)

        team_to_category = {}
        for category, teams in team_categories.items():
            for team in teams:
                team_to_category[team] = category

        df_timeseries["opponent_type"] = df_timeseries["Opponent"].map(team_to_category).fillna("uncategorized")

        metrics = [
            'Gls', 'Ast', 'xG', 'npxG', 'xAG', 'SoT',
            'SCA', 'GCA', 'Cmp', 'Cmp%', 'PrgP', 'Carries', 'PrgC', 'Succ', 'Min',
            'Tkl', 'Int', 'Blocks', 'Touches'
        ]

        agg_funcs = {col: 'sum' for col in metrics if col != 'Cmp%'}
        agg_funcs['Cmp%'] = 'mean'

        summary_by_type = df_timeseries.groupby("opponent_type")[metrics].agg(agg_funcs).reset_index()
        match_counts = df_timeseries.groupby("opponent_type").size().reset_index(name='match_count')
        summary_by_type = summary_by_type.merge(match_counts, on='opponent_type')

        per90_cols = [col for col in metrics if col not in ['Cmp%', 'Min']]
        per90_df = summary_by_type.copy()

        for col in per90_cols:
            per90_df[col] = (per90_df[col] / per90_df['Min']) * 90

        cols_to_keep = ['opponent_type', 'match_count', 'Cmp%', 'Min'] + per90_cols
        per90_df = per90_df[cols_to_keep].round(2)

        df_player_series = df_timeseries.groupby(['Player', 'team']).sum().reset_index().assign(
            difference=lambda x: x.Gls - x.xG
        )
        fm_ids = pd.read_csv("/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/Top6_leagues_fotmob_ids.csv")
        fm_ids = fm_ids[["team", "team_id"]]
        df_player_series = df_player_series.merge(fm_ids, on='team', how='left')
        df_player_series = df_player_series[['Player', 'team', 'Gls', 'xG', 'Ast', 'xAG','team_id', 'difference']]
        df_player_series['Ast_difference'] = df_player_series['Ast'] - df_player_series['xAG']

        team_id = df_player_series['team_id'].iloc[0]

        # Save directory
        output_path = f"Player_profiles/{player_name}/attacking_performance/"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Plotting
        self.plot_per90_summary(per90_df, player_name)
        self.plot_cumulative_chart(df_player_series, df_timeseries, player_name, plot_type='goals')
        self.plot_cumulative_chart(df_player_series, df_timeseries, player_name, plot_type='assists')


        print(f"[INFO] Summary plots for {player_name} complete.")

        return team_id


    def combine_attacking_performance_plots(self, player_name, team_id):
        base_path = f"Player_profiles/{player_name}/attacking_performance/"
        
        per90_path = os.path.join(base_path, f"{player_name}_performance_types.png")
        goals_path = os.path.join(base_path, f"{player_name}_goals_xg.png")
        assists_path = os.path.join(base_path, f"{player_name}_assist_xag.png")
        output_path = os.path.join(base_path, f"{player_name}_attacking_combined.png")

        # Open images
        per90_img = Image.open(per90_path)
        goals_img = Image.open(goals_path)
        assists_img = Image.open(assists_path)

        # Resize goals and assists to fit
        max_width = max(goals_img.width, assists_img.width)
        goals_img = goals_img.resize((max_width, per90_img.height // 2))
        assists_img = assists_img.resize((max_width, per90_img.height // 2))

        combined_right = Image.new("RGB", (max_width, per90_img.height), (237, 237, 237))
        combined_right.paste(goals_img, (0, 0))
        combined_right.paste(assists_img, (0, goals_img.height))

        total_width = per90_img.width + combined_right.width
        total_height = per90_img.height

        combined_final = Image.new("RGB", (total_width, total_height), (237, 237, 237))
        combined_final.paste(per90_img, (0, 0))
        combined_final.paste(combined_right, (per90_img.width, 0))

        # Create final figure
        fig, ax = plt.subplots(figsize=(total_width / 100, total_height / 100), dpi=100)
        ax.axis('off')
        ax.imshow(combined_final)

        # -- Club logo (top-left)
        try:
            logo_ax = fig.add_axes([0.11, 0.92, 0.1, 0.15])
            fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
            club_icon = Image.open(urllib.request.urlopen(f'{fotmob_url}{int(team_id)}.png'))
            logo_ax.imshow(club_icon)
            logo_ax.axis('off')
        except Exception as e:
            print(f"[WARN] Could not load club logo for {player_name}: {e}")

        # -- Title and subtitle (top-center)
        fig_text(
            x=0.5, y=1, 
            s=f'{player_name} – Attacking Performance Summary',
            ha='center', va='bottom',
            fontsize=60, weight='bold', color='black'
        )

        fig_text(
            x=0.5, y=0.97, 
            s='Includes <per 90 breakdown> by opponent category, <cumulative output> vs xG/xAG trends | Data from FBREF | Viz by @stephenaq7',
            ha='center', va='bottom',
            highlight_textprops=[
                {'weight': 'bold', 'color': '#287271'},
                {'weight': 'bold', 'color': '#D81159'}
            ],
            fontsize=45, color='#4E616C', font='DejaVu Sans'
        )

        fig_text(
            x=0.5, y=0.94, 
            s='Opponent categories: <Big Teams> = UCL clubs, <2nd Tier> = EL/ECL clubs, <Mid-Table> = 8th–13th, <Relegation> = bottom 6 at time of match',
            ha='center', va='bottom',
            highlight_textprops=[
                {'weight': 'bold', 'color': '#003f5c'},   # Big Teams
                {'weight': 'bold', 'color': '#58508d'},   # 2nd Tier
                {'weight': 'bold', 'color': '#bc5090'},   # Mid-table
                {'weight': 'bold', 'color': '#ffa600'}    # Relegation
            ],
            fontsize=40, color='#4E616C', font='DejaVu Sans'
        )


        # -- Personal logo (top-right)
        try:
            ax_logo = fig.add_axes([0.83, 0.92, 0.1, 0.15])
            brand_img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
            ax_logo.imshow(brand_img)
            ax_logo.axis('off')
        except Exception as e:
            print(f"[WARN] Could not load personal branding logo: {e}")

        # Save
        fig.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.show()

        return output_path

    def generate_attacking_summary(self, Cannoniq_DB, team_categories, player_name):
        """
        Full pipeline to fetch player match logs, compute attacking summaries,
        generate charts, and output a combined attacking performance image.

        Returns:
            int: FotMob team ID for the player's club.
        """
        team_id = self.process_and_plot_player_summary(Cannoniq_DB, team_categories, player_name)
        self.combine_attacking_performance_plots(player_name, team_id)
        return team_id
    
    def historic_performance(self, Cannoniq_DB, player_name, num_seasons=2):
        """
        Get player performance data for the current season and previous N seasons from fbref.com
        
        Args:
            Cannoniq_DB: DataFrame containing player data with 'match_logs' column
            player_name: Name of the player to get data for
            num_seasons: Number of previous seasons to retrieve (default: 3)
                        Note: This will get current season + 3 previous = 4 seasons total
        
        Returns:
            DataFrame with consolidated match data across all seasons
        """
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        import warnings
        import time
        import re
        
        # Validate inputs
        if 'match_logs' not in Cannoniq_DB.columns:
            raise ValueError("'match_logs' column not found in Cannoniq_DB.")
        
        player_row = Cannoniq_DB[Cannoniq_DB['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"Player '{player_name}' not found in Cannoniq_DB.")
        
        # Get the base URL
        raw_links = player_row['match_logs'].values[0]
        
        if isinstance(raw_links, str):
            base_url = raw_links.strip()
        elif isinstance(raw_links, list):
            base_url = raw_links[0].strip()
        else:
            raise ValueError("Unsupported format in 'match_logs'. Must be string or list of URLs.")
        
        # Extract current season from URL and generate URLs for previous seasons
        current_season_match = re.search(r'/(\d{4}-\d{4})/', base_url)
        if not current_season_match:
            raise ValueError("Could not extract season from URL format")
        
        current_season = current_season_match.group(1)
        current_start_year = int(current_season.split('-')[0])
        
        # Generate URLs for current season and previous seasons
        match_data_urls = []
        
        # Add current season first
        match_data_urls.append(base_url)
        
        # Add previous seasons
        for i in range(1, num_seasons + 1):
            prev_start_year = current_start_year - i
            prev_end_year = prev_start_year + 1
            prev_season = f"{prev_start_year}-{prev_end_year}"
            prev_url = base_url.replace(current_season, prev_season)
            match_data_urls.append(prev_url)
        
        # Fetch and process data for all seasons
        data_append = []
        
        for url in match_data_urls:
            print(f"Fetching: {url}")
            warnings.filterwarnings("ignore")
            
            try:
                page = requests.get(url)
                page.raise_for_status()  # Raise exception for bad status codes
                soup = BeautifulSoup(page.content, 'html.parser')
                
                html_content = page.text.replace('<!--', '').replace('-->', '')
                df = pd.read_html(html_content)
                
                try:
                    df[0].columns = df[0].columns.droplevel(0)
                except ValueError:
                    pass
                
                stats = df[0]
                stats = stats[
                    (stats['Comp'].isin(['La Liga', 'Premier League', 'Bundesliga', 'Serie A', 'Ligue 1', 'Primeira Liga'])) &
                    (stats['Pos'] != "On matchday squad, but did not play")
                ]
                
                keep_cols = [
                    'Opponent', 'Start', 'Pos', 'Date', 'Gls', 'Ast', 'xG', 'npxG', 'xAG', 'SoT',
                    'Squad', 'CrdY', 'CrdR', 'Result', 'SCA', 'GCA', 'Cmp', 'Cmp%', 'Min',
                    'PrgP', 'Carries', 'PrgC', 'Succ', 'Tkl', 'Int', 'Blocks', 'Touches'
                ]
                season = stats[keep_cols].copy()
                
                # Handle percentage columns
                if season['Cmp%'].dtype == 'object':
                    season['Cmp%'] = season['Cmp%'].str.replace('%', '', regex=False).astype(float)
                
                # Convert numeric columns
                columns_to_convert = [
                    'Gls', 'Ast', 'xG', 'npxG', 'xAG', 'SoT', 'SCA', 'GCA', 'Cmp', 'Min',
                    'Cmp%', 'PrgP', 'Carries', 'PrgC', 'Succ', 'Tkl', 'Int', 'Blocks', 'Touches'
                ]
                for col in columns_to_convert:
                    if col in season.columns and isinstance(season[col], pd.Series):
                        season[col] = pd.to_numeric(season[col], errors='coerce').fillna(0).astype(float)
                    else:
                        season[col] = 0.0
                
                season = season.rename({'Squad': 'team'}, axis=1)
                season['Player'] = player_name
                
                # Extract and add season information
                season_match = re.search(r'/(\d{4}-\d{4})/', url)
                if season_match:
                    season['Season'] = season_match.group(1)
                time.sleep(10)  # Be respectful to the server
                data_append.append(season)
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")
                continue
            except Exception as e:
                print(f"Error processing {url}: {e}")
                continue
                
            time.sleep(1)  # Be respectful to the server
        
        if not data_append:
            raise ValueError("No data could be retrieved for any season")
        
        # Concatenate all seasons
        df_total = pd.concat(data_append, ignore_index=True)
        
        # Final data type conversions
        df_total['CrdY'] = pd.to_numeric(df_total['CrdY'], errors='coerce').fillna(0).astype(int)
        df_total['CrdR'] = pd.to_numeric(df_total['CrdR'], errors='coerce').fillna(0).astype(int)
        df_total['Min'] = pd.to_numeric(df_total['Min'], errors='coerce').fillna(0).astype(int)
        fm_ids = pd.read_csv("/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/Top6_leagues_fotmob_ids.csv")
        fm_ids = fm_ids[["team", "team_id"]]
        df_total = df_total.merge(fm_ids, on='team', how='left')
        
        # Sort by season and date
        df_total = df_total.sort_values(['Season', 'Date'], ascending=[False, True])
        
        return df_total

    def plot_player_performance(self, historic_df, player_name, window=10):
        """
        Plot comprehensive player performance metrics with moving averages
        
        Args:
            historic_df: DataFrame with historic player performance data
            player_name: Name of the player
            window: Moving average window size (default: 5)
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
        from matplotlib import style
        import pandas as pd
        from PIL import Image
        import urllib.request
        from highlight_text import fig_text
        from matplotlib import image
        
        # Sort by season and date to ensure proper chronological order
        df = historic_df.sort_values(['Season', 'Date'], ascending=[True, True]).reset_index(drop=True)
        
        # Calculate cumulative and moving averages
        df['cumulative_goals'] = df['Gls'].cumsum()
        df['cumulative_xg'] = df['xG'].cumsum()
        df['cumulative_assists'] = df['Ast'].cumsum()
        df['cumulative_xa'] = df['xAG'].cumsum()
        
        # 5-game moving averages
        df['ma_xg'] = df['xG'].rolling(window=window, min_periods=1).mean()
        df['ma_goals'] = df['Gls'].rolling(window=window, min_periods=1).mean()
        df['ma_assists'] = df['Ast'].rolling(window=window, min_periods=1).mean()
        df['ma_xa'] = df['xAG'].rolling(window=window, min_periods=1).mean()
        df['ma_sca'] = df['SCA'].rolling(window=window, min_periods=1).mean()
        df['ma_gca'] = df['GCA'].rolling(window=window, min_periods=1).mean()
        df['ma_sot'] = df['SoT'].rolling(window=window, min_periods=1).mean()
        df['ma_tackles_int'] = (df['Tkl'] + df['Int']).rolling(window=window, min_periods=1).mean()
        df['ma_cmp_percent'] = df['Cmp%'].rolling(window=window, min_periods=1).mean()
        
        # Create match index
        df['match_index'] = range(len(df))
        
        # Find season boundaries for vertical lines
        season_boundaries = []
        seasons_list = df['Season'].unique()
        for i, season in enumerate(seasons_list):  # Skip first season
            boundary = df[df['Season'] == season].index[0]
            season_boundaries.append(boundary)
        
        # Get team info
        team_id = df['team_id'].iloc[-1]
        team_name = df['team'].iloc[-1]
        
        # Set up the plot
        style.use('fivethirtyeight')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        fig.suptitle(f'{player_name} Performance Analysis', fontsize=22, fontweight='bold', y=1.02)
        
        main_color = '#0057B8'
        
        # Plot 1: Cumulative Goals vs xG
        ax1 = axes[0, 0]
        ax1.plot(df['match_index'], df['cumulative_goals'], label='Cumulative Goals', color='red', linewidth=2)
        ax1.plot(df['match_index'], df['cumulative_xg'], label='Cumulative xG', color='blue', linewidth=2)
        ax1.set_title('Cumulative Goals vs xG', fontweight='bold', fontsize=10)
        ax1.set_ylabel('Cumulative Count', fontsize=9)
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle='dotted', alpha=0.7)
        
        # Plot 2: Cumulative Assists vs xA
        ax2 = axes[0, 1]
        ax2.plot(df['match_index'], df['cumulative_assists'], label='Cumulative Assists', color='green', linewidth=2)
        ax2.plot(df['match_index'], df['cumulative_xa'], label='Cumulative xA', color='orange', linewidth=2)
        ax2.set_title('Cumulative Assists vs xA', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Cumulative Count', fontsize=9)
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle='dotted', alpha=0.7)
        
        # Plot 3: 5-Game MA SCA and GCA
        ax3 = axes[0, 2]
        ax3.plot(df['match_index'], df['ma_sca'], label='10-Game MA SCA', color='purple', linewidth=2)
        ax3.plot(df['match_index'], df['ma_gca'], label='10-Game MA GCA', color='brown', linewidth=2)
        ax3.set_title('10-Game Moving Average: SCA & GCA', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Average per Game', fontsize=9)
        ax3.legend(fontsize=8)
        ax3.grid(True, linestyle='dotted', alpha=0.7)
        
        # Plot 4: 5-Game MA Shots on Target
        ax4 = axes[1, 0]
        ax4.plot(df['match_index'], df['ma_sot'], label='10-Game MA SoT', color='darkred', linewidth=2)
        ax4.set_title('10-Game Moving Average: Shots on Target', fontweight='bold', fontsize=10)
        ax4.set_ylabel('Average per Game', fontsize=9)
        ax4.set_xlabel('Match Index', fontsize=9)
        ax4.legend(fontsize=8)
        ax4.grid(True, linestyle='dotted', alpha=0.7)
        
        # Plot 5: 5-Game MA Tackles + Interceptions
        ax5 = axes[1, 1]
        ax5.plot(df['match_index'], df['ma_tackles_int'], label='10-Game MA Tackles + Int', color='darkgreen', linewidth=2)
        ax5.set_title('10-Game Moving Average: Tackles + Interceptions', fontweight='bold', fontsize=10)
        ax5.set_ylabel('Average per Game', fontsize=9)
        ax5.set_xlabel('Match Index', fontsize=9)
        ax5.legend(fontsize=8)
        ax5.grid(True, linestyle='dotted', alpha=0.7)
        
        # Plot 6: 5-Game MA Pass Completion %
        ax6 = axes[1, 2]
        ax6.plot(df['match_index'], df['ma_cmp_percent'], label='10-Game MA Pass %', color='navy', linewidth=2)
        ax6.set_title('10-Game Moving Average: Pass Completion %', fontweight='bold', fontsize=10)
        ax6.set_ylabel('Pass Completion %', fontsize=9)
        ax6.set_xlabel('Match Index', fontsize=9)
        ax6.legend(fontsize=8)
        ax6.grid(True, linestyle='dotted', alpha=0.7)
        
        all_axes = axes.flat
        for ax in all_axes:
            for boundary in season_boundaries:
                ax.axvline(x=boundary, color='grey', linestyle='--', alpha=0.5, linewidth=1)

        # Add season labels to all plots (separate loop)
        for ax in all_axes:
            # Add label for the first season at the beginning
            first_season = seasons_list[0]
            ax.annotate(
                f'{first_season}',
                xy=(0, ax.get_ylim()[1] * 0.95),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=10,
                color='grey',
                rotation=90,
                ha='left',
                va='top',
                alpha=0.8
            )
            
            # Then add labels for remaining seasons at their boundaries
            for i, boundary in enumerate(season_boundaries):
                if i + 1 < len(seasons_list):  # Safety check to prevent IndexError
                    season_name = seasons_list[i + 1]
                    ax.annotate(
                        f'{season_name}',
                        xy=(boundary, ax.get_ylim()[1] * 0.95),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=10,
                        color='grey',
                        rotation=90,
                        ha='left',
                        va='top',
                        alpha=0.8
                    )
        # Add team logo
        try:
            fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
            club_icon = Image.open(urllib.request.urlopen(f"{fotmob_url}{team_id}.png"))
            
            # Add logo to upper left corner
            ax_logo = fig.add_axes([0.01, 0.9, 0.12, 0.12])
            ax_logo.imshow(club_icon)
            ax_logo.axis('off')
            
        except Exception as e:
            print(f"Could not load team logo: {e}")
        
        # Add subtitle with player and team info
        fig.text(0.5, 0.98, f'{team_name} | {len(df)} matches across {len(seasons_list)} seasons', 
                ha='center', va='center', fontsize=15, style='italic')
        
        fig.text(0.5, 0.94, 'Analysis of cumulative goals/assists vs expected values and rolling averages for shot creation, defensive actions, and passing accuracy', 
                ha='center', va='center', fontsize=10, style='normal', color='grey')
        
        ax7 = fig.add_axes([0.85, 0.9, 0.11, 0.14])
        ax7.axis('off')
        img = plt.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
        ax7.imshow(img)

        output_dir = f"Player_profiles/{player_name}/Seasonal_Performance/"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{player_name}_Historic_Performance.png")

        plt.savefig(save_path, dpi=500, facecolor="#EBEBE9", bbox_inches="tight", edgecolor="none", transparent=False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3)
        plt.show()
        
        return fig
    
    def analyze_player_performance(self, Cannoniq_DB, player_name, num_seasons=2):
        """
        Get historic performance data and create visualization in one step
        """
        performance_data = self.historic_performance(Cannoniq_DB, player_name, num_seasons)
        return self.plot_player_performance(performance_data, player_name)


    






player_role_templates = {
    "Ball Playing CB": [
        # Defensive
        "Tkl%", "Int", "Blocks", "Passes Blocked", "Clr",
        # Possession
        "Total - Cmp%", "Short - Cmp%", "Long - Cmp%", "PrgP", "Carries - PrgDist",
        # Attacking
        "xA", "Ast", "KP", "Sh/90", "G/Sh"
    ],
    "Classic CB": [
        # Defensive
        "Tkl%", "Int", "Clr", "Recov", "Shots Blocked",
        # Possession
        "Touches", "Total - Att", "Total - Cmp%", "Long - Cmp%", "Rec",
        # Attacking
        "xG", "Sh/90", "Ast", "SoT%", "KP"
    ],
    "Classic Fullback": [
        # Defensive
        "Tkl%", "Int", "Blocks", "Tkl+Int", "Passes Blocked",
        # Possession
        "Touches", "Carries", "Carries - TotDist", "PrgR", "Short - Cmp%",
        # Attacking
        "xA", "Ast", "KP", "CrsPA", "1/3"
    ],
    "Inverted Fullback": [
        # Defensive
        "Tkl%", "Int", "Blocks", "Passes Blocked", "Dribblers- Tkl",
        # Possession
        "Total - Cmp%", "Short - Cmp%", "Carries - PrgDist", "PrgP", "Touches",
        # Attacking
        "xA", "KP", "Ast", "Sh/90", "SCA90"
    ],
    "Attacking Fullback": [
        # Defensive
        "Tkl%", "Tkl+Int", "Blocks", "Int", "Dribblers- Tkl",
        # Possession
        "Carries - PrgC", "Carries - PrgDist", "Touches", "Take Ons - Attempted", "PrgR",
        # Attacking
        "xA", "KP", "CrsPA", "Ast", "SCA - PassLive"
    ],
    "Destroyer DM": [
        # Defensive
        "Tkl+Int", "Int", "Passes Blocked", "Passes Blocked", "Blocks",
        # Possession
        "Touches", "Total - Cmp%", "Carries - PrgDist", "Short - Cmp%", "Rec",
        # Attacking
        "xA", "Long - Cmp%", "Ast", "Sh/90", "KP"
    ],
    "Deep Lying Playmaker CM": [
        # Defensive
        "Tkl%", "Passes Blocked", "Tkl+Int", "Shots Blocked", "Int",
        # Possession
        "Total - Cmp%", "Short - Cmp%", "PrgP", "Carries - PrgDist", "Touches",
        # Attacking
        "xA", "Long - Cmp%", "Ast", "Sh/90", "KP"
    ],
    "Box to Box CM": [
        # Defensive
        "Tkl%", "Tkl+Int", "Int", "Blocks", "Passes Blocked",
        # Possession
        "Carries - PrgDist", "Carries - PrgC", "Touches", "Short - Cmp%", "Rec",
        # Attacking
        "xG", "xA", "KP", "Ast", "Sh/90"
    ],
    "Playmaker CM": [
        # Defensive
        "Tkl%", "Int", "Passes Blocked", "Blocks", "Tkl+Int",
        # Possession
        "Touches", "Carries - PrgDist", "Short - Cmp%", "Total - Cmp%", "PrgP",
        # Attacking
        "xA", "Ast", "KP", "SCA90", "GCA90"
    ],
    "Classic AM": [
        # Defensive
        "Tkl%", "Tkl+Int", 'Tackles - Att 3rd', "Int", "Blocks",
        # Possession
        "Touches", "Carries - PrgC", "Short - Cmp%", "Total - Cmp%", "Rec",
        # Attacking
        "xA", "Ast", "KP", "Sh/90", "xG"
    ],
    "Inside Forward": [
        # Defensive
        "Tkl%", "Tkl+Int", "Int", "Blocks", "Dribblers- Tkl",
        # Possession
        "Carries - PrgC", "Carries - PrgDist", "Touches", "Take Ons - Attempted", "PrgR",
        # Attacking
        "xG", "xA", "KP", "Ast", "Sh/90"
    ],
    "Winger": [
        # Defensive
        "Tkl%", "Tkl+Int", "Blocks", "Int", "Dribblers- Tkl",
        # Possession
        "Carries - PrgDist", "Carries - PrgC", "Touches", "Take Ons - Attempted", "1/3",
        # Attacking
        "xA", "KP", "Ast", "CrsPA", "Sh/90"
    ],
    "Center Forward": [
        # Defensive
        "Tkl%", "Int", "Tkl+Int", "Blocks", "Passes Blocked",
        # Possession
        "Touches", "Rec", "Carries - PrgC", "Short - Cmp%", "Fld",
        # Attacking
        "xG", "xA", "KP", "Ast", "Sh/90"
    ],
    "False 9": [
        # Defensive
        "Tkl%", "Tkl+Int", "Int", "Blocks", "Passes Blocked",
        # Possession
        "Touches", "Total - Cmp%", "Short - Cmp%", "Carries - PrgC", "Rec",
        # Attacking
        "xG", "xA", "KP", "Ast", "SCA - PassLive"
    ]
}


role_to_position_group = {
    "Ball Playing CB": "Defender",
    "Classic CB": "Defender",
    "Classic Fullback": "Defender",
    "Inverted Fullback": "Defender",
    "Attacking Fullback": "Wing-Back",
    "Destroyer DM": "Central Midfielders",
    "Deep Lying Playmaker CM": "Central Midfielders",
    "Box to Box CM": "Central Midfielders",
    "Playmaker CM": "Central Midfielders",
    "Classic AM": "Central Midfielders",
    "Inside Forward": "Forwards",
    "Winger": "Forwards",
    "Center Forward": "Forwards",
    "False 9": "Forwards"
}


clean_stat_mapping = {
    # Defensive
    "Tkl%": "Tackle Success %",
    "Int": "Interceptions",
    "Blocks": "Blocks",
    "Passes Blocked": "Passes Blocked",
    "Clr": "Clearances",
    "Recov": "Recoveries",
    "Shots Blocked": "Shots Blocked",
    "Tkl+Int": "Tackles + Interceptions",
    "Dribblers- Tkl": "Tackles vs Dribblers",
    "Pass": "Passes Blocked",
    "CrdY": "Yellow Cards",
    "Fls": "Fouls Committed",

    # Possession
    "Touches": "Touches",
    "Carries": "Carries",
    "Carries - TotDist": "Carry Distance (Total)",
    "Carries - PrgDist": "Progressive Carry Distance",
    "Carries - PrgC": "Progressive Carries",
    "Carries - 1/3": "Carries into Final Third",
    "Carries - CPA": "Carries into Penalty Area",
    "Take Ons - Attempted": "Dribbles Attempted",
    "Rec": "Passes Received",
    "Total - Cmp%": "Passing % (Total)",
    "Total - Att": "Passes Attempted (Total)",
    "Short - Cmp%": "Passing % (Short)",
    "Long - Cmp%": "Passing % (Long)",
    "PrgR": "Progressive Receptions",
    "PrgP": "Progressive Passes",
    "Fld": "Fouls Drawn",

    # Attacking
    "xA": "Expected Assists (xA)",
    "Ast": "Assists",
    "KP": "Key Passes",
    "Sh/90": "Shots per 90",
    "xG": "Expected Goals (xG)",
    "CrsPA": "Crosses into Penalty Area",
    "SCA90": "Shot Creating Actions /90",
    "GCA90": "Goal Creating Actions /90",
    "1/3": "Passes into Final Third",
    "PPA": "Passes into Penalty Area",
    "SoT%": "Shots on Target %",
    "G/Sh": "Goals per Shot",
    "npxG": "Non-Penalty xG",
    "Long - Cmp%": "Passing % (Long)",
    "SCA - PassLive": "SCA – Live Pass",
    "GCA - PassLive": "GCA – Live Pass",
    "SCA - PassDead": "SCA – Dead Ball",
    "GCA - PassDead": "GCA – Dead Ball"
}

team_categories = {
    'big_teams': [
        'Real Madrid', 'Barcelona', 'Atlético Madrid', 'Juventus', 'Roma',
        'Napoli', 'Inter', 'Bayern Munich', 'Arsenal', 'Chelsea',
        'Liverpool', 'Manchester City','Manchester Utd', 'Milan','Dortmund', 'Sporting CP', 'Porto', 'Benfica'
    ],
    'european_2nd_tier': [ 'Braga',

         'Aston Villa','Athletic Club','Sevilla', 'Villarreal', 'Real Sociedad', 'Fiorentina', 'Lazio','Lille','Eint Frankfurt',
        'Atalanta', 'Marseille', 'Nice', 'Monaco', 'RB Leipzig', 'Leverkusen','Paris S-G','Tottenham', 'Newcastle Utd', 
    ],
    'mid_table': ['Betis', 'Getafe', 'Bologna', 'Torino',
        'Lens',  'Hoffenheim', 'Stuttgart', 'Freiburg',
        'Brentford', 'West Ham', 'Crystal Palace', 'Wolves', 'Brighton',
        'Alavés', 'Angers', 'Augsburg', 'Auxerre',
        'Bochum', 'Bournemouth', 'Brest', 'Cagliari', 'Darmstadt 98',  'Espanyol', 'Everton', 
        'Fulham', 'Genoa', 'Girona', 'Gladbach', 'Heidenheim',
        'Lecce', 'Leganés', 'Lyon','Boavista','Rio Ave', 'Santa Clara',
        'Mainz 05', 'Mallorca',  'Montpellier','Estoril','Vitória',
        'Monza', 'Nantes', "Nott'ham Forest", 'Osasuna',
        'Oviedo',  'Rayo Vallecano', 'Reims', 'Rennes','Famalicão',
        'Saint-Étienne', 'Strasbourg', 'Toulouse','Moreirense',
        'Udinese', 'Union Berlin', 'Valencia',  'Werder Bremen','Getafe'
        'Wolfsburg'],
    'relegation_teams': [
        'Celta Vigo', 'Como','Parma','Farense','Venezia','Salernitana','Cesena','Empoli', 'Sassuolo', 'Metz','Valladolid','Köln', 'Las Palmas', 'Le Havre','Hellas Verona', 
        'Ipswich Town','Nacional','Frosinone','Southampton', 'AVS Futebol','Gil Vicente FC','Leicester City', 'Lorient', 'St. Pauli','Holstein Kiel', 'Estrela','Casa Pia', 'Gil Vicente', 'Marítimo', 'Portimonense', 'Vizela', 'Tondela', 'Arouca', 'Rio Ave', 'Santa Clara'
    ]
}





# charts.generate_full_player_profile_with_pizza(player_name="Eberechi Eze", role="Inside Forward", df=Cannoniq_DB)
# charts.plot_role_based_comparison("Eberechi Eze", role="Inside Forward", df=Cannoniq_DB, comparative_list=["Rodrygo"])

# df_result = charts.generate_full_similarity_analysis(
#     player_name='Eberechi Eze',
#     role='Inside Forward',
#     Cannoniq_DB=Cannoniq_DB
# )
# charts.create_player_scorecard_plot(player_name='Eberechi Eze' , role = 'Inside Forward', Cannoniq_DB=Cannoniq_DB)


