# # import os
# # import requests
# # import pandas as pd
# # from bs4 import BeautifulSoup
# # import seaborn as sb
# # import matplotlib.pyplot as plt
# # import matplotlib as mpl
# # import warnings
# # import numpy as np
# # from math import pi
# # from urllib.request import urlopen
# # import matplotlib.patheffects as pe
# # from highlight_text import fig_text
# # from adjustText import adjust_text
# # from tabulate import tabulate
# # import matplotlib.style as style
# # import unicodedata
# # from fuzzywuzzy import fuzz
# # from fuzzywuzzy import process
# # import matplotlib.ticker as ticker
# # import matplotlib.patheffects as path_effects
# # import matplotlib.font_manager as fm
# # import matplotlib.colors as mcolors
# # from matplotlib import cm
# # from highlight_text import fig_text

# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # # %matplotlib inline

# # from sklearn.cluster import KMeans
# # from sklearn import preprocessing
# # from sklearn.decomposition import PCA
# # from sklearn.preprocessing import MinMaxScaler


# # from mplsoccer import PyPizza
# # import matplotlib.pyplot as plt
# # import matplotlib.image as image
# # import pandas as pd
# # import numpy as np
# # import os

# # style.use('fivethirtyeight')

# # from PIL import Image
# # import urllib
# # import os
# # import math
# # from PIL import Image
# # import matplotlib.image as image
# # pd.options.display.max_columns = None

# # from highlight_text import fig_text
# # from adjustText import adjust_text
# # # from soccerplots.radar_chart import Radar
# # from mplsoccer import PyPizza, add_image, FontManager

# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import matplotlib.ticker as ticker
# # import matplotlib.gridspec as gridspec
# # import matplotlib.style as style
# # import numpy as np


# # import matplotlib.pyplot as plt
# # from PIL import Image
# # from matplotlib import image
# # from mplsoccer import Radar
# # from highlight_text import fig_text

# # style.use('fivethirtyeight')


# # warnings.filterwarnings('ignore') 

# # import sys
# # import os

# # # Add the directory containing FBREF_Aggregations.py to the Python path
# # module_path = os.path.abspath(os.path.join('/Users/stephenahiabah/Desktop/Code/cannoniq'))
# # if module_path not in sys.path:
# #     sys.path.append(module_path)

# # import FBREF_Aggregations as fbref

# # font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
# #                           'src/hinted/Roboto-Regular.ttf')
# # font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
# #                           'src/hinted/Roboto-Italic.ttf')
# # font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
# #                         'RobotoSlab[wght].ttf')


# # def create_cannoniq_database(
# #     phonebook_path='/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/24-25player_phonebook.csv',
# #     output_path='/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/CannonIQ_DB.csv'
# # ):
# #     # Initialize database class
# #     fb_ref_db = fbref.CreateFBRefDatabase()

# #     # Generate IQ scores
# #     CannonIQ_DB = fb_ref_db.generate_pitch_iq_scores()

# #     # Load player phonebook
# #     player_phonebook = pd.read_csv(phonebook_path)

# #     # Normalize names in CannonIQ_DB for merging
# #     CannonIQ_DB['player_name_match'] = CannonIQ_DB['Player'].apply(fb_ref_db.remove_accents)

# #     # Standardize phonebook column name
# #     player_phonebook = player_phonebook.rename(columns={'Player': 'player_name_match'})

# #     # Drop duplicates in phonebook (for merge integrity)
# #     player_phonebook = player_phonebook.drop_duplicates(subset='player_name_match', keep='first')

# #     # Merge scouting info into IQ DB
# #     CannonIQ_DB = pd.merge(
# #         CannonIQ_DB,
# #         player_phonebook[['player_name_match', 'scouting_url', 'match_logs']],
# #         on='player_name_match',
# #         how='left'
# #     )

# #     # Create and normalize full stat dataset
# #     full_stats = fb_ref_db.create_full_stats_db()
# #     full_stats = fb_ref_db.per_90fi(full_stats)

# #     # Final tidy-up before merge
# #     CannonIQ_DB = CannonIQ_DB.drop_duplicates(subset=['Player'], keep='first')
# #     CannonIQ_DB = CannonIQ_DB[[
# #         'Player', 'Nation', 'Pos', 'Squad', 'position_group',
# #         'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score',
# #         'player_name_match', 'scouting_url', 'match_logs'
# #     ]]

# #     # Merge per90 stats
# #     CannonIQ_DB = pd.merge(
# #         CannonIQ_DB,
# #         full_stats,
# #         on=['Player', 'Nation', 'Pos', 'Squad'],
# #         how='inner'
# #     )

# #     # Clean column names
# #     columns_to_drop = [col for col in CannonIQ_DB.columns if col.endswith('_y')]
# #     CannonIQ_DB = CannonIQ_DB.drop(columns=columns_to_drop)
# #     CannonIQ_DB.columns = [col.replace('_x', '') for col in CannonIQ_DB.columns]
# #     CannonIQ_DB = CannonIQ_DB.drop(columns=['TO'], errors='ignore')

# #     # Output
# #     CannonIQ_DB.to_csv(output_path, index=False)
# #     print(f'CannonIQ_DB saved to {output_path}')
    
# #     return CannonIQ_DB


# # # --- Radar plotting function ---
# # def plot_role_based_comparison(player_name, role, df, comparative_list=None):
# #     plt.style.use("fivethirtyeight")


# #     if role not in player_role_templates:
# #         raise ValueError(f"Unknown role '{role}'.")
    
# #     player_dir = f"Player_profiles/{player_name}"
# #     os.makedirs(player_dir, exist_ok=True)

# #     params = player_role_templates[role]

# #     # Use readable names for radar labels
# #     readable_params = [clean_stat_mapping.get(p, p) for p in params]

# #     main_player_row = df[df['Player'] == player_name]
# #     if main_player_row.empty:
# #         raise ValueError(f"{player_name} not found in dataset")

# #     pos_group = main_player_row['position_group'].values[0]
# #     scaled_df = df[df['position_group'] == pos_group]
# #     scaled_df = df[df['90s'] > 10]

# #     def get_player_data(name):
# #         row = df[df['Player'] == name]
# #         if row.empty:
# #             raise ValueError(f"{name} not found in dataset")
# #         return row[params].values.flatten().tolist()

# #     main_values = [float(x) for x in get_player_data(player_name)]

# #     if comparative_list:
# #         comp_values_list = [[float(x) for x in get_player_data(name)] for name in comparative_list]
# #     else:
# #         comp_values_list = []

# #     if not comp_values_list:
# #         max_values = scaled_df[params].max().tolist()
# #         low = [0 for _ in max_values]
# #         high = [float(val) for val in max_values]

# #         radar = Radar(readable_params, low, high, round_int=[False]*len(params),
# #                       num_rings=5, ring_width=1, center_circle_radius=1)

# #         fig, ax = radar.setup_axis()
# #         fig.patch.set_facecolor('#ededed')
# #         ax.set_facecolor('#ededed')

# #         radar.draw_circles(ax=ax, facecolor='#dddddd')
# #         radar.draw_radar(main_values, ax=ax, kwargs_radar={'facecolor': '#aaaaaa', 'alpha': 0.65})
# #         radar.draw_range_labels(ax=ax, fontsize=15)
# #         radar.draw_param_labels(ax=ax, fontsize=15)
# #         ax.legend([player_name], loc='upper right', fontsize=12)

# #         club = main_player_row['Squad'].values[0]
# #         player = main_player_row['Player'].values[0]
# #         age = float(main_player_row['Age'].values[0])
# #         nineties = float(main_player_row["90s"].values[0])
# #         season = "2024–2025"
# #         role_label = role.replace("_", " ").title()
# #         pos_group = main_player_row['position_group'].values[0]

# #         fig.savefig(f"{player_dir}/{player_name}_radar.png", dpi=300, bbox_inches="tight")


# #         fig_text(
# #             x=0.66, y=0.93,
# #             s=f"{club} | {player}\n"
# #               f"90's Played: {nineties:.1f} | Age: {age:.1f}\n"
# #               f"Season: {season}\n"
# #               f"{role_label} Template compared to {pos_group}",
# #             va="bottom", ha="right",
# #             fontsize=14, color="black", weight="book"
# #         )


# #     else:
# #         for i, comp_values in enumerate(comp_values_list):
# #             low = [min(m, c) * 0.5 for m, c in zip(main_values, comp_values)]
# #             high = [max(m, c) * 1.05 for m, c in zip(main_values, comp_values)]

# #             radar = Radar(readable_params, low, high, round_int=[False]*len(params),
# #                           num_rings=5, ring_width=1, center_circle_radius=1)

# #             fig, ax = radar.setup_axis()
# #             fig.patch.set_facecolor('#f0f0f0')
# #             ax.set_facecolor('#f0f0f0')

# #             radar.draw_circles(ax=ax, facecolor='#ffb2b2')
# #             radar.draw_radar_compare(main_values, comp_values, ax=ax,
# #                                      kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6},
# #                                      kwargs_compare={'facecolor': '#d80499', 'alpha': 0.6})
# #             radar.draw_range_labels(ax=ax, fontsize=15)
# #             radar.draw_param_labels(ax=ax, fontsize=15)
# #             ax.legend([player_name, comparative_list[i]], loc='upper right', fontsize=12)

# #             main_row = df[df['Player'] == player_name]
# #             comp_row = df[df['Player'] == comparative_list[i]]

# #             main_club = main_row['Squad'].values[0]
# #             main_name = main_row['Player'].values[0]
# #             main_age = float(main_row['Age'].values[0])
# #             main_90s = float(main_row['90s'].values[0])

# #             comp_name = comp_row['Player'].values[0]
# #             comp_age = float(comp_row['Age'].values[0])
# #             comp_90s = float(comp_row['90s'].values[0])

# #             season = "2024–2025"
# #             role_label = role.replace("_", " ").title()
# #             pos_group = main_row['position_group'].values[0]

# #             fig.savefig(f"{player_dir}/{main_name}_vs_{comp_name}_radar.png", dpi=300, bbox_inches="tight")


# #             fig_text(
# #                 x=0.65, y=0.93,
# #                 s=f"{main_club} | {main_name} vs {comp_name}\n"
# #                   f"Season: {season}\n"
# #                   f"{role_label} Template compared to {pos_group}s",
# #                 va="bottom", ha="right",
# #                 fontsize=14, color="black", weight="book"
# #             )


# #     try:
# #         # league_icon = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images/premier-league-2-logo.png")
# #         badge_img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')

# #         # league_ax = fig.add_axes([0.002, 0.89, 0.20, 0.15], zorder=1)
# #         # league_ax.imshow(league_icon)
# #         # league_ax.axis("off")

# #         ax3 = fig.add_axes([0.002, 0.89, 0.20, 0.15], zorder=1)
# #         ax3.axis('off')
# #         ax3.imshow(badge_img)

# #     except FileNotFoundError:
# #         print("Logo or badge image not found, skipping visual extras.")


# #     # Save radar plot

# #     plt.show()


# # def plot_role_based_kde(player_name, role, df):
# #     if role not in player_role_templates:
# #         raise ValueError(f"Invalid role '{role}'.")

# #     params = player_role_templates[role]
# #     player_row = df[df['Player'] == player_name]
# #     if player_row.empty:
# #         raise ValueError(f"{player_name} not found in dataset.")

# #     pos_group = player_row['position_group'].values[0]
# #     filtered_df = df[(df['position_group'] == pos_group) & (df['90s'] > 10)]

# #     num_params = len(params)
# #     fig, axs = plt.subplots(num_params, 1, figsize=(7, num_params * 1.2))
# #     plt.subplots_adjust(hspace=0.8) 

# #     for i, param in enumerate(params):
# #         ax = axs[i]

# #         data = pd.to_numeric(filtered_df[param], errors='coerce').dropna().astype(np.float64)
# #         player_val = float(player_row[param].values[0])
# #         percentile = (data < player_val).mean() * 100

# #         # Pad left to start KDE from 0
# #         data = np.concatenate(([0], data))

# #         # KDE lines
# #         sns.kdeplot(data, color="gray", ax=ax, linewidth=1)
# #         sns.kdeplot(data, fill=True, alpha=0.35, color="black", ax=ax, linewidth=0,
# #                     clip=(data.min(), player_val))

# #         # Player marker
# #         ax.axvline(player_val, color='red', linestyle='-', lw=1)
# #         ax.plot(player_val, 0, '^', color='red', markersize=6)

# #         # Axes cleanup
# #         ax.set_yticks([])
# #         ax.set_ylabel('')
# #         ax.set_xlabel('')
# #         ax.tick_params(axis='x', labelsize=6)
# #         ax.set_xlim(left=0)

# #         for spine in ax.spines.values():
# #             spine.set_visible(False)
# #         ax.xaxis.grid(True, linestyle=':', linewidth=0.5, color='gray')

# #         # Stat label as top-right "title"
# #         clean_label = clean_stat_mapping.get(param, param)
# #         ax.text(
# #             ax.get_xlim()[0],
# #             # + (ax.get_xlim()[1] * 0.1),

# #             ax.get_ylim()[1] * 1.2,
# #             f"{clean_label}: {player_val:.2f}",
# #             fontsize=10, ha='left', va='center', fontweight='book', color='black'
# #         )

# #         # Percentile rank below
# #         ax.text(
# #             ax.get_xlim()[1],
# #             ax.get_ylim()[1] * 1.2,
# #             f"Rk: {percentile:.1f}%",
# #             fontsize=10, ha='right', va='center', fontweight='book', color='red'
# #         )

# #     # Directory setup
# #     player_dir = f"Player_profiles/{player_name}"
# #     os.makedirs(player_dir, exist_ok=True)

# #     # Save KDE plot
# #     fig.savefig(f"{player_dir}/{player_name}_kde.png", dpi=300, bbox_inches="tight")

# #     plt.show()


# # from PIL import Image
# # import matplotlib.pyplot as plt

# # def combine_player_profile_plots(player_name, df,include_logos=True, title=None):

# #     main_player_row = df[df['Player'] == player_name]


# #     club = main_player_row['Squad'].values[0]
# #     player = main_player_row['Player'].values[0]
# #     age = float(main_player_row['Age'].values[0])
# #     nineties = float(main_player_row["90s"].values[0])
# #     season = "2024–2025"
# #     pos_group = main_player_row['position_group'].values[0]
    
# #     # Paths
# #     radar_path = f"Player_profiles/{player_name}/{player_name}_radar.png"
# #     kde_path = f"Player_profiles/{player_name}/{player_name}_kde.png"
# #     output_path = f"Player_profiles/{player_name}/{player_name}_combined_profile.png"

# #     # Open plots
# #     radar_img = Image.open(radar_path)
# #     kde_img = Image.open(kde_path)

# #     # Resize KDE to match radar height
# #     radar_w, radar_h = radar_img.size
# #     kde_img = kde_img.resize((int(kde_img.width), radar_h))
# #     kde_w, kde_h = kde_img.size

# #     # Create figure
# #     fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
# #     fig.patch.set_facecolor('#ededed')  # FiveThirtyEight-style background
# #     ax.axis('off')

# #     # Combine into one image in Matplotlib canvas
# #     combined_img = Image.new("RGB", (radar_w + kde_w, radar_h), (237, 237, 237))
# #     combined_img.paste(radar_img, (0, 50))  # Push radar down
# #     combined_img.paste(kde_img, (radar_w, 0))  # Push KDE right

# #     # Show base image on canvas
# #     ax.imshow(combined_img)

# #     # Logos (in axes)
# #     if include_logos:
# #         try:
# #             league_icon = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images/premier-league-2-logo.png")
# #             badge_img = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png")


# #             # League logo (left-top)
# #             ax_league = fig.add_axes([0.15, 0.88, 0.12, 0.18])  # ← Moved right (x: 0.01 → 0.03) and up (y: 0.82 → 0.87)
# #             ax_league.imshow(league_icon)
# #             ax_league.axis('off')

# #             # Badge logo (right-top)
# #             ax_badge = fig.add_axes([0.75, 0.88, 0.12, 0.18])   # ← Moved slightly left (x: 0.91 → 0.90) and up (y: 0.82 → 0.87)
# #             ax_badge.imshow(badge_img)
# #             ax_badge.axis('off')

# #         except FileNotFoundError:
# #             print("Logos not found – skipping logos.")

# #     # Title
# #     title = (
# #         f"{club} | {player}\n"
# #         f"90's Played: {nineties:.1f} | Age: {age:.1f}\n"
# #         f"Season: {season}\n"
# #         f"Template compared to {pos_group}s"
# #     )

# #     # add credits
# #     CREDIT_1 = "viz by @pitchiq.bsky.social\ndata via FBREF / Opta"
# #     CREDIT_2 = "inspired by: @cannonstats.com, @FootballSlices"

# #     fig.text(
# #         0.88, 0.00, f"{CREDIT_1}\n{CREDIT_2}", size=9,
# #         fontproperties=font_italic.prop, color="#000000",
# #         ha="right"
# #     )

# #     plt.suptitle(title, fontsize=13, fontweight='book', ha='left', x=0.25, y=1.02)

# #     # Save and show
# #     plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor())
# #     plt.show()

# #     return output_path






# # def create_percentile_Pizza(player_name, role, df, role_templates, output_dir):
# #     font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf')
# #     font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Italic.ttf')
# #     font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf')

# #     if role not in role_templates:
# #         raise ValueError(f"Invalid role '{role}'.")

# #     params = role_templates[role]
# #     readable_params = [clean_stat_mapping.get(p, p) for p in params]
# #     player_row = df[df['Player'] == player_name]
# #     if player_row.empty:
# #         raise ValueError(f"{player_name} not found in dataset.")

# #     pos_group = player_row['position_group'].values[0]
# #     filtered_df = df[(df['position_group'] == pos_group) & (df['90s'] > 10)]

# #     percentiles = []
# #     for param in params:
# #         data = pd.to_numeric(filtered_df[param], errors='coerce').dropna().astype(np.float64)
# #         player_val = float(player_row[param].values[0])
# #         percentile = round((data < player_val).mean() * 100)
# #         percentiles.append(percentile)
# #     print(role)

# #     if "cb" in role.lower():
# #         slice_colors = ["#8B0000"] * 5 + ["#B22222"] * 5 + ["#DC143C"] * 5
# #     elif any(pos in role.lower() for pos in ["cm", "dm"]):
# #         slice_colors = ["#097969"] * 5 + ["#AFE1AF"] * 5 + ["#088F8F"] * 5
# #     elif any(pos in role.lower() for pos in ["am", "winger"]):
# #         slice_colors = ["#00008B"] * 5 + ["#4169E1"] * 5 + ["#87CEFA"] * 5
# #     elif any(pos in role.lower() for pos in ["f9", "forward"]):
# #         slice_colors = ["#B0B0B0"] * 5 + ["#808080"] * 5 + ["#404040"] * 5

# #     else:
# #         slice_colors = ["#D70232"] * 5 + ["#FF9300"] * 5 + ["#1A78CF"] * 5
# #     text_colors = ["#000000"] * 15



# #     baker = PyPizza(
# #         params=readable_params,
# #         background_color="#EBEBE9",
# #         straight_line_color="#EBEBE9",
# #         straight_line_lw=1,
# #         last_circle_lw=0,
# #         other_circle_lw=0,
# #         inner_circle_size=20
# #     )

# #     fig, ax = baker.make_pizza(
# #         percentiles,
# #         figsize=(8, 8.5),
# #         color_blank_space="same",
# #         slice_colors=slice_colors,
# #         value_colors=text_colors,
# #         value_bck_colors=slice_colors,
# #         blank_alpha=0.4,
# #         kwargs_slices=dict(edgecolor="#F2F2F2", zorder=2, linewidth=1),
# #         kwargs_params=dict(color="#000000", fontsize=9, va="center", fontproperties=font_normal.prop),
# #         kwargs_values=dict(
# #             color="#000000", fontsize=11, fontproperties=font_normal.prop, zorder=3,
# #             bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1)
# #         )
# #     )

# #     club = player_row['Squad'].values[0]
# #     age = float(player_row['Age'].values[0])
# #     nineties = float(player_row['90s'].values[0])



# #     fig.text(0.05, 0.985, f"{player_name} - {club} - {role} Template", size=14, ha="left", fontproperties=font_bold.prop, color="#000000")
# #     fig.text(0.05, 0.963, f"Percentile Rank vs Top-Five League {pos_group} | Season 2024-25", size=10, ha="left", fontproperties=font_bold.prop, color="#000000")
# #     fig.text(0.08, 0.925, "Attacking          Possession        Distribution", size=12, fontproperties=font_bold.prop, color="#000000")

# #     # Legend color mapping based on role
# #     if "CB" in role:
# #         att_color = "#DC143C"       # light red
# #         pos_color = "#B22222"       # medium red
# #         def_color = "#8B0000"       # dark red
# #     elif "CM" in role or "DM" in role:
# #         att_color = "#088F8F"       # light green
# #         pos_color = "#AFE1AF"       # medium green
# #         def_color = "#097969"       # dark green
# #     elif "AM" in role or "Winger" in role:
# #         att_color = "#87CEFA"       # light blue
# #         pos_color = "#4169E1"       # medium blue
# #         def_color = "#00008B"       # dark blue
# #     elif "F9" in role or "Forward" in role:
# #         att_color = "#696969"       # light grey
# #         pos_color = "#A9A9A9"       # dark grey
# #         def_color = "#D3D3D3"       # dim grey
# #     else:
# #         att_color = "#1A78CF"
# #         pos_color = "#FF9300"
# #         def_color = "#D70232"

# #     # Update legend color rectangles
# #     fig.patches.extend([
# #         plt.Rectangle((0.05, 0.9225), 0.025, 0.021, fill=True, color=att_color, transform=fig.transFigure, figure=fig),
# #         plt.Rectangle((0.2, 0.9225), 0.025, 0.021, fill=True, color=pos_color, transform=fig.transFigure, figure=fig),
# #         plt.Rectangle((0.351, 0.9225), 0.025, 0.021, fill=True, color=def_color, transform=fig.transFigure, figure=fig),
# #     ])


# #     # Credits
# #     CREDIT_1 = "@cannoniq.bsky.com\ndata via FBREF / Opta"
# #     CREDIT_2 = "inspired by: @Worville, @FootballSlices"
# #     fig.text(0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9, fontproperties=font_italic.prop, color="#000000", ha="right")

# #     try:
# #         ax3 = fig.add_axes([0.80, 0.075, 0.15, 1.75])
# #         ax3.axis('off')
# #         img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
# #         ax3.imshow(img)
# #     except FileNotFoundError:
# #         print("Logo image not found.")

# #     os.makedirs(output_dir, exist_ok=True)
# #     save_path = os.path.join(output_dir, f"{player_name}_percentile_pizza.png")
# #     plt.savefig(save_path, dpi=500, facecolor="#EBEBE9", bbox_inches="tight", edgecolor="none", transparent=False)
# #     plt.show()

# #     return save_path


# # import time 
# # def generate_advanced_data(scout_links):
# #     appended_data_per90 = []
# #     appended_data_percent = []
# #     for x in scout_links:
# #         warnings.filterwarnings("ignore")
# #         url = x
# #         page =requests.get(url)
# #         soup = BeautifulSoup(page.content, 'html.parser')
# #         name = [element.text for element in soup.find_all("span")]
# #         name = name[7]
# #         html_content = requests.get(url).text
# #         df = pd.read_html(html_content)
# #         df[-1].columns = df[-1].columns.droplevel(0) # drop top header row
# #         stats = df[-1]
# #         # stats = df[0]
# #         advanced_stats = stats.loc[(stats['Statistic'] != "Statistic" ) & (stats['Statistic'] != ' ')]
# #         advanced_stats = advanced_stats.dropna(subset=['Statistic',"Per 90", "Percentile"])
# #         per_90_df = advanced_stats[['Statistic',"Per 90",]].set_index("Statistic").T
# #         per_90_df["Name"] = name
# #         percentile_df = advanced_stats[['Statistic',"Percentile",]].set_index("Statistic").T
# #         percentile_df["Name"] = name
# #         appended_data_per90.append(per_90_df)
# #         appended_data_percent.append(percentile_df)
# #         del df, soup
# #         time.sleep(10)
# #         print(name)
# #     appended_data_per90 = pd.concat(appended_data_per90)
# #     appended_data_per90 = appended_data_per90.reset_index(drop=True)
# #     appended_data_per90 = appended_data_per90[['Name'] + [col for col in appended_data_per90.columns if col != 'Name']]
# #     appended_data_per90 = appended_data_per90.loc[:,~appended_data_per90.columns.duplicated()]
# #     appended_data_percentile = pd.concat(appended_data_percent)
# #     appended_data_percentile = appended_data_percentile.reset_index(drop=True)
# #     # del appended_data_percentile.columns.name
# #     appended_data_percentile = appended_data_percentile[['Name'] + [col for col in appended_data_percentile.columns if col != 'Name']]
# #     appended_data_percentile = appended_data_percentile.loc[:,~appended_data_percentile.columns.duplicated()]
# #     list_of_dfs = [appended_data_per90,appended_data_percentile]
# #     return list_of_dfs

# # def create_single_Pizza(df_in,player_name): 
# #     # parameter list
# #     params = [
# #         "Non-Penalty Goals", "npxG + xAG", "Assists",
# #         "Shot-Creating Actions", "Carries into Penalty Area",
# #         "Touches", "Progressive Passes", "Progressive Carries",
# #         "Passes into Penalty Area", "Crosses",
# #         "Interceptions", "Tackles Won",
# #         "Passes Blocked", "Ball Recoveries", "Aerials Won"
# #     ]

# #     subset_of_data = df_in.query('Player == @player_name' )
# #     scout_links = list(subset_of_data.scouting_url.unique())
# #     appended_data_percentile = generate_advanced_data(scout_links)[1]
# #     appended_data_percentile = appended_data_percentile[params]
# #     cols = appended_data_percentile.columns
# #     appended_data_percentile[cols] = appended_data_percentile[cols].apply(pd.to_numeric)
# #     params = list(appended_data_percentile.columns)
# #     # params = params[1:]


# #     values = appended_data_percentile.iloc[0].values.tolist()
# #     # values = values[1:]

# #     teams = subset_of_data['Squad'].unique()[0]


# #     style.use('fivethirtyeight')


# #     # color for the slices and text
# #     slice_colors = ["#1A78CF"] * 5 + ["#FF9300"] * 5 + ["#D70232"] * 5
# #     text_colors = ["#000000"] * 10 + ["#F2F2F2"] * 5

# #     # instantiate PyPizza class
# #     baker = PyPizza(
# #         params=params,                  # list of parameters
# #         background_color="#EBEBE9",     # background color
# #         straight_line_color="#EBEBE9",  # color for straight lines
# #         straight_line_lw=1,             # linewidth for straight lines
# #         last_circle_lw=0,               # linewidth of last circle
# #         other_circle_lw=0,              # linewidth for other circles
# #         inner_circle_size=20            # size of inner circle
# #     )

# #     # plot pizza
# #     fig, ax = baker.make_pizza(
# #         values,                          # list of values
# #         figsize=(8, 8.5),                # adjust figsize according to your need
# #         color_blank_space="same",        # use same color to fill blank space
# #         slice_colors=slice_colors,       # color for individual slices
# #         value_colors=text_colors,        # color for the value-text
# #         value_bck_colors=slice_colors,   # color for the blank spaces
# #         blank_alpha=0.4,                 # alpha for blank-space colors
# #         kwargs_slices=dict(
# #             edgecolor="#F2F2F2", zorder=2, linewidth=1
# #         ),                               # values to be used when plotting slices
# #         kwargs_params=dict(
# #             color="#000000", fontsize=9,
# #             fontproperties=font_normal.prop, va="center"
# #         ),                               # values to be used when adding parameter
# #         kwargs_values=dict(
# #             color="#000000", fontsize=11,
# #             fontproperties=font_normal.prop, zorder=3,
# #             bbox=dict(
# #                 edgecolor="#000000", facecolor="cornflowerblue",
# #                 boxstyle="round,pad=0.2", lw=1
# #             )
# #         )                                # values to be used when adding parameter-values
# #     )

# #     # add title
# #     fig.text(
# #         0.05, 0.985, f"{player_name} - {teams}", size=14,
# #         ha="left", fontproperties=font_bold.prop, color="#000000"
# #     )

# #     # add subtitle
# #     fig.text(
# #         0.05, 0.963,
# #         f"Percentile Rank vs Top-Five League {subset_of_data.position_group.unique()[0]} | Season 2024-25",
# #         size=10,
# #         ha="left", fontproperties=font_bold.prop, color="#000000"
# #     )

# #     # add credits
# #     CREDIT_1 = "@stephenaq7\ndata via FBREF / Opta"
# #     CREDIT_2 = "inspired by: @Worville, @FootballSlices"

# #     fig.text(
# #         0.99, 0.005, f"{CREDIT_1}\n{CREDIT_2}", size=9,
# #         fontproperties=font_italic.prop, color="#000000",
# #         ha="right"
# #     )

# #     # add text
# #     fig.text(
# #         0.08, 0.925, "Attacking          Possession        Distribution", size=12,
# #         fontproperties=font_bold.prop, color="#000000"
# #     )

# #     # add rectangles
# #     fig.patches.extend([
# #         plt.Rectangle(
# #             (0.05, 0.9225), 0.025, 0.021, fill=True, color="#1a78cf",
# #             transform=fig.transFigure, figure=fig
# #         ),
# #         plt.Rectangle(
# #             (0.2, 0.9225), 0.025, 0.021, fill=True, color="#ff9300",
# #             transform=fig.transFigure, figure=fig
# #         ),
# #         plt.Rectangle(
# #             (0.351, 0.9225), 0.025, 0.021, fill=True, color="#d70232",
# #             transform=fig.transFigure, figure=fig
# #         ),
# #     ])

# #     # add image
# #     ### Add Stats by Steve logo
# #     ax3 = fig.add_axes([0.80, 0.075, 0.15, 1.75])
# #     ax3.axis('off')
# #     img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
# #     ax3.imshow(img)
# #     plt.savefig(
# # 	f"/Users/stephenahiabah/Desktop/Code/cannoniq/Substack_Images/ML_DOF_DCAM/{player_name} - plot.png",
# # 	dpi = 500,
# # 	facecolor = "#EFE9E6",
# # 	bbox_inches="tight",
# #     edgecolor="none",
# # 	transparent = False
# # )
# #     plt.show()


# #     player_role_percentile_templates = {
# #         "Ball Playing CB": [
# #             # Defensive
# #             "Tkl%", "Int", "Blocks", "Passes Blocked", "Clr",
# #             # Possession
# #             "Total - Cmp%", "Short - Cmp%", "Long - Cmp%", "PrgP", "Carries - PrgDist",
# #             # Attacking
# #             "xA", "Ast", "KP", "Sh/90", "G/Sh"
# #         ],
# #         "Classic CB": [
# #             # Defensive
# #             "Tkl%", "Int", "Clr", "Recov", "Shots Blocked",
# #             # Possession
# #             "Touches", "Total - Att", "Total - Cmp%", "Long - Cmp%", "Rec",
# #             # Attacking
# #             "xG", "Sh/90", "Ast", "SoT%", "KP"
# #         ],
# #         "Classic Fullback": [
# #             # Defensive
# #             "Tkl%", "Int", "Blocks", "Tkl+Int", "Passes Blocked",
# #             # Possession
# #             "Touches", "Carries", "Carries - TotDist", "PrgR", "Short - Cmp%",
# #             # Attacking
# #             "xA", "Ast", "KP", "CrsPA", "1/3"
# #         ],
# #         "Inverted Fullback": [
# #             # Defensive
# #             "Tkl%", "Int", "Blocks", "Pressures", "Dribblers- Tkl",
# #             # Possession
# #             "Total - Cmp%", "Short - Cmp%", "Carries - PrgDist", "PrgP", "Touches",
# #             # Attacking
# #             "xA", "KP", "Ast", "Sh/90", "SCA90"
# #         ],
# #         "Attacking Fullback": [
# #             # Defensive
# #             "Tkl%", "Tkl+Int", "Blocks", "Int", "Dribblers- Tkl",
# #             # Possession
# #             "Carries - PrgC", "Carries - PrgDist", "Touches", "Take Ons - Attempted", "PrgR",
# #             # Attacking
# #             "xA", "KP", "CrsPA", "Ast", "SCA - PassLive"
# #         ],
# #         "Destroyer DM": [
# #             # Defensive
# #             "Tkl+Int", "Int", "Passes Blocked", "Pressures", "Blocks",
# #             # Possession
# #             "Touches", "Total - Cmp%", "Carries - PrgDist", "Short - Cmp%", "Rec",
# #             # Attacking
# #             "xA", "Long - Cmp%", "Ast", "Sh/90", "KP"
# #         ],
# #         "Deep Lying Playmaker CM": [
# #             # Defensive
# #             "Tkl%", "Passes Blocked", "Tkl+Int", "Shots Blocked", "Int",
# #             # Possession
# #             "Total - Cmp%", "Short - Cmp%", "PrgP", "Carries - PrgDist", "Touches",
# #             # Attacking
# #             "xA", "Long - Cmp%", "Ast", "Sh/90", "KP"
# #         ],
# #         "Box to Box CM": [
# #             # Defensive
# #             "Tkl%", "Tkl+Int", "Int", "Blocks", "Passes Blocked",
# #             # Possession
# #             "Carries - PrgDist", "Carries - PrgC", "Touches", "Short - Cmp%", "Rec",
# #             # Attacking
# #             "xG", "xA", "KP", "Ast", "Sh/90"
# #         ],
# #         "Playmaker CM": [
# #             # Defensive
# #             "Tkl%", "Int", "Passes Blocked", "Blocks", "Tkl+Int",
# #             # Possession
# #             "Touches", "Carries - PrgDist", "Short - Cmp%", "Total - Cmp%", "PrgP",
# #             # Attacking
# #             "xA", "Ast", "KP", "SCA90", "GCA90"
# #         ],
# #         "Classic AM": [
# #             # Defensive
# #             "Tkl%", "Tkl+Int", "Pressures", "Int", "Blocks",
# #             # Possession
# #             "Touches", "Carries - PrgC", "Short - Cmp%", "Total - Cmp%", "Rec",
# #             # Attacking
# #             "xA", "Ast", "KP", "Sh/90", "xG"
# #         ],
# #         "Inside Forward": [
# #             # Defensive
# #             "Tkl%", "Tkl+Int", "Int", "Blocks", "Dribblers- Tkl",
# #             # Possession
# #             "Carries - PrgC", "Carries - PrgDist", "Touches", "Take Ons - Attempted", "PrgR",
# #             # Attacking
# #             "xG", "xA", "KP", "Ast", "Sh/90"
# #         ],
# #         "Winger": [
# #             # Defensive
# #             "Tkl%", "Tkl+Int", "Blocks", "Int", "Dribblers- Tkl",
# #             # Possession
# #             "Carries - PrgDist", "Carries - PrgC", "Touches", "Take Ons - Attempted", "1/3",
# #             # Attacking
# #             "xA", "KP", "Ast", "CrsPA", "Sh/90"
# #         ],
# #         "Center Forward": [
# #             # Defensive
# #             "Tkl%", "Int", "Tkl+Int", "Blocks", "Pressures",
# #             # Possession
# #             "Touches", "Rec", "Carries - PrgC", "Short - Cmp%", "Fld",
# #             # Attacking
# #             "xG", "xA", "KP", "Ast", "Sh/90"
# #         ],
# #         "False 9": [
# #             # Defensive
# #             "Tkl%", "Tkl+Int", "Int", "Blocks", "Pressures",
# #             # Possession
# #             "Touches", "Total - Cmp%", "Short - Cmp%", "Carries - PrgC", "Rec",
# #             # Attacking
# #             "xG", "xA", "KP", "Ast", "SCA - PassLive"
# #         ]
# #     }

# #     clean_stat_mapping = {
# #         # Defensive
# #         "Tkl%": "Tackle Success %",
# #         "Int": "Interceptions",
# #         "Blocks": "Blocks",
# #         "Passes Blocked": "Passes Blocked",
# #         "Clr": "Clearances",
# #         "Recov": "Recoveries",
# #         "Shots Blocked": "Shots Blocked",
# #         "Tkl+Int": "Tackles + Interceptions",
# #         "Dribblers- Tkl": "Tackles vs Dribblers",
# #         "Pressures": "Pressures",
# #         "CrdY": "Yellow Cards",
# #         "Fls": "Fouls Committed",

# #         # Possession
# #         "Touches": "Touches",
# #         "Carries": "Carries",
# #         "Carries - TotDist": "Carry Distance (Total)",
# #         "Carries - PrgDist": "Progressive Carry Distance",
# #         "Carries - PrgC": "Progressive Carries",
# #         "Carries - 1/3": "Carries into Final Third",
# #         "Carries - CPA": "Carries into Penalty Area",
# #         "Take Ons - Attempted": "Dribbles Attempted",
# #         "Rec": "Passes Received",
# #         "Total - Cmp%": "Shot Stopping % (Total)",
# #         "Total - Att": "Passes Attempted (Total)",
# #         "Short - Cmp%": "Shot Stopping % (Short)",
# #         "Long - Cmp%": "Shot Stopping % (Long)",
# #         "PrgR": "Progressive Receptions",
# #         "PrgP": "Progressive Passes",
# #         "Fld": "Fouls Drawn",

# #         # Attacking
# #         "xA": "Expected Assists (xA)",
# #         "Ast": "Assists",
# #         "KP": "Key Passes",
# #         "Sh/90": "Shots per 90",
# #         "xG": "Expected Goals (xG)",
# #         "CrsPA": "Crosses into Penalty Area",
# #         "SCA90": "Shot Creating Actions /90",
# #         "GCA90": "Goal Creating Actions /90",
# #         "1/3": "Passes into Final Third",
# #         "PPA": "Passes into Penalty Area",
# #         "SoT%": "Shots on Target %",
# #         "G/Sh": "Goals per Shot",
# #         "npxG": "Non-Penalty xG",
# #         "Long - Cmp%": "Shot Stopping % (Long)",
# #         "SCA - PassLive": "SCA – Live Pass",
# #         "GCA - PassLive": "GCA – Live Pass",
# #         "SCA - PassDead": "SCA – Dead Ball",
# #         "GCA - PassDead": "GCA – Dead Ball"
# #     }




# # import matplotlib.ticker as ticker
# # import numpy as np
# # from sklearn.linear_model import LinearRegression


# # fig = plt.figure(figsize=(12, 8), dpi=200)
# # ax = plt.subplot(111)
# # style.use('fivethirtyeight')

# # ax.plot(X, conc, label="Rolling Goals Agains")

# # ax.set_xlabel("Match index", fontsize=12)
# # ax.set_ylabel("Goals Against", fontsize=12)

# # # Set y-axis tick labels as percentages
# # # ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

# # ax.legend()

# # ax.grid(True, linestyle='dotted')
# # main_color = '#0057B8'
# # # Adding vertical dotted lines and labels
# # plt.axvline(x=10, color='grey', linestyle='dotted')
# # plt.axvline(x=17, color='grey', linestyle='dotted')
# # plt.axvline(x=29, color='grey', linestyle='dotted')
# # plt.axvline(x=35, color='grey', linestyle='dotted')


# # fig_text(
# #     x=0.15, y=1.12, s='<Arsenal> Goal Conceded Stats - EPL', family='DM Sans',
# #     ha='left', va='center', weight='normal', size='large',
# #     highlight_textprops=[{'weight':'bold', 'size':'x-large'}]
# # )
# # fig_text(
# #     x=0.15, y=1.07, s='<5-game moving average> Goals Conceded in the 2022/2023 Premier League.\n Viz by @stephenaq7.',
# #     family='Karla',
# #     ha='left', va='center', size='small',
# #     highlight_textprops=[{'weight':'bold', 'color':main_color}]
# # )
# # ax.annotate(
# #     xy = (10, .50),
# #     xytext = (20, 10),
# #     textcoords = "offset points",
# #     text = "Zinchenko Injury #1",
# #     size = 10,
# #     color = "grey",
# #     arrowprops=dict(
# #         arrowstyle="->", shrinkA=0, shrinkB=5, color="grey", linewidth=0.75,
# #         connectionstyle="angle3,angleA=50,angleB=-30"
# #     ) # Arrow to connect annotation
# # )

# # ax.annotate(
# #     xy = (17, .55),
# #     xytext = (20, 10),
# #     textcoords = "offset points",
# #     text = "G.Jesus Injury",
# #     size = 10,
# #     color = "grey",
# #     arrowprops=dict(
# #         arrowstyle="->", shrinkA=0, shrinkB=5, color="grey", linewidth=0.75,
# #         connectionstyle="angle3,angleA=50,angleB=-30"
# #     ) # Arrow to connect annotation
# # )


# # ax.annotate(
# #     xy = (29, .55),
# #     xytext = (20, 10),
# #     textcoords = "offset points",
# #     text = "W.Saliba Injury",
# #     size = 10,
# #     color = "grey",
# #     arrowprops=dict(
# #         arrowstyle="->", shrinkA=0, shrinkB=5, color="grey", linewidth=0.75,
# #         connectionstyle="angle3,angleA=50,angleB=-30"
# #     ) # Arrow to connect annotation
# # )
# # ax_size = 0.15
# # image_ax = fig.add_axes(
# #     [0.01, 1.0, ax_size, ax_size],
# #     fc='None'
# # )


# # ax.annotate(
# #     xy = (35, .50),
# #     xytext = (20, 10),
# #     textcoords = "offset points",
# #     text = "Zinchenko Injury #2",
# #     size = 10,
# #     color = "grey",
# #     arrowprops=dict(
# #         arrowstyle="->", shrinkA=0, shrinkB=5, color="grey", linewidth=0.75,
# #         connectionstyle="angle3,angleA=50,angleB=-30"
# #     ) # Arrow to connect annotation
# # )
# # fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
# # club_icon = Image.open(urllib.request.urlopen(f"{fotmob_url}9825.png"))
# # image_ax.imshow(club_icon)
# # image_ax.axis('off')

# # ax3 = fig.add_axes([0.85, 0.22, 0.11, 1.75])
# # ax3.axis('off')
# # img = image.imread('/Users/stephenahiabah/Desktop/GitHub/Webs-scarping-for-Fooball-Data-/outputs/logo_transparent_background.png')
# # ax3.imshow(img)

# # plt.tight_layout()  # Adjust spacing between subplots

# # plt.show()


# # fig = plt.figure(dpi=300)
# # ax = plt.subplot()
# # plt.rcParams['hatch.linewidth'] = 0.2

# # bars_ = ax.bar(profit_df.index, profit_df['share'], hatch='////////', ec=ax.get_facecolor(), width=1)
# # for index, b in enumerate(bars_):
# #     if profit_df['bin'].iloc[index].left < 0:
# #         color = '#495371'
# #     else:
# #         color = '#287271'
# #     b.set_facecolor(color)
# #     ax.annotate(
# #         xy=(index, b.get_height()), text=f'{b.get_height():.1%}',
# #         xytext=(0,8), textcoords='offset points', 
# #         ha='center', va='center',
# #         color=color, size=9
# #     )

# # ticks = ax.set_xticks(
# #     ticks=profit_df.index,
# #     labels=profit_df['bin'],
# #     rotation=90
# # )
# # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0%}'))
# # plt.savefig(
# # 	"figures/dice_game_hist.png",
# # 	dpi = 600,
# # 	facecolor = "none",
# # 	bbox_inches="tight",
# #     edgecolor="none",
# # 	transparent = True
# # )


# # import matplotlib.ticker as ticker
# # import matplotlib.patheffects as path_effects

# # def plot_point_difference(ax, player, label_y=False, data=timeseries):
# #     ax.grid(ls='--', color='lightgrey')
# #     for spine in ax.spines.values():
# #         spine.set_edgecolor('lightgrey')
# #         spine.set_linestyle('dashed')
# #         spine.set_visible(False)
# #     ax.tick_params(color='lightgrey', labelsize=6, labelcolor='grey')
    
# #     # Get cumulative data for the player
# #     test_df = get_cumulative(player, data)

# #     # Plot cumulative goals and xG
# #     ax.plot(test_df.index + 1, test_df['cum_goals'], marker='o', mfc='white', ms=1, linewidth=1, color='#287271')
# #     ax.plot(test_df.index + 1, test_df['cum_xgoals'], marker='o', mfc='white', ms=1, linewidth=1, color='#D81159')
    
# #     # Set y-axis ticks and format
# #     ax.yaxis.set_major_locator(ticker.MultipleLocator(3))
# #     ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}'))
    
# #     # Remove fixed y-axis limits to allow dynamic scaling
# #     # ax.set_ylim(-1, 30)  # Remove this line
    
# #     # Set x-axis limits and remove x-axis labels
# #     ax.set_xlim(0, 30)
# #     ax.set_xticklabels([])
    
# #     # Add difference annotation if data is available
# #     if len(test_df) > 0:
# #         difference = test_df['cum_goals'].iloc[-1] - test_df['cum_xgoals'].iloc[-1]
# #         mid_point = test_df['cum_goals'].iloc[-1] + (test_df['cum_xgoals'].iloc[-1] - test_df['cum_goals'].iloc[-1]) / 2

# #         text_ = ax.annotate(
# #             xy=(ax.get_xlim()[1], mid_point),
# #             text=f'{difference:.1f}',
# #             xytext=(-5, 0),
# #             ha='center',
# #             va='center',
# #             color='#D81159',
# #             weight='bold',
# #             size=7,
# #             textcoords='offset points'
# #         )
# #         text_.set_path_effects(
# #             [path_effects.Stroke(linewidth=1.5, foreground='white'), path_effects.Normal()]
# #         )
    
# #     # Set y-axis label if required
# #     if label_y:
# #         ax.set_ylabel('Goals', color='grey', size=8)
# #     else:
# #         ax.set_yticklabels([])

#     def generate_gk_similarity_score_card(self, player_name, mertrics_similarity, Cannoniq_DB):
#         mertrics_similarity = mertrics_similarity.rename(columns={'Squad': 'team'})
#         fm_ids = pd.read_csv("/Users/stephenahiabah/Desktop/Code/cannoniq/CSVs/Top6_leagues_fotmob_ids.csv")
#         fm_ids = fm_ids[["team", "team_id"]]

#         mertrics_similarity = mertrics_similarity.merge(fm_ids, on='team', how='left')
#         mertrics_similarity = mertrics_similarity.dropna(subset=['team_id'])
#         mertrics_similarity['team_id'] = mertrics_similarity['team_id'].astype(float).astype(int)

#         mertrics_similarity[['perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']] = \
#             mertrics_similarity[['perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']].round(2)

#         df_final = mertrics_similarity[['Player', 'Pos','team_id','perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']]
#         metric_scores =['Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score']

#         sim_player_vals = Cannoniq_DB[Cannoniq_DB['Player'] == player_name][metric_scores].values.tolist()
#         sim_player_vals = [val for sublist in sim_player_vals for val in sublist]

#         df_final['Δ% Shot Stopping'] = ((df_final['Shot_Stopping_Score'] - sim_player_vals[0]) / sim_player_vals[0]).round(1) * 100
#         df_final['Δ% Distribution'] = ((df_final['Distribution_Score'] - sim_player_vals[1]) / sim_player_vals[1]).round(1) * 100
#         df_final['Δ% Claiming'] = ((df_final['Claiming_Score'] - sim_player_vals[2]) / sim_player_vals[2]).round(1) * 100
#         df_final['Δ% Sweeping'] = ((df_final['Sweeping_Score'] - sim_player_vals[3]) / sim_player_vals[3]).round(1) * 100
#         df_final = df_final[::-1]

#         def perc_battery(perc_similarity, ax):
#             '''
#             This function takes an integer and an axes and 
#             plots a battery chart.
#             '''
#             ax.set_xlim(0,1)
#             ax.set_ylim(0,1)
#             ax.barh([0.5], [1], fc = 'white', ec='black', height=.35)
#             ax.barh([0.5], [perc_similarity/100], fc = '#00529F', height=.35)
#             text_ = ax.annotate(
#                 xy=((perc_similarity/100), .5),
#                 text=f'{(perc_similarity/100):.0%}',
#                 xytext=(-8,0),
#                 textcoords='offset points',
#                 weight='bold',
#                 color='#EFE9E6',
#                 va='center',
#                 ha='center',
#                 size=8
#             )
#             ax.set_axis_off()
#             return ax

#         def ax_logo(team_id, ax):
#             '''
#             Plots the logo of the team at a specific axes.
#             Args:
#                 team_id (int): the id of the team according to Fotmob. You can find it in the url of the team page.
#                 ax (object): the matplotlib axes where we'll draw the image.
#             '''
#             fotmob_url = 'https://images.fotmob.com/image_resources/logo/teamlogo/'
#             club_icon = Image.open(urllib.request.urlopen(f'{fotmob_url}{team_id}.png'))
#             ax.imshow(club_icon)
#             ax.axis('off')
#             return ax

#         fig = plt.figure(figsize=(17,17), dpi=400)
#         ax = plt.subplot()

#         ncols = 12
#         nrows = df_final.shape[0]

#         ax.set_xlim(0, ncols + 1)
#         ax.set_ylim(0, nrows + 1)

#         positions = [0.25, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5]
#         columns = ['Player', 'Pos', 'perc_similarity', 'Shot_Stopping_Score', 'Distribution_Score', 'Claiming_Score', 'Sweeping_Score', 'Δ% Shot Stopping', 'Δ% Distribution', 'Δ% Claiming', 'Δ% Sweeping']

#         # -- Add table's main text
#         for i in range(nrows):
#             for j, column in enumerate(columns):
#                 if j == 0:
#                     ha = 'left'
#                 else:
#                     ha = 'center'
#                 if column == 'perc_similarity':
#                     continue
#                 else:
#                     text_label = f'{df_final[column].iloc[i]}'
#                     weight = 'normal'
#                 ax.annotate(
#                     xy=(positions[j], i + .5),
#                     text=text_label,
#                     ha=ha,
#                     va='center',
#                     size = 10,
#                     weight=weight
#                 )

#         # -- Transformation functions
#         DC_to_FC = ax.transData.transform
#         FC_to_NFC = fig.transFigure.inverted().transform
#         # -- Take data coordinates and transform them to normalized figure coordinates
#         DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))
#         # -- Add nation axes
#         ax_point_1 = DC_to_NFC([2.25, 0.25])
#         ax_point_2 = DC_to_NFC([2.75, 0.75])
#         ax_width = abs(ax_point_1[0] - ax_point_2[0])
#         ax_height = abs(ax_point_1[1] - ax_point_2[1])
#         for x in range(0, nrows):
#             ax_coords = DC_to_NFC([2.25, x + .25])
#             flag_ax = fig.add_axes(
#                 [ax_coords[0], ax_coords[1], ax_width, ax_height]
#             )
#             ax_logo(df_final['team_id'].iloc[x], flag_ax)

#         ax_point_1 = DC_to_NFC([4, 0.05])
#         ax_point_2 = DC_to_NFC([5, 0.95])
#         ax_width = abs(ax_point_1[0] - ax_point_2[0])
#         ax_height = abs(ax_point_1[1] - ax_point_2[1])
#         for x in range(0, nrows):
#             ax_coords = DC_to_NFC([4, x + .025])
#             bar_ax = fig.add_axes(
#                 [ax_coords[0], ax_coords[1], ax_width, ax_height]
#             )
#             perc_battery(df_final['perc_similarity'].iloc[x], bar_ax)

#         # -- Add column names
#         column_names = ['Player', 'Position', 'Percent\nSimilarity','Shot Stopping', 'Distribution', 'Claiming', 'Sweeping','Δ%\nShot Stopping','Δ%\nDistribution','Δ%\nClaiming','Δ%\nSweeping']
#         for index, c in enumerate(column_names):
#                 if index == 0:
#                     ha = 'left'
#                 else:
#                     ha = 'center'
#                 ax.annotate(
#                     xy=(positions[index], nrows + .25),
#                     text=column_names[index],
#                     ha=ha,
#                     va='bottom',
#                     size = 12,
#                     weight='book'
#                 )

#         # Add dividing lines
#         ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows], lw=1.5, color='black', marker='', zorder=4)
#         ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1.5, color='black', marker='', zorder=4)
#         for x in range(1, nrows):
#             ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=1.15, color='gray', ls=':', zorder=3 , marker='')

#         ax.fill_between(
#             x=[0,2],
#             y1=nrows,
#             y2=0,
#             color='lightgrey',
#             alpha=0.5,
#             ec='None'
#         )

#         # Custom colormap and normalization
#         cmap = mcolors.LinearSegmentedColormap.from_list('red_green', ['red', 'white', 'green'])
#         norm = mcolors.Normalize(vmin=-50, vmax=50)

#         # Example of Δ% columns to be visualized
#         delta_columns = ['Δ% Shot Stopping', 'Δ% Distribution', 'Δ% Claiming', 'Δ% Sweeping']

#         # Loop through delta columns and fill between with corresponding color
#         for idx, col in enumerate(delta_columns):
#             for i in range(nrows):
#                 value = df_final[col].iloc[i]
#                 ax.fill_between(
#                     x=[9 + idx, 10 + idx],
#                     y1=i + 1,
#                     y2=i,
#                     color=cmap(norm(value)),
#                     alpha=0.6,
#                     ec='None'
#                 )

#         ax.set_axis_off()

#         # -- Final details
#         league_icon = Image.open("/Users/stephenahiabah/Desktop/Code/cannoniq/Images//premier-league-2-logo.png")
#         league_ax = fig.add_axes([0.06, 0.88, 0.10, 0.10], zorder=1)
#         league_ax.imshow(league_icon)
#         league_ax.axis("off")

#         ax.tick_params(axis='both', which='major', labelsize=8)

#         fig_text(
#             x = 0.5, y = 0.95, 
#             s = f'{player_name} - PIQ Similarity Model Results 24/25',
#             va = "bottom", ha = "center",
#             fontsize = 17, color = "black", weight = "bold"
#         )

#         fig_text(
#             x = 0.5, y = 0.92, 
#             s = f'Shot Stopping, Distribution, Claiming & Sweeping scores generated via weighted aggregated metrics from FBREF (Out of 10)',
#             va = "bottom", ha = "center",
#             fontsize = 12, color = "black", weight = "book"
#         )
#         fig_text(
#             x = 0.5, y = 0.90, 
#             s = f'Δ% columns are PIQ score percentage change, -ve% means {player_name} has a better score and vice versa',
#             va = "bottom", ha = "center",
#             fontsize = 12, color = "black", weight = "book"
#         )

#         ### Add Stats by Steve logo
#         ax3 = fig.add_axes([0.83, 0.13, 0.09, 1.60])
#         ax3.axis('off')
#         img = plt.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
#         ax3.imshow(img)
  
#         output_path = f"GK_profiles/{player_name}/scorecards/"
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         save_path = os.path.join(output_path, f"{player_name}_similarity_score_card.png")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')  
        
#         plt.show()

#         return df_final