# cannoniq_charts.py

import os
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as image
from PIL import Image
from bs4 import BeautifulSoup
from mplsoccer import PyPizza, FontManager
from highlight_text import fig_text
from adjustText import adjust_text



class CannoniqCharts:
    def __init__(self, fbref_module_path, phonebook_path, logo_path):
        import sys
        if fbref_module_path not in sys.path:
            sys.path.append(fbref_module_path)
        import FBREF_Aggregations as fbref

        self.fbref = fbref
        self.db_builder = fbref.CreateFBRefDatabase()
        self.db = self._build_master_db(phonebook_path)
        self.font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Regular.ttf')
        self.font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/src/hinted/Roboto-Italic.ttf')
        self.font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/RobotoSlab[wght].ttf')
        self.logo_path = logo_path

    def _build_master_db(self, phonebook_path):
        db = self.db_builder.generate_pitch_iq_scores()
        db['player_name_match'] = db['Player'].apply(lambda x: self.db_builder.remove_accents(x))
        phonebook = pd.read_csv(phonebook_path).rename(columns={'Player': 'player_name_match'})
        phonebook = phonebook.drop_duplicates(subset='player_name_match', keep='first')
        db = pd.merge(db, phonebook[['player_name_match', 'scouting_url', 'match_logs']], on='player_name_match', how='left')
        time.sleep(100)
        full_stats = self.db_builder.create_full_stats_db()
        full_stats = self.db_builder.per_90fi(full_stats)

        db = db.drop_duplicates(subset=['Player'], keep='first')
        db = pd.merge(db, full_stats, on=['Player', 'Nation', 'Pos', 'Squad'], how='inner')
        db.drop(columns=[col for col in db.columns if col.endswith('_y')], inplace=True)
        db.columns = [col.replace('_x', '') for col in db.columns]
        return db

    def generate_advanced_data(self, scout_links):
        appended_data_per90 = []
        appended_data_percent = []
        for url in scout_links:
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            name = [element.text for element in soup.find_all("span")][7]
            df = pd.read_html(page.text)
            df[-1].columns = df[-1].columns.droplevel(0)
            stats = df[-1].loc[(df[-1]['Statistic'] != "Statistic") & (df[-1]['Statistic'] != ' ')]
            stats = stats.dropna(subset=['Statistic', "Per 90", "Percentile"])

            per_90_df = stats[['Statistic', "Per 90"]].set_index("Statistic").T
            per_90_df["Name"] = name

            percentile_df = stats[['Statistic', "Percentile"]].set_index("Statistic").T
            percentile_df["Name"] = name

            appended_data_per90.append(per_90_df)
            appended_data_percent.append(percentile_df)

            time.sleep(10)

        per90 = pd.concat(appended_data_per90).reset_index(drop=True)
        per90 = per90[['Name'] + [col for col in per90.columns if col != 'Name']].loc[:, ~per90.columns.duplicated()]

        percentiles = pd.concat(appended_data_percent).reset_index(drop=True)
        percentiles = percentiles[['Name'] + [col for col in percentiles.columns if col != 'Name']].loc[:, ~percentiles.columns.duplicated()]

        return [per90, percentiles]

    def create_single_pizza(self, player_name, param_list):
        player_row = self.db[self.db['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"Player '{player_name}' not found in database")

        scout_links = player_row['scouting_url'].dropna().unique().tolist()
        data = self.generate_advanced_data(scout_links)[1]  # percentiles
        data = data[param_list]
        data[param_list] = data[param_list].apply(pd.to_numeric)
        values = data.iloc[0].values.tolist()

        slice_colors = ["#1A78CF"] * 5 + ["#FF9300"] * 5 + ["#D70232"] * 5
        text_colors = ["#000000"] * 10 + ["#F2F2F2"] * 5

        baker = PyPizza(
            params=param_list,
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
            kwargs_params=dict(color="#000000", fontsize=9, va="center", fontproperties=self.font_normal.prop),
            kwargs_values=dict(
                color="#000000", fontsize=11, fontproperties=self.font_normal.prop, zorder=3,
                bbox=dict(edgecolor="#000000", facecolor="cornflowerblue", boxstyle="round,pad=0.2", lw=1)
            )
        )

        club = player_row['Squad'].values[0]
        age = float(player_row['Age'].values[0])
        pos_group = player_row['position_group'].values[0]

        fig.text(0.05, 0.985, f"{player_name} - {club}", size=14, ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.05, 0.963, f"Percentile Rank vs Top-Five League {pos_group} | Season 2024-25", size=10, ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.08, 0.925, "Attacking          Possession        Defending", size=12, fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.99, 0.005, "@stephenaq7\ndata via FBREF / Opta\ninspired by: @Worville, @FootballSlices", size=9, fontproperties=self.font_italic.prop, color="#000000", ha="right")

        fig.patches.extend([
            plt.Rectangle((0.05, 0.9225), 0.025, 0.021, fill=True, color="#1a78cf", transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.2, 0.9225), 0.025, 0.021, fill=True, color="#ff9300", transform=fig.transFigure, figure=fig),
            plt.Rectangle((0.351, 0.9225), 0.025, 0.021, fill=True, color="#d70232", transform=fig.transFigure, figure=fig),
        ])

        try:
            ax3 = fig.add_axes([0.80, 0.075, 0.15, 1.75])
            ax3.axis('off')
            img = image.imread(self.logo_path)
            ax3.imshow(img)
        except FileNotFoundError:
            print("Logo image not found.")

        out_path = f"Player_profiles/{player_name}/{player_name}_single_pizza.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=500, facecolor="#EFE9E6", bbox_inches="tight")
        plt.close()
        return out_path
    
    

    def plot_role_based_kde(self, player_name, role, role_templates):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        if role not in role_templates:
            raise ValueError(f"Invalid role '{role}'.")

        params = role_templates[role]
        player_row = self.db[self.db['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        pos_group = player_row['position_group'].values[0]
        filtered_df = self.db[(self.db['position_group'] == pos_group) & (self.db['90s'] > 10)]

        num_params = len(params)
        fig, axs = plt.subplots(num_params, 1, figsize=(7, num_params * 1.2))
        plt.subplots_adjust(hspace=0.8)

        for i, param in enumerate(params):
            ax = axs[i]
            data = pd.to_numeric(filtered_df[param], errors='coerce').dropna()
            player_val = float(player_row[param].values[0])
            sns.kdeplot(data, ax=ax, fill=True, bw_adjust=0.5)
            ax.axvline(player_val, color="crimson", linestyle="--", lw=2)
            ax.set_title(self.clean_stat_mapping.get(param, param), loc='left', fontsize=10)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines[['top', 'right', 'left']].set_visible(False)

        fig.suptitle(f"{player_name} vs {pos_group} ({role})", fontsize=14, fontweight='bold')
        fig.text(0.99, 0.01, "@stephenaq7 | Cannoniq\nFBref/Opta data", ha="right", fontsize=8)
        out_path = f"Player_profiles/{player_name}/{player_name}_kde.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#EFE9E6")
        plt.close()
        return out_path
    
    def plot_role_based_comparison(self, player_name, role, role_templates):
        if role not in role_templates:
            raise ValueError(f"Invalid role '{role}'.")

        params = role_templates[role]
        player_row = self.db[self.db['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        pos_group = player_row['position_group'].values[0]
        filtered_df = self.db[(self.db['position_group'] == pos_group) & (self.db['90s'] > 10)]

        comparisons = []
        for param in params:
            series = pd.to_numeric(filtered_df[param], errors='coerce')
            player_val = float(player_row[param].values[0])
            percentile = round((series < player_val).mean() * 100)
            comparisons.append((self.clean_stat_mapping.get(param, param), percentile))

        df_plot = pd.DataFrame(comparisons, columns=['Stat', 'Percentile']).sort_values("Percentile")
        cmap = plt.get_cmap("RdYlGn")
        colors = [cmap(x / 100) for x in df_plot["Percentile"]]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(df_plot["Stat"], df_plot["Percentile"], color=colors)
        ax.axvline(50, color="gray", linestyle="--", lw=1)
        ax.set_xlim(0, 100)
        ax.set_title(f"{player_name} vs {pos_group} - {role}", fontsize=14, weight="bold")
        ax.set_xlabel("Percentile Rank")
        fig.text(0.99, 0.01, "@stephenaq7 | Cannoniq\nFBref/Opta", ha="right", fontsize=8)
        out_path = f"Player_profiles/{player_name}/{player_name}_bar.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#EFE9E6")
        plt.close()
        return out_path

    def create_percentile_Pizza(self, player_name, role, role_templates):
        from matplotlib.patches import Rectangle

        if role not in role_templates:
            raise ValueError(f"Invalid role '{role}'.")

        params = role_templates[role]
        readable_params = [self.clean_stat_mapping.get(p, p) for p in params]
        player_row = self.db[self.db['Player'] == player_name]
        if player_row.empty:
            raise ValueError(f"{player_name} not found in dataset.")

        pos_group = player_row['position_group'].values[0]
        filtered_df = self.db[(self.db['position_group'] == pos_group) & (self.db['90s'] > 10)]

        percentiles = []
        for param in params:
            data = pd.to_numeric(filtered_df[param], errors='coerce').dropna().astype(np.float64)
            player_val = float(player_row[param].values[0])
            percentile = round((data < player_val).mean() * 100)
            percentiles.append(percentile)

        # Color theming by role
        if "CB" in role:
            slice_colors = ["#8B0000"] * 5 + ["#B22222"] * 5 + ["#DC143C"] * 5
        elif "CM" in role or "DM" in role:
            slice_colors = ["#097969"] * 5 + ["#AFE1AF"] * 5 + ["#088F8F"] * 5
        elif "AM" in role or "Winger" in role:
            slice_colors = ["#00008B"] * 5 + ["#4169E1"] * 5 + ["#87CEFA"] * 5
        elif "F9" in role or "Forward" in role:
            slice_colors = ["#A9A9A9"] * 5 + ["#C0C0C0"] * 5 + ["#D3D3D3"] * 5
        else:
            slice_colors = ["#D70232"] * 5 + ["#FF9300"] * 5 + ["#1A78CF"] * 5

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
            value_colors=["#000000"] * 15,
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
        fig.text(0.05, 0.985, f"{player_name} - {club} - {role} Template", size=14, ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.05, 0.963, f"Percentile Rank vs Top-Five League {pos_group} | Season 2024-25", size=10, ha="left", fontproperties=self.font_bold.prop, color="#000000")
        fig.text(0.08, 0.925, "Attacking          Possession        Defending", size=12, fontproperties=self.font_bold.prop, color="#000000")

        # Legend
        fig.patches.extend([
            Rectangle((0.05, 0.9225), 0.025, 0.021, fill=True, color=slice_colors[0], transform=fig.transFigure, figure=fig),
            Rectangle((0.2, 0.9225), 0.025, 0.021, fill=True, color=slice_colors[5], transform=fig.transFigure, figure=fig),
            Rectangle((0.351, 0.9225), 0.025, 0.021, fill=True, color=slice_colors[10], transform=fig.transFigure, figure=fig),
        ])

        fig.text(0.99, 0.005, "@cannoniq.bsky.com\ndata via FBREF / Opta\ninspired by: @Worville, @FootballSlices", size=9, fontproperties=self.font_italic.prop, color="#000000", ha="right")

        try:
            ax3 = fig.add_axes([0.80, 0.075, 0.15, 1.75])
            ax3.axis('off')
            img = image.imread(self.logo_path)
            ax3.imshow(img)
        except FileNotFoundError:
            print("Logo image not found.")

        out_path = f"Player_profiles/{player_name}/{player_name}_percentile_pizza.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=500, facecolor="#EBEBE9", bbox_inches="tight")
        plt.close()
        return out_path

    def combine_player_profile_plots(self, player_name):
        import matplotlib.image as mpimg

        plot_types = ["_single_pizza", "_percentile_pizza", "_kde", "_bar"]
        paths = [f"Player_profiles/{player_name}/{player_name}{pt}.png" for pt in plot_types]
        images = [mpimg.imread(p) for p in paths if os.path.exists(p)]

        if not images:
            raise FileNotFoundError("No images found to combine.")

        widths, heights = zip(*(img.shape[1::-1] for img in images))
        total_width = sum(widths)
        max_height = max(heights)

        combined = np.ones((max_height, total_width, 3), dtype=np.uint8) * 255
        current_x = 0
        for img in images:
            h, w, _ = img.shape
            combined[:h, current_x:current_x+w] = img[:, :, :3]
            current_x += w

        out_path = f"Player_profiles/{player_name}/{player_name}_profile_panel.png"
        plt.imsave(out_path, combined)
        return out_path
    
    
    def run_full_pipeline(self, player_name, role, role_templates):
        self.create_single_pizza(player_name, role_templates[role])
        self.create_percentile_Pizza(player_name, role, role_templates)
        self.plot_role_based_kde(player_name, role, role_templates)
        self.plot_role_based_comparison(player_name, role, role_templates)
        return self.combine_player_profile_plots(player_name)

    # Define the role-based templates
    player_role_templates = {
        "ball_playing_cb": [
            'Touches', 'Def 3rd', 'Mid 3rd', 'Total - Cmp%',
            'Short - Cmp%', 'Medium - Cmp%', 'Long - Cmp%',  'PrgDist',
            'PrgP', 'Int', 'TklW', 'Total Blocks',
        ],
        "classic_cb": [
            'Touches', 'Def 3rd', 'Clr', 'Won', 'Lost', 'Won%', 'Int', 'TklW',
            'Total - Tkl', 'Tackles - Def 3rd', 'Dribblers- Tkl', 'Total Blocks',
            'Shots Blocked', 'Passes Blocked', 'Err', 'OG'
        ],
        "classic_fullback": [
            'Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Crs', 'Int', 'TklW',
            'Total - Tkl', 'Tackles - Mid 3rd', 'Carries', 'Carries - PrgDist',
            'Carries - PrgC', 'Carries - 1/3', 'Blocks', 'Fld', 'Off'
        ],
        "defensive_fullback": [
            'Touches', 'Def 3rd', 'Mid 3rd', 'Int', 'TklW', 'Total - Tkl',
            'Tackles - Def 3rd', 'Tkl+Int', 'Blocks', 'Fld', 'Clr', 'Err',
            'Carries - TotDist', 'Passes Blocked'
        ],
        "inverted_fullback": [
            'Touches', 'Mid 3rd', 'Carries - PrgDist', 'Carries - PrgC',
            'Carries - 1/3', 'Total - Cmp%', 'Medium - Cmp%', 'Short - Cmp%',
            'PrgP', 'PPA', 'KP', 'Int', 'TklW', 'SCA', 'SCA90'
        ],
        "attacking_fullback": [
            'Touches', 'Mid 3rd', 'Att 3rd', 'Crs', 'CrsPA', 'Carries - PrgC',
            'Carries - 1/3', 'Carries - CPA', 'PrgP', 'PPA', 'KP', 'Ast', 'xA',
            'SCA', 'SCA90'
        ],
        "classic_dm": [
            'Touches', 'Def 3rd', 'Mid 3rd', 'TklW', 'Int', 'Recov', 'Fls', 'Fld',
            'Blocks', 'Passes Blocked', 'Total - Tkl', 'Tkl+Int', 'Clr',
            'Carries - TotDist', 'Short - Cmp%', 'Medium - Cmp%'
        ],
        "destroyer_dm": [
            'Total - Tkl', 'Tkl+Int', 'Tackles - Def 3rd', 'Tackles - Mid 3rd',
            'Dribblers- Tkl', 'Int', 'Fls', 'Fld', '2CrdY', 'CrdY', 'Blocks',
            'Recov', 'Carries - Dis'
        ],
        "deeplying_playmaker_dm": [
            'Touches', 'Def 3rd', 'Mid 3rd', 'Total - Cmp%', 'Medium - Cmp%',
            'Long - Cmp%', 'TotDist', 'PrgDist', 'PrgP', 'PPA', 'KP', 'TB',
            'SCA - PassLive', 'xA', 'Ast'
        ],
        "classic_cm": [
            'Touches', 'Mid 3rd', 'Att 3rd', 'Carries', 'Carries - PrgC',
            'Total - Cmp%', 'Medium - Cmp%', 'Tkl+Int', 'Fld', 'Fls', 'Int',
            'Recov'
        ],
        "playmaker_cm": [
            'Touches', 'Att 3rd', 'Carries - PrgC', 'Carries - CPA', 'PPA', 'KP',
            'SCA', 'SCA90', 'xA', 'Ast', 'TB', 'Total - Cmp%', 'PrgP'
        ],
        "workhorse_cm": [
            'Touches', 'Carries', 'Carries - TotDist', 'Tkl+Int',
            'Tackles - Mid 3rd', 'Fld', 'Fls', 'Blocks', 'Recov', 'Clr', 'Sh/90'
        ],
        "classic_am": [
            'Touches', 'Att 3rd', 'PPA', 'KP', 'xA', 'Ast', 'SCA', 'SCA90',
            'GCA', 'GCA90', 'xAG', 'A-xAG', 'Sh/90', 'SoT/90'
        ],
        "playmaker_am": [
            'Touches', 'Att 3rd', 'Carries - PrgC', 'Carries - CPA', 'PPA', 'KP',
            'xA', 'Ast', 'SCA - PassLive', 'SCA - Fld', 'GCA - PassLive', 'PrgP'
        ],
        "winger": [
            'Touches', 'Att 3rd', 'Take Ons - Attempted', 'Succ', 'Succ%',
            'Carries - PrgC', 'Carries - CPA', 'Crs', 'CrsPA', 'PPA', 'Sh', 'SoT',
            'xG', 'xA'
        ],
        "inside_forward": [
            'Touches', 'Att 3rd', 'Carries - PrgC', 'Carries - CPA', 'Sh', 'SoT',
            'xG', 'npxG', 'Gls', 'SCA', 'GCA', 'PPA', 'KP'
        ],
        "centre_forward": [
            'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'Gls', 'xG', 'npxG', 'G/Sh',
            'G/SoT', 'Dist', 'PK', 'SCA - Sh', 'GCA - Sh'
        ],
        "false_9": [
            'Touches', 'Att 3rd', 'Carries - PrgC', 'PPA', 'KP', 'Ast', 'xA',
            'xG', 'Sh/90', 'SoT/90', 'SCA90', 'GCA90'
        ]
    }

    clean_stat_mapping = {
        '1/3': 'Passes into Final Third',
        '2CrdY': 'Second Yellow Cards',
        'A-xAG': 'Assists minus xAG',
        'Ast': 'Assists',
        'Att 3rd': 'Touches in Attacking Third',
        'Blocks': 'Blocks',
        'Carries': 'Carries',
        'Carries - 1/3': 'Carries into Final Third',
        'Carries - CPA': 'Carries into Penalty Area',
        'Carries - Dis': 'Dispossessed on Carries',
        'Carries - Mis': 'Miscontrolled Carries',
        'Carries - PrgC': 'Progressive Carries',
        'Carries - PrgDist': 'Progressive Carry Distance',
        'Carries - TotDist': 'Total Carry Distance',
        'Clr': 'Clearances',
        'CrdR': 'Red Cards',
        'CrdY': 'Yellow Cards',
        'Crs': 'Crosses',
        'CrsPA': 'Crosses into Penalty Area',
        'Dead': 'Dead Ball Passes',
        'Def 3rd': 'Touches in Defensive Third',
        'Dist': 'Average Shot Distance',
        'Dribblers- Tkl': 'Tackles vs Dribblers',
        'Err': 'Errors Leading to Goals',
        'Fld': 'Fouled',
        'Fls': 'Fouls Committed',
        'G-xG': 'Goals minus xG',
        'GCA': 'Goal-Creating Actions',
        'GCA - Def': 'GCA via Defensive Actions',
        'GCA - Fld': 'GCA via Fouls Drawn',
        'GCA - PassDead': 'GCA via Dead Balls',
        'GCA - PassLive': 'GCA via Live Passes',
        'GCA - Sh': 'GCA via Shots',
        'GCA90': 'Goal-Creating Actions per 90',
        'Gls': 'Goals',
        'Int': 'Interceptions',
        'KP': 'Key Passes',
        'Long - Cmp%': 'Long Pass Completion %',
        'Lost': 'Aerial Duels Lost',
        'Medium - Cmp%': 'Medium Pass Completion %',
        'Mid 3rd': 'Touches in Middle Third',
        'OG': 'Own Goals',
        'Off': 'Offsides',
        'PK': 'Penalties Scored',
        'PKatt': 'Penalty Attempts',
        'PPA': 'Passes into Penalty Area',
        'Passes Blocked': 'Passes Blocked',
        'PrgDist': 'Progressive Passing Distance',
        'PrgP': 'Progressive Passes',
        'Recov': 'Ball Recoveries',
        'SCA': 'Shot-Creating Actions',
        'SCA - Def': 'SCA via Defensive Actions',
        'SCA - Fld': 'SCA via Fouls Drawn',
        'SCA - PassDead': 'SCA via Dead Balls',
        'SCA - PassLive': 'SCA via Live Passes',
        'SCA - Sh': 'SCA via Shots',
        'SCA90': 'Shot-Creating Actions per 90',
        'Sh': 'Shots',
        'Sh/90': 'Shots per 90',
        'Short - Cmp%': 'Short Pass Completion %',
        'Shots Blocked': 'Shots Blocked',
        'SoT': 'Shots on Target',
        'SoT%': 'Shot on Target %',
        'SoT/90': 'Shots on Target per 90',
        'Succ': 'Successful Take-Ons',
        'Succ%': 'Take-On Success %',
        'Sw': 'Switches of Play',
        'TB': 'Through Balls',
        'Tackles - Def 3rd': 'Tackles in Defensive Third',
        'Tackles - Mid 3rd': 'Tackles in Middle Third',
        'Take Ons - Attempted': 'Take-Ons Attempted',
        'Tkl+Int': 'Tackles + Interceptions',
        'TklW': 'Tackles Won',
        'TotDist': 'Total Passing Distance',
        'Total - Att': 'Total Passes Attempted',
        'Total - Cmp': 'Total Passes Completed',
        'Total - Cmp%': 'Total Pass Completion %',
        'Total - Tkl': 'Total Tackles',
        'Total Blocks': 'Blocks',
        'Touches': 'Touches',
        'Won': 'Aerial Duels Won',
        'Won%': 'Aerial Duel Win %',
        'np:G-xG': 'Non-Penalty Goals minus xG',
        'npxG': 'Non-Penalty xG',
        'npxG/Sh': 'Non-Penalty xG per Shot',
        'xA': 'Expected Assists (xA)',
        'xAG': 'Expected Assisted Goals',
        'xG': 'Expected Goals (xG)'
    }

    player_role_percentile_templates = {
        "Ball Playing CB": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Passes Blocked", "Clr",
            # Possession
            "Touches", "Total - Cmp%", "Short - Cmp%", "PrgR", "Carries - PrgDist",
            # Attacking
            "xA", "KP", "Long - Cmp%", "Sh/90", "Ast"
        ],
        "Classic CB": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Clr", "Recov",
            # Possession
            "Touches", "Total - Cmp%", "Long - Cmp%", "Short - Cmp%", "PrgR",
            # Attacking
            "xG", "KP", "xA", "Sh/90", "Ast"
        ],
        "Classic Fullback": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Tkl+Int", "Passes Blocked",
            # Possession
            "Carries - PrgDist", "Touches", "PrgR", "Total - Cmp%", "Att 3rd",
            # Attacking
            "xA", "KP", "CrsPA", "Ast", "Sh/90"
        ],
        "Inverted Fullback": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Clr", "Tkl+Int",
            # Possession
            "PrgR", "Carries - PrgDist", "Touches", "Short - Cmp%", "Total - Cmp%",
            # Attacking
            "xA", "KP", "CrsPA", "Ast", "xG"
        ],
        "Attacking Fullback": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Clr", "Tkl+Int",
            # Possession
            "Carries - PrgC", "Carries - PrgDist", "Touches", "PrgR", "Total - Cmp%",
            # Attacking
            "xA", "KP", "CrsPA", "Ast", "Sh/90"
        ],
        "Destroyer DM": [
            # Defensive
            "Tkl%", "Tkl+Int", "Blocks", "Int", "Passes Blocked",
            # Possession
            "Touches", "Total - Cmp%", "PrgR", "Carries - PrgDist", "Short - Cmp%",
            # Attacking
            "xA", "KP", "Ast", "Sh/90", "Long - Cmp%"
        ],
        "Deep Lying Playmaker CM": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Tkl+Int", "Passes Blocked",
            # Possession
            "Total - Cmp%", "PrgP", "Touches", "Carries - PrgDist", "Short - Cmp%",
            # Attacking
            "xA", "KP", "Ast", "Long - Cmp%", "Sh/90"
        ],
        "Box to Box CM": [
            # Defensive
            "Tkl%", "Int", "Tkl+Int", "Blocks", "Passes Blocked",
            # Possession
            "Touches", "Carries - PrgC", "PrgR", "Total - Cmp%", "Short - Cmp%",
            # Attacking
            "xG", "xA", "KP", "Ast", "Sh/90"
        ],
        "Playmaker CM": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Tkl+Int", "Passes Blocked",
            # Possession
            "Touches", "PrgP", "Carries - PrgDist", "Short - Cmp%", "Total - Cmp%",
            # Attacking
            "xA", "KP", "Ast", "xG", "SCA90"
        ],
        "Classic AM": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Tkl+Int", "Pressures",
            # Possession
            "Touches", "Carries - PrgC", "PrgR", "Total - Cmp%", "Short - Cmp%",
            # Attacking
            "xA", "KP", "Ast", "xG", "Sh/90"
        ],
        "Inside Forward": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Tkl+Int", "Dribblers- Tkl",
            # Possession
            "Carries - PrgDist", "Touches", "PrgR", "Total - Cmp%", "Short - Cmp%",
            # Attacking
            "xG", "xA", "KP", "Sh/90", "Ast"
        ],
        "Winger": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Tkl+Int", "Dribblers- Tkl",
            # Possession
            "Carries - PrgDist", "Carries - PrgC", "Touches", "PrgR", "Take Ons - Attempted",
            # Attacking
            "xA", "KP", "Ast", "Sh/90", "CrsPA"
        ],
        "Center Forward": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Pressures", "Dribblers- Tkl",
            # Possession
            "Touches", "Carries - PrgC", "PrgR", "Short - Cmp%", "Rec",
            # Attacking
            "xG", "xA", "KP", "Sh/90", "Ast"
        ],
        "False 9": [
            # Defensive
            "Tkl%", "Int", "Blocks", "Tkl+Int", "Pressures",
            # Possession
            "Touches", "Carries - PrgC", "Total - Cmp%", "Short - Cmp%", "Rec",
            # Attacking
            "xG", "xA", "KP", "Ast", "Sh/90"
        ]
    }
        

