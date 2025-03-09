import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import numpy as np
from math import pi
from urllib.request import urlopen
import matplotlib.patheffects as pe
from highlight_text import fig_text
from adjustText import adjust_text
from tabulate import tabulate
import matplotlib.style as style
import unicodedata
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from matplotlib import cm
from highlight_text import fig_text

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

style.use('fivethirtyeight')

from PIL import Image
import urllib
import os
import math
from PIL import Image
import matplotlib.image as image
pd.options.display.max_columns = None

class CreateFBRefDatabase:
    def __init__(self, passing_url=None, shooting_url=None, pass_type_url=None, defence_url=None, gca_url=None, poss_url=None, misc_url=None):
        # Default URLs (can be overridden by passing arguments)
        self.fbref_passing = passing_url or 'https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats'
        self.fbref_shooting = shooting_url or 'https://fbref.com/en/comps/Big5/shooting/players/Big-5-European-Leagues-Stats'
        self.fbref_pass_type = pass_type_url or 'https://fbref.com/en/comps/Big5/passing_types/players/Big-5-European-Leagues-Stats'
        self.fbref_defence = defence_url or 'https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats'
        self.fbref_gca = gca_url or 'https://fbref.com/en/comps/Big5/gca/players/Big-5-European-Leagues-Stats'
        self.fbref_poss = poss_url or 'https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats'
        self.fbref_misc = misc_url or 'https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats'

        # Position groupings
        self.keepers = ['GK']
        self.defenders = ["DF", 'DF,MF']
        self.wing_backs = ['FW,DF', 'DF,FW']
        self.midfielders = ['MF,DF', 'MF']
        self.forwards = ['FW', 'MF,FW', "FW,MF"]

    def position_grouping(self, x):
        if x in self.keepers:
            return "GK"
        elif x in self.defenders:
            return "Defender"
        elif x in self.wing_backs:
            return "Wing-Back"
        elif x in self.midfielders:
            return "Central Midfielders"
        elif x in self.forwards:
            return "Forwards"
        else:
            return "unidentified position"

    def create_full_stats_db(self):
        # Passing columns 
        pass_ = self.fbref_passing
        page = requests.get(pass_)
        soup = BeautifulSoup(page.content, 'html.parser')
        html_content = requests.get(pass_).text.replace('<!--', '').replace('-->', '')
        pass_df = pd.read_html(html_content)
        pass_df[-1].columns = pass_df[-1].columns.droplevel(0)
        pass_stats = pass_df[-1]
        pass_prefixes = {1: 'Total - ', 2: 'Short - ', 3: 'Medium - ', 4: 'Long - '}
        pass_column_occurrences = {'Cmp': 0, 'Att': 0, 'Cmp%': 0}
        pass_new_column_names = []
        for col_name in pass_stats.columns:
            if col_name in pass_column_occurrences:
                pass_column_occurrences[col_name] += 1
                prefix = pass_prefixes[pass_column_occurrences[col_name]]
                pass_new_column_names.append(prefix + col_name)
            else:
                pass_new_column_names.append(col_name)
        pass_stats.columns = pass_new_column_names
        pass_stats = pass_stats[pass_stats['Player'] != 'Player']

        # Shooting columns 
        shot_ = self.fbref_shooting
        page = requests.get(shot_)
        soup = BeautifulSoup(page.content, 'html.parser')
        html_content = requests.get(shot_).text.replace('<!--', '').replace('-->', '')
        shot_df = pd.read_html(html_content)
        shot_df[-1].columns = shot_df[-1].columns.droplevel(0) # drop top header row
        shot_stats = shot_df[-1]
        shot_stats = shot_stats[shot_stats['Player'] != 'Player']    

        # Pass Type columns 
        pass_type = self.fbref_pass_type
        page = requests.get(pass_type)
        soup = BeautifulSoup(page.content, 'html.parser')
        html_content = requests.get(pass_type).text.replace('<!--', '').replace('-->', '')
        pass_type_df = pd.read_html(html_content)
        pass_type_df[-1].columns = pass_type_df[-1].columns.droplevel(0) # drop top header row
        pass_type_stats = pass_type_df[-1]
        pass_type_stats = pass_type_stats[pass_type_stats['Player'] != 'Player']

        # GCA columns 
        gca_ = self.fbref_gca
        page = requests.get(gca_)
        soup = BeautifulSoup(page.content, 'html.parser')
        html_content = requests.get(gca_).text.replace('<!--', '').replace('-->', '')
        gca_df = pd.read_html(html_content)
        gca_df[-1].columns = gca_df[-1].columns.droplevel(0)
        gca_stats = gca_df[-1]
        gca_prefixes = {1: 'SCA - ', 2: 'GCA - '}
        gca_column_occurrences = {'PassLive': 0, 'PassDead': 0, 'TO%': 0, 'Sh': 0, 'Fld': 0, 'Def': 0}
        gca_new_column_names = []
        for col_name in gca_stats.columns:
            if col_name in gca_column_occurrences:
                gca_column_occurrences[col_name] += 1
                prefix = gca_prefixes[gca_column_occurrences[col_name]]
                gca_new_column_names.append(prefix + col_name)
            else:
                gca_new_column_names.append(col_name)
        gca_stats.columns = gca_new_column_names
        gca_stats = gca_stats[gca_stats['Player'] != 'Player']
        
        # Defense columns 
        defence_ = self.fbref_defence
        page = requests.get(defence_)
        soup = BeautifulSoup(page.content, 'html.parser')
        html_content = requests.get(defence_).text.replace('<!--', '').replace('-->', '')
        defence_df = pd.read_html(html_content)
        defence_df[-1].columns = defence_df[-1].columns.droplevel(0) # drop top header row
        defence_stats = defence_df[-1]
        rename_columns = {
        'Def 3rd': 'Tackles - Def 3rd',
        'Mid 3rd': 'Tackles - Mid 3rd',
        'Att 3rd': 'Tackles - Att 3rd',
        'Blocks': 'Total Blocks',
        'Sh': 'Shots Blocked',
        'Pass': 'Passes Blocked'}
        defence_stats.rename(columns = rename_columns, inplace=True)
        defence_prefixes = {1: 'Total - ', 2: 'Dribblers- '}
        defence_column_occurrences = {'Tkl': 0}
        new_column_names = []
        for col_name in defence_stats.columns:
            if col_name in defence_column_occurrences:
                defence_column_occurrences[col_name] += 1
                prefix = defence_prefixes[defence_column_occurrences[col_name]]
                new_column_names.append(prefix + col_name)
            else:
                new_column_names.append(col_name)
        defence_stats.columns = new_column_names
        defence_stats = defence_stats[defence_stats['Player'] != 'Player']

        # possession columns 
        poss_ = self.fbref_poss
        page = requests.get(poss_)
        soup = BeautifulSoup(page.content, 'html.parser')
        html_content = requests.get(poss_).text.replace('<!--', '').replace('-->', '')
        poss_df = pd.read_html(html_content)
        poss_df[-1].columns = poss_df[-1].columns.droplevel(0) # drop top header row
        poss_stats = poss_df[-1]
        rename_columns = {
        'TotDist': 'Carries - TotDist',
        'PrgDist': 'Carries - PrgDist',
        'PrgC': 'Carries - PrgC',
        '1/3': 'Carries - 1/3',
        'CPA': 'Carries - CPA',
        'Mis': 'Carries - Mis',
        'Dis': 'Carries - Dis',
        'Att': 'Take Ons - Attempted'  }
        poss_stats.rename(columns=rename_columns, inplace=True)
        poss_stats = poss_stats[poss_stats['Player'] != 'Player']

        # misc columns 
        misc_ = self.fbref_misc
        page = requests.get(misc_)
        soup = BeautifulSoup(page.content, 'html.parser')
        html_content = requests.get(misc_).text.replace('<!--', '').replace('-->', '')
        misc_df = pd.read_html(html_content)
        misc_df[-1].columns = misc_df[-1].columns.droplevel(0) # drop top header row
        misc_stats = misc_df[-1]
        misc_stats = misc_stats[misc_stats['Player'] != 'Player']

        index_df = misc_stats[['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s']]

        data_frames = [poss_stats, misc_stats, pass_stats ,defence_stats, shot_stats, gca_stats, pass_type_stats]
        for df in data_frames:
            if df is not None:  # Checking if the DataFrame exists
                df.drop(columns=['Matches', 'Rk', 'Comp'], inplace=True, errors='ignore')
                df.dropna(axis=0, how='any', inplace=True)

                index_df = pd.merge(index_df, df, on=['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', '90s'], how='left')
        index_df["position_group"] = index_df.Pos.apply(lambda x: self.position_grouping(x))  

        index_df.fillna(0, inplace=True)

        non_numeric_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'position_group']
        
        def clean_non_convertible_values(value):
            try:
                return pd.to_numeric(value)
            except (ValueError, TypeError):
                return np.nan

        index_df = index_df.reset_index()

        # Iterate through each column, converting non-numeric columns to numeric
        for col in index_df.columns:
            if col not in non_numeric_cols:
                index_df[col] = index_df[col].apply(clean_non_convertible_values)

        
        return index_df

    def per_90fi(self, dataframe):
        # Replace empty strings ('') with NaN
        dataframe = dataframe.replace('', np.nan)
        
        # Fill NaN values with 0
        dataframe = dataframe.fillna(0)
        
        # Identify numeric columns excluding '90s' and columns with '90' or '%' in their names
        exclude_columns = ['index', 'Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', 'position_group']
        numeric_columns = [col for col in dataframe.columns 
                           if pd.api.types.is_numeric_dtype(dataframe[col])  # Check if column is numeric
                           and col != '90s'  # Exclude '90s' column
                           and not any(exc_col in col for exc_col in exclude_columns)  # Exclude non-numeric columns
                           and ('90' not in col)  # Exclude columns with '90' in their names
                           and ('%' not in col)]  # Exclude columns with '%' in their names
        
        # Create a mask to avoid division by zero
        mask = (dataframe['90s'] != 0)
        
        # Divide each numeric column by the '90s' column row-wise
        dataframe.loc[mask, numeric_columns] = dataframe.loc[mask, numeric_columns].div(dataframe.loc[mask, '90s'], axis=0)

        return dataframe

    def key_stats_db(self, df, position):
        non_numeric_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'position_group']
        core_stats = ['90s','Total - Cmp%','KP', 'TB','Sw','PPA', 'PrgP','Tkl%','Blocks', 'Tkl+Int','Clr', 'Carries - PrgDist','SCA90','GCA90','CrsPA','xA', 'Rec','PrgR','xG', 'Sh','SoT']
        df.dropna(axis=0, how='any', inplace=True)
        key_stats_df = df[df['position_group'] == position]
        key_stats_df = key_stats_df[non_numeric_cols + core_stats]
        key_stats_df = key_stats_df[key_stats_df['90s'] > 5]
        key_stats_df = self.per_90fi(key_stats_df)
        return key_stats_df

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