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
from tqdm import tqdm
import time


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

    def generate_pitch_iq_scores(self, position):
        """
        Generate Pitch IQ scores for players in a specific position.
        
        Parameters:
        -----------
        position : str
            The position group to filter players by (e.g., "Forwards", "Defenders", etc.).
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the players' stats and their Pitch IQ scores.
        """
        # Step 1: Create the database
        stats = self.create_full_stats_db()

        # Step 2: Filter players by position
        key_stats_df = self.key_stats_db(stats, position)

        # Step 3: Calculate metric scores
        pitch_iq_scoring = self.create_metrics_scores(key_stats_df)

        # Step 4: Adjust player ratings
        pitch_iq_scoring = self.adjust_player_rating_range(pitch_iq_scoring)

        # Step 5: Merge results
        pitch_iq_scores = pd.merge(key_stats_df, pitch_iq_scoring[['Player', 'Passing_Score', 'Defending_Score', 'Creation_Score', 'Shooting_Score']], on='Player', how='left')

        return pitch_iq_scores
    
    def fuzzy_merge(self, df_1, df_2, key1, key2, threshold=97, limit=1):
        """
        Perform a fuzzy merge between two DataFrames based on string similarity.
        
        :param df_1: the left table to join
        :param df_2: the right table to join
        :param key1: key column of the left table
        :param key2: key column of the right table
        :param threshold: how close the matches should be to return a match, based on Levenshtein distance
        :param limit: the amount of matches that will get returned, sorted high to low
        :return: dataframe with both keys and matches
        """
        s = df_2[key2].tolist()
        
        m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))    
        df_1['matches'] = m
        
        m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
        df_1['matches'] = m2
    
        return df_1

    def remove_accents(self, input_str):
        """
        Remove accents from a string, replace special characters, and convert it to ASCII.
        
        :param input_str: input string with accents and special characters
        :return: string without accents and with special characters replaced
        """
        # Replace ø with o and ð with d
        input_str = input_str.replace('ø', 'o').replace('ð', 'd').replace('Ø', 'O').replace('Ð', 'D').replace('ı', 'i')
        
        # Normalize the string to decompose accented characters
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        
        # Encode to ASCII, ignoring non-ASCII characters
        only_ascii = nfkd_form.encode('ASCII', 'ignore')
        
        # Convert bytes to string and remove the "b''" wrapper
        only_ascii = str(only_ascii)
        only_ascii = only_ascii[2:-1]
        
        # Replace hyphens with spaces
        only_ascii = only_ascii.replace('-', ' ')
        
        return only_ascii

    def years_converter(self, variable_value):
        """
        Convert age values in the format 'YYYY-DDDD' to a float representing years.
        
        :param variable_value: age value in 'YYYY-DDDD' format
        :return: age as a float
        """
        if len(variable_value) > 3:
            years = variable_value[:-4]
            days = variable_value[3:]
            years_value = pd.to_numeric(years)
            days_value = pd.to_numeric(days)
            day_conv = days_value / 365
            final_val = years_value + day_conv
        else:
            final_val = pd.to_numeric(variable_value)

        return final_val
    
    def get_team_urls(self, x):  
        url = x
        data  = requests.get(url).text
        soup = BeautifulSoup(data)
        player_urls = []
        links = BeautifulSoup(data).select('th a')
        urls = [link['href'] for link in links]
        urls = list(set(urls))
        full_urls = []
        for y in urls:
            full_url = "https://fbref.com"+y
            full_urls.append(full_url)
        team_names = []
        for team in urls: 
            team_name_slice = team[20:-6]
            team_names.append(team_name_slice)
        list_of_tuples = list(zip(team_names, full_urls))
        Team_url_database = pd.DataFrame(list_of_tuples, columns = ['team_names', 'urls'])
        return Team_url_database


    def general_url_database(self, full_urls):
        """
        Create a database of player URLs and stats from a list of team URLs.
        
        :param full_urls: list of team URLs
        :return: DataFrame containing player stats and URLs
        """
        appended_data = []
        for team_url in full_urls:
            print(team_url)
            player_db = pd.DataFrame()
            player_urls = []
            links = BeautifulSoup(requests.get(team_url).text).select('th a')
            urls = [link['href'] for link in links]
            player_urls.append(urls)
            player_urls = [item for sublist in player_urls for item in sublist]
            player_urls.sort()
            player_urls = list(set(player_urls))
            p_url = list(filter(lambda k: 'players' in k, player_urls))
            url_final = []
            for y in p_url:
                full_url = "https://fbref.com" + y
                url_final.append(full_url)
            player_names = []
            for player in p_url:
                player_name_slice = player[21:]
                player_name_slice = player_name_slice.replace('-', ' ')
                player_names.append(player_name_slice)
            list_of_tuples = list(zip(player_names, url_final))
            play_url_database = pd.DataFrame(list_of_tuples, columns=['Player', 'urls'])
            player_db = pd.concat([play_url_database])

            table = BeautifulSoup(requests.get(team_url).text, 'html5').find('table')
            cols = []
            for header in table.find_all('th'):
                cols.append(header.string)
            cols = [i for i in cols if i is not None]
            columns = cols[6:39]  # Gets necessary column headers
            players = cols[39:-2]

            rows = []
            for rownum, row in enumerate(table.find_all('tr')):
                if len(row.find_all('td')) > 0:
                    rowdata = []
                    for i in range(0, len(row.find_all('td'))):
                        rowdata.append(row.find_all('td')[i].text)
                    rows.append(rowdata)
            df = pd.DataFrame(rows, columns=columns)

            df.drop(df.tail(2).index, inplace=True)
            df["Player"] = players
            df = df[["Player", "Pos", "Age", "Starts"]]

            df['Player'] = df.apply(lambda x: self.remove_accents(x['Player']), axis=1)
            test_merge = self.fuzzy_merge(df, player_db, 'Player', 'Player', threshold=90)
            test_merge = test_merge.rename(columns={'matches': 'Player', 'Player': 'matches'})
            final_merge = test_merge.merge(player_db, on='Player', how='left')
            del df, table
            time.sleep(10)
            appended_data.append(final_merge)
        appended_data = pd.concat(appended_data)
        return appended_data

    def get_360_scouting_report(self, url):
        """
        Generate the 360 scouting report URL for a player.
        
        :param url: player URL
        :return: 360 scouting report URL
        """
        start = url[0:38] + "scout/365_m1/"
        mod_string = url[38:]
        final_string = start + mod_string + "-Scouting-Report"
        return final_string

    def get_match_logs(self, url):
        """
        Generate the match logs URL for a player.
        
        :param url: player URL
        :return: match logs URL
        """
        start = url[0:38] + "matchlogs/2024-2025/summary/"
        mod_string = url[38:]
        final_string = start + mod_string + "-Match-Logs"
        return final_string

    def create_player_phonebook(self, top_5_league_stats_urls):
        """
        Create a player phonebook with scouting URLs and match logs.
        
        :param top_5_league_stats_urls: list of URLs for top 5 league stats
        :return: DataFrame containing player phonebook
        """
        list_of_dfs = []
        for url in tqdm(top_5_league_stats_urls):
            team_urls = self.get_team_urls(url)
            full_urls = list(team_urls.urls.unique())
            Player_db = self.general_url_database(full_urls)
            Player_db['Age'] = Player_db.apply(lambda x: self.years_converter(x['Age']), axis=1)
            Player_db = Player_db.drop(columns=['matches'])
            Player_db = Player_db.dropna()
            Player_db['scouting_url'] = Player_db.apply(lambda x: self.get_360_scouting_report(x['urls']), axis=1)
            Player_db['match_logs'] = Player_db.apply(lambda x: self.get_match_logs(x['urls']), axis=1)
            Player_db["position_group"] = Player_db.Pos.apply(lambda x: self.position_grouping(x))
            Player_db.reset_index(drop=True)
            Player_db[["Starts"]] = Player_db[["Starts"]].apply(pd.to_numeric)
            list_of_dfs.append(Player_db)
        dfs = pd.concat(list_of_dfs)
        return dfs
    
    def league_performance_df(self, match_links):
        data_append = []
        for x in match_links:
            print(x)
            warnings.filterwarnings("ignore")
            url = x
            page =requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            name = [element.text for element in soup.find_all("span")]
            name = name[7]
            html_content = requests.get(url).text.replace('<!--', '').replace('-->', '')
            df = pd.read_html(html_content)
            df[0].columns = df[0].columns.droplevel(0) # drop top header row
            stats = df[0]
            stats = stats[(stats.Comp.isin(['La Liga', 'Premier League', 'Bundesliga', 'Serie A', 'Ligue 1'])) & (stats.Pos != "On matchday squad, but did not play")]
            season = stats[['Date','Gls', 'Ast',  'xG', 'npxG', 'xAG', 'Squad']]
            columns_to_convert = ['Gls', 'Ast', 'xG', 'npxG', 'xAG']
            for col in columns_to_convert:
                season[col] = pd.to_numeric(season[col], errors='coerce').fillna(0).astype(float)
            season = season.rename({'Squad': 'team'}, axis=1)
            season['Player'] = name
            data_append.append(season)
            del df, soup
            time.sleep(10)
        df_total = pd.concat(data_append)

        return df_total
    
    def create_kmeans_df(self, df): 
        KMeans_cols = ['Player','Total - Cmp%','KP', 'TB','Sw','PPA', 'PrgP','Tkl%','Blocks', 'Tkl+Int','Clr', 'Carries - PrgDist','SCA90','GCA90','CrsPA','xA', 'Rec','PrgR','xG', 'Sh','SoT']

        df = df[KMeans_cols]
        player_names = df['Player'].tolist() 

        df = df.drop(['Player'], axis = 1) 

        x = df.values 
        scaler = preprocessing.MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        X_norm = pd.DataFrame(x_scaled)

        pca = PCA(n_components = 2)
        reduced = pd.DataFrame(pca.fit_transform(X_norm))

        wcss = [] 
        for i in range(1, 11): 
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(reduced) 
            wcss.append(kmeans.inertia_)

        kmeans = KMeans(n_clusters=7)
        kmeans = kmeans.fit(reduced)

        labels = kmeans.predict(reduced)
        clusters = kmeans.labels_.tolist()

        reduced['cluster'] = clusters
        reduced['name'] = player_names
        reduced.columns = ['x', 'y', 'cluster', 'name']
        reduced.head()

        return reduced
    
    def create_clustering_chart(self, df,position):
        # Create the scatter plot using lmplot
        ax = sns.lmplot(x="x", y="y", hue='cluster', data=df, legend=False,
                        fit_reg=False, height=20,scatter_kws={"s": 250})

        texts = []
        for x, y, s in zip(df.x, df.y, df.name):
            texts.append(plt.text(x, y, s,fontweight='light'))


        # Additional axes for logos and titles
        fig = plt.gcf()
        ax1 = plt.gca()

        # Add title and logos to the current figure
        fig.text(.1, 1.08, f'KMeans clustering - {position}', size=30, font='Karla')
        fig.text(.1, 1.03, '24/25 Season | Viz by @stephenaq7 | Data via FBREF', size=20, font='Karla')

        ax2 = fig.add_axes([0.01, 0.175, 0.07, 1.75])
        ax2.axis('off')
        img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/premier-league-2-logo.png')
        ax2.imshow(img)

        ax3 = fig.add_axes([0.85, 0.175, 0.1, 1.75])
        ax3.axis('off')
        img = image.imread('/Users/stephenahiabah/Desktop/Code/cannoniq/Images/piqmain.png')
        ax3.imshow(img)

        # Set axis limits and labels for the lmplot
        ax1.set(ylim=(-2, 2))
        plt.tick_params(labelsize=15)
        plt.xlabel("PC 1", fontsize=20)
        plt.ylabel("PC 2", fontsize=20)

        plt.tight_layout()
        plt.show()




    
    