import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class PremierLeagueDataLoader:
    """
    Loads and preprocesses Premier League data from Football Data UK
    """
    
    def __init__(self, data_dir: str = "premier_league_data"):
        self.data_dir = data_dir
        self.base_url = "https://www.football-data.co.uk/mmz4281/"
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Season codes (last two digits of each year)
        self.season_codes = {
            '2015-16': '1516', '2016-17': '1617', '2017-18': '1718', 
            '2018-19': '1819', '2019-20': '1920', '2020-21': '2021',
            '2021-22': '2122', '2022-23': '2223', '2023-24': '2324'
        }
        
        # Column mapping from Football Data UK to our model format
        self.column_mapping = {
            # Basic match info
            'Date': 'date',
            'HomeTeam': 'home_team', 
            'AwayTeam': 'away_team',
            'Referee': 'referee',
            
            # Half-time results
            'HTHG': 'ht_goals_home',  # Half Time Home Goals
            'HTAG': 'ht_goals_away',  # Half Time Away Goals
            
            # Full-time results  
            'FTHG': 'ft_goals_home',  # Full Time Home Goals
            'FTAG': 'ft_goals_away',  # Full Time Away Goals
            
            # Shots
            'HS': 'shots_home',       # Home Shots
            'AS': 'shots_away',       # Away Shots
            'HST': 'ft_sot_home',     # Home Shots on Target
            'AST': 'ft_sot_away',     # Away Shots on Target
            
            # Cards and fouls
            'HF': 'fouls_home',       # Home Fouls
            'AF': 'fouls_away',       # Away Fouls  
            'HC': 'corners_home',     # Home Corners
            'AC': 'corners_away',     # Away Corners
            'HY': 'yellows_home',     # Home Yellow Cards
            'AY': 'yellows_away',     # Away Yellow Cards
            'HR': 'ht_reds_home',     # Home Red Cards (we'll use as half-time proxy)
            'AR': 'ht_reds_away',     # Away Red Cards (we'll use as half-time proxy)
        }
    
    def download_season_data(self, season: str, force_redownload: bool = False) -> str:
        """Download data for a specific season"""
        if season not in self.season_codes:
            available = ', '.join(self.season_codes.keys())
            raise ValueError(f"Season {season} not available. Available: {available}")
        
        season_code = self.season_codes[season]
        filename = f"E0{season_code}.csv"  # E0 = Premier League
        filepath = os.path.join(self.data_dir, filename)
        
        # Check if file exists and don't redownload unless forced
        if os.path.exists(filepath) and not force_redownload:
            print(f"✅ Season {season} data already exists: {filepath}")
            return filepath
        
        url = f"{self.base_url}{season_code}/E0.csv"
        print(f"📥 Downloading {season} data from {url}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {filepath}")
            return filepath
            
        except requests.RequestException as e:
            print(f"❌ Failed to download {season}: {e}")
            return None
    
    def download_multiple_seasons(self, seasons: List[str], force_redownload: bool = False) -> List[str]:
        """Download data for multiple seasons"""
        filepaths = []
        for season in seasons:
            filepath = self.download_season_data(season, force_redownload)
            if filepath:
                filepaths.append(filepath)
        return filepaths
    
    def load_season_data(self, season: str) -> pd.DataFrame:
        """Load and preprocess data for one season"""
        season_code = self.season_codes.get(season)
        if not season_code:
            raise ValueError(f"Season {season} not supported")
        
        filename = f"E0{season_code}.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Data file not found. Downloading {season}...")
            filepath = self.download_season_data(season)
            if not filepath:
                raise FileNotFoundError(f"Could not download data for {season}")
        
        # Load the CSV
        print(f"📊 Loading {season} data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Add season column
        df['season'] = season
        
        # Add league column
        df['league'] = 'Premier League'
        
        print(f"   Loaded {len(df)} matches for {season}")
        return df
    
    def load_multiple_seasons(self, seasons: List[str]) -> pd.DataFrame:
        """Load and combine multiple seasons"""
        all_data = []
        
        for season in seasons:
            try:
                season_data = self.load_season_data(season)
                all_data.append(season_data)
            except Exception as e:
                print(f"❌ Error loading {season}: {e}")
        
        if not all_data:
            raise ValueError("No season data could be loaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Combined {len(combined_df)} matches from {len(all_data)} seasons")
        
        return combined_df
    
    def preprocess_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data to match our model's expected format"""
        print("🔄 Preprocessing data for model...")
        
        # Create a copy
        processed_df = df.copy()
        
        # Convert date
        processed_df['Date'] = pd.to_datetime(processed_df['Date'], format='%d/%m/%Y')
        
        # Rename columns using our mapping
        available_cols = set(processed_df.columns)
        rename_dict = {old: new for old, new in self.column_mapping.items() if old in available_cols}
        processed_df = processed_df.rename(columns=rename_dict)
        
        # Add venue column (assume all matches are at home team's venue)
        processed_df['venue'] = 'Home'
        
        # Create half-time shots on target (estimate from full-time data)
        # We'll assume half-time SoT is roughly 40-60% of full-time SoT
        if 'ft_sot_home' in processed_df.columns:
            processed_df['ht_sot_home'] = (processed_df['ft_sot_home'] * 
                                          np.random.uniform(0.4, 0.6, len(processed_df))).round().astype(int)
            processed_df['ht_sot_away'] = (processed_df['ft_sot_away'] * 
                                          np.random.uniform(0.4, 0.6, len(processed_df))).round().astype(int)
        
        # Ensure half-time shots on target don't exceed half-time goals
        if 'ht_sot_home' in processed_df.columns and 'ht_goals_home' in processed_df.columns:
            processed_df['ht_sot_home'] = np.maximum(processed_df['ht_sot_home'], 
                                                    processed_df['ht_goals_home'])
            processed_df['ht_sot_away'] = np.maximum(processed_df['ht_sot_away'], 
                                                    processed_df['ht_goals_away'])
        
        # Add simple ELO ratings (we'll create a basic system)
        processed_df = self._add_elo_ratings(processed_df)
        
        # Select only the columns we need for the model
        model_columns = [
            'date', 'league', 'season', 'venue',
            'ht_goals_home', 'ht_goals_away', 'ht_sot_home', 'ht_sot_away',
            'ht_reds_home', 'ht_reds_away', 'elo_home', 'elo_away',
            'ft_goals_home', 'ft_goals_away', 'ft_sot_home', 'ft_sot_away'
        ]
        
        # Only keep columns that exist
        available_model_cols = [col for col in model_columns if col in processed_df.columns]
        processed_df = processed_df[available_model_cols]
        
        # Fill missing values
        processed_df = self._fill_missing_values(processed_df)
        
        print(f"✅ Preprocessed data shape: {processed_df.shape}")
        print(f"   Columns: {list(processed_df.columns)}")
        
        return processed_df
    
    def _add_elo_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple ELO ratings based on historical performance"""
        print("   Adding ELO ratings...")
        
        # Initialize ELO ratings
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        elo_ratings = {team: 1500 for team in teams}  # Start all teams at 1500
        
        # Sort by date to process matches chronologically
        df_sorted = df.sort_values('date').copy()
        
        elo_home_list = []
        elo_away_list = []
        
        K = 32  # ELO K-factor
        
        for _, row in df_sorted.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            
            # Get current ratings
            elo_home = elo_ratings[home_team]
            elo_away = elo_ratings[away_team]
            
            elo_home_list.append(elo_home)
            elo_away_list.append(elo_away)
            
            # Calculate expected scores
            expected_home = 1 / (1 + 10**((elo_away - elo_home) / 400))
            expected_away = 1 - expected_home
            
            # Determine actual result
            home_goals = row['ft_goals_home'] if 'ft_goals_home' in row else 0
            away_goals = row['ft_goals_away'] if 'ft_goals_away' in row else 0
            
            if home_goals > away_goals:
                actual_home, actual_away = 1, 0
            elif away_goals > home_goals:
                actual_home, actual_away = 0, 1
            else:
                actual_home, actual_away = 0.5, 0.5
            
            # Update ratings
            elo_ratings[home_team] += K * (actual_home - expected_home)
            elo_ratings[away_team] += K * (actual_away - expected_away)
        
        # Add to dataframe (restore original order)
        df_sorted['elo_home'] = elo_home_list
        df_sorted['elo_away'] = elo_away_list
        
        # Merge back to original dataframe
        df = df.merge(df_sorted[['date', 'home_team', 'away_team', 'elo_home', 'elo_away']], 
                     on=['date', 'home_team', 'away_team'], how='left')
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with reasonable defaults"""
        # Numeric columns - fill with 0 or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.startswith('ht_') or col.startswith('ft_'):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "premier_league_processed.csv"):
        """Save processed data to CSV"""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Saved processed data to: {filepath}")
        return filepath

def main():
    """Example usage"""
    print("🏈 Premier League Data Loader")
    print("=" * 50)
    
    # Initialize loader
    loader = PremierLeagueDataLoader()
    
    # Define seasons to load (last 5 seasons)
    seasons_to_load = ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24']
    
    try:
        # Download data if needed
        print("📥 Downloading season data...")
        filepaths = loader.download_multiple_seasons(seasons_to_load)
        
        # Load and combine all seasons
        print("\n📊 Loading season data...")
        combined_df = loader.load_multiple_seasons(seasons_to_load)
        
        print(f"\nRaw data sample:")
        print(combined_df.head())
        
        # Preprocess for our model
        print(f"\n🔄 Preprocessing for model...")
        processed_df = loader.preprocess_for_model(combined_df)
        
        print(f"\nProcessed data sample:")
        print(processed_df.head())
        
        print(f"\nProcessed data info:")
        print(processed_df.info())
        
        # Save processed data
        saved_path = loader.save_processed_data(processed_df, "matches.csv")
        
        print(f"\n✅ Premier League data ready!")
        print(f"   📁 File: {saved_path}")
        print(f"   📊 Matches: {len(processed_df)}")
        print(f"   📅 Date range: {processed_df['date'].min()} to {processed_df['date'].max()}")
        print(f"\n🚀 You can now use this data with your football prediction model!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try with fewer seasons")
        print("3. Check if the Football Data UK website is accessible")

if __name__ == "__main__":
    main()
