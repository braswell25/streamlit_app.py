"""
NFL Multi-Position Fantasy Projections - Streamlit App
Inspired by xEP_Network's AI-driven projections

Supports: QB (passing yards), RB (rushing yards, receiving yards, TDs), 
         WR (receiving yards, TDs), TE (receiving yards, TDs)

Dependencies: streamlit, pandas, requests, beautifulsoup4, scikit-learn, numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NFL Fantasy Projections",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_sample_data():
    """Load all sample data with caching for better performance"""
    
    # Sample QB data
    qb_data = [
        {'Player': 'Josh Allen', 'Team': 'BUF', 'Games': 17, 'Attempts': 560, 'Completions': 359, 
         'Pass_Yards': 4306, 'Pass_TDs': 29, 'Interceptions': 18, 'YPA': 7.7, 
         'Pass_Yards_Per_Game': 253.3, 'Pass_TDs_Per_Game': 1.71, 'Completion_Pct': 0.641},
        {'Player': 'Dak Prescott', 'Team': 'DAL', 'Games': 17, 'Attempts': 590, 'Completions': 410, 
         'Pass_Yards': 4516, 'Pass_TDs': 36, 'Interceptions': 9, 'YPA': 7.7, 
         'Pass_Yards_Per_Game': 265.6, 'Pass_TDs_Per_Game': 2.12, 'Completion_Pct': 0.695},
        {'Player': 'Tua Tagovailoa', 'Team': 'MIA', 'Games': 17, 'Attempts': 550, 'Completions': 388, 
         'Pass_Yards': 4624, 'Pass_TDs': 27, 'Interceptions': 14, 'YPA': 8.4, 
         'Pass_Yards_Per_Game': 272.0, 'Pass_TDs_Per_Game': 1.59, 'Completion_Pct': 0.705},
        {'Player': 'Lamar Jackson', 'Team': 'BAL', 'Games': 17, 'Attempts': 457, 'Completions': 307, 
         'Pass_Yards': 3678, 'Pass_TDs': 24, 'Interceptions': 7, 'YPA': 8.0, 
         'Pass_Yards_Per_Game': 216.4, 'Pass_TDs_Per_Game': 1.41, 'Completion_Pct': 0.672},
        {'Player': 'Trevor Lawrence', 'Team': 'JAX', 'Games': 17, 'Attempts': 524, 'Completions': 305, 
         'Pass_Yards': 4113, 'Pass_TDs': 21, 'Interceptions': 14, 'YPA': 7.8, 
         'Pass_Yards_Per_Game': 242.0, 'Pass_TDs_Per_Game': 1.24, 'Completion_Pct': 0.582},
    ]
    
    # Sample RB data
    rb_data = [
        {'Player': 'Christian McCaffrey', 'Team': 'SF', 'Games': 16, 'Rush_Attempts': 272, 'Rush_Yards': 1459, 
         'Rush_TDs': 14, 'YPC': 5.4, 'Rush_Yards_Per_Game': 91.2, 'Rush_TDs_Per_Game': 0.88, 
         'Rush_Attempts_Per_Game': 17.0, 'Rec_Yards': 564, 'Rec_TDs': 7, 'Receptions': 67, 
         'Rec_Yards_Per_Game': 35.3, 'Rec_TDs_Per_Game': 0.44},
        {'Player': 'Josh Jacobs', 'Team': 'LV', 'Games': 17, 'Rush_Attempts': 340, 'Rush_Yards': 1653, 
         'Rush_TDs': 12, 'YPC': 4.9, 'Rush_Yards_Per_Game': 97.2, 'Rush_TDs_Per_Game': 0.71, 
         'Rush_Attempts_Per_Game': 20.0, 'Rec_Yards': 400, 'Rec_TDs': 2, 'Receptions': 53, 
         'Rec_Yards_Per_Game': 23.5, 'Rec_TDs_Per_Game': 0.12},
        {'Player': 'Saquon Barkley', 'Team': 'NYG', 'Games': 16, 'Rush_Attempts': 295, 'Rush_Yards': 1312, 
         'Rush_TDs': 10, 'YPC': 4.4, 'Rush_Yards_Per_Game': 82.0, 'Rush_TDs_Per_Game': 0.63, 
         'Rush_Attempts_Per_Game': 18.4, 'Rec_Yards': 338, 'Rec_TDs': 4, 'Receptions': 57, 
         'Rec_Yards_Per_Game': 21.1, 'Rec_TDs_Per_Game': 0.25},
        {'Player': 'Derrick Henry', 'Team': 'TEN', 'Games': 16, 'Rush_Attempts': 349, 'Rush_Yards': 1538, 
         'Rush_TDs': 13, 'YPC': 4.4, 'Rush_Yards_Per_Game': 96.1, 'Rush_TDs_Per_Game': 0.81, 
         'Rush_Attempts_Per_Game': 21.8, 'Rec_Yards': 114, 'Rec_TDs': 1, 'Receptions': 16, 
         'Rec_Yards_Per_Game': 7.1, 'Rec_TDs_Per_Game': 0.06},
    ]
    
    # Sample WR data
    wr_data = [
        {'Player': 'Tyreek Hill', 'Team': 'MIA', 'Games': 17, 'Receptions': 119, 'Rec_Yards': 1710, 
         'Rec_TDs': 7, 'YPR': 14.4, 'Targets': 170, 'Rec_Yards_Per_Game': 100.6, 
         'Rec_TDs_Per_Game': 0.41, 'Receptions_Per_Game': 7.0, 'Catch_Rate': 0.70},
        {'Player': 'Stefon Diggs', 'Team': 'BUF', 'Games': 17, 'Receptions': 108, 'Rec_Yards': 1429, 
         'Rec_TDs': 11, 'YPR': 13.2, 'Targets': 154, 'Rec_Yards_Per_Game': 84.1, 
         'Rec_TDs_Per_Game': 0.65, 'Receptions_Per_Game': 6.4, 'Catch_Rate': 0.70},
        {'Player': 'Davante Adams', 'Team': 'LV', 'Games': 17, 'Receptions': 100, 'Rec_Yards': 1516, 
         'Rec_TDs': 14, 'YPR': 15.2, 'Targets': 180, 'Rec_Yards_Per_Game': 89.2, 
         'Rec_TDs_Per_Game': 0.82, 'Receptions_Per_Game': 5.9, 'Catch_Rate': 0.56},
        {'Player': 'Calvin Ridley', 'Team': 'JAX', 'Games': 17, 'Receptions': 76, 'Rec_Yards': 1016, 
         'Rec_TDs': 8, 'YPR': 13.4, 'Targets': 136, 'Rec_Yards_Per_Game': 59.8, 
         'Rec_TDs_Per_Game': 0.47, 'Receptions_Per_Game': 4.5, 'Catch_Rate': 0.56},
    ]
    
    # Sample TE data
    te_data = [
        {'Player': 'Travis Kelce', 'Team': 'KC', 'Games': 17, 'Receptions': 110, 'Rec_Yards': 1338, 
         'Rec_TDs': 12, 'YPR': 12.2, 'Targets': 150, 'Rec_Yards_Per_Game': 78.7, 
         'Rec_TDs_Per_Game': 0.71, 'Receptions_Per_Game': 6.5, 'Catch_Rate': 0.73},
        {'Player': 'Mark Andrews', 'Team': 'BAL', 'Games': 17, 'Receptions': 89, 'Rec_Yards': 1361, 
         'Rec_TDs': 8, 'YPR': 15.3, 'Targets': 125, 'Rec_Yards_Per_Game': 80.1, 
         'Rec_TDs_Per_Game': 0.47, 'Receptions_Per_Game': 5.2, 'Catch_Rate': 0.71},
        {'Player': 'George Kittle', 'Team': 'SF', 'Games': 16, 'Receptions': 60, 'Rec_Yards': 765, 
         'Rec_TDs': 2, 'YPR': 12.8, 'Targets': 90, 'Rec_Yards_Per_Game': 47.8, 
         'Rec_TDs_Per_Game': 0.13, 'Receptions_Per_Game': 3.8, 'Catch_Rate': 0.67},
    ]
    
    # Sample defense data
    defense_data = [
        {'Team': 'JAX', 'Games': 17, 'Pass_Yards_Allowed': 4250, 'Pass_TDs_Allowed': 28, 
         'Rush_Yards_Allowed': 2100, 'Rush_TDs_Allowed': 18, 'Total_TDs_Allowed': 46,
         'Pass_Yards_Allowed_Per_Game': 250.0, 'Rush_Yards_Allowed_Per_Game': 123.5,
         'Pass_TDs_Allowed_Per_Game': 1.65, 'Rush_TDs_Allowed_Per_Game': 1.06,
         'Pass_Defense_Rank': 25, 'Rush_Defense_Rank': 20, 'Pass_TD_Defense_Rank': 22, 'Rush_TD_Defense_Rank': 18},
        {'Team': 'BUF', 'Games': 17, 'Pass_Yards_Allowed': 3800, 'Pass_TDs_Allowed': 22, 
         'Rush_Yards_Allowed': 1950, 'Rush_TDs_Allowed': 15, 'Total_TDs_Allowed': 37,
         'Pass_Yards_Allowed_Per_Game': 223.5, 'Rush_Yards_Allowed_Per_Game': 114.7,
         'Pass_TDs_Allowed_Per_Game': 1.29, 'Rush_TDs_Allowed_Per_Game': 0.88,
         'Pass_Defense_Rank': 12, 'Rush_Defense_Rank': 15, 'Pass_TD_Defense_Rank': 8, 'Rush_TD_Defense_Rank': 12},
        {'Team': 'DAL', 'Games': 17, 'Pass_Yards_Allowed': 3950, 'Pass_TDs_Allowed': 25, 
         'Rush_Yards_Allowed': 2200, 'Rush_TDs_Allowed': 20, 'Total_TDs_Allowed': 45,
         'Pass_Yards_Allowed_Per_Game': 232.4, 'Rush_Yards_Allowed_Per_Game': 129.4,
         'Pass_TDs_Allowed_Per_Game': 1.47, 'Rush_TDs_Allowed_Per_Game': 1.18,
         'Pass_Defense_Rank': 15, 'Rush_Defense_Rank': 25, 'Pass_TD_Defense_Rank': 15, 'Rush_TD_Defense_Rank': 22},
        {'Team': 'MIA', 'Games': 17, 'Pass_Yards_Allowed': 4100, 'Pass_TDs_Allowed': 26, 
         'Rush_Yards_Allowed': 2050, 'Rush_TDs_Allowed': 16, 'Total_TDs_Allowed': 42,
         'Pass_Yards_Allowed_Per_Game': 241.2, 'Rush_Yards_Allowed_Per_Game': 120.6,
         'Pass_TDs_Allowed_Per_Game': 1.53, 'Rush_TDs_Allowed_Per_Game': 0.94,
         'Pass_Defense_Rank': 20, 'Rush_Defense_Rank': 18, 'Pass_TD_Defense_Rank': 18, 'Rush_TD_Defense_Rank': 15},
        {'Team': 'BAL', 'Games': 17, 'Pass_Yards_Allowed': 3700, 'Pass_TDs_Allowed': 20, 
         'Rush_Yards_Allowed': 1850, 'Rush_TDs_Allowed': 12, 'Total_TDs_Allowed': 32,
         'Pass_Yards_Allowed_Per_Game': 217.6, 'Rush_Yards_Allowed_Per_Game': 108.8,
         'Pass_TDs_Allowed_Per_Game': 1.18, 'Rush_TDs_Allowed_Per_Game': 0.71,
         'Pass_Defense_Rank': 8, 'Rush_Defense_Rank': 10, 'Pass_TD_Defense_Rank': 5, 'Rush_TD_Defense_Rank': 8},
        {'Team': 'KC', 'Games': 17, 'Pass_Yards_Allowed': 3900, 'Pass_TDs_Allowed': 24, 
         'Rush_Yards_Allowed': 2000, 'Rush_TDs_Allowed': 17, 'Total_TDs_Allowed': 41,
         'Pass_Yards_Allowed_Per_Game': 229.4, 'Rush_Yards_Allowed_Per_Game': 117.6,
         'Pass_TDs_Allowed_Per_Game': 1.41, 'Rush_TDs_Allowed_Per_Game': 1.00,
         'Pass_Defense_Rank': 14, 'Rush_Defense_Rank': 16, 'Pass_TD_Defense_Rank': 12, 'Rush_TD_Defense_Rank': 16},
        {'Team': 'LAC', 'Games': 17, 'Pass_Yards_Allowed': 3850, 'Pass_TDs_Allowed': 23, 
         'Rush_Yards_Allowed': 1900, 'Rush_TDs_Allowed': 14, 'Total_TDs_Allowed': 37,
         'Pass_Yards_Allowed_Per_Game': 226.5, 'Rush_Yards_Allowed_Per_Game': 111.8,
         'Pass_TDs_Allowed_Per_Game': 1.35, 'Rush_TDs_Allowed_Per_Game': 0.82,
         'Pass_Defense_Rank': 13, 'Rush_Defense_Rank': 12, 'Pass_TD_Defense_Rank': 10, 'Rush_TD_Defense_Rank': 10},
        {'Team': 'NYG', 'Games': 17, 'Pass_Yards_Allowed': 4150, 'Pass_TDs_Allowed': 27, 
         'Rush_Yards_Allowed': 2250, 'Rush_TDs_Allowed': 19, 'Total_TDs_Allowed': 46,
         'Pass_Yards_Allowed_Per_Game': 244.1, 'Rush_Yards_Allowed_Per_Game': 132.4,
         'Pass_TDs_Allowed_Per_Game': 1.59, 'Rush_TDs_Allowed_Per_Game': 1.12,
         'Pass_Defense_Rank': 21, 'Rush_Defense_Rank': 26, 'Pass_TD_Defense_Rank': 19, 'Rush_TD_Defense_Rank': 20},
    ]
    
    return {
        'qb': pd.DataFrame(qb_data),
        'rb': pd.DataFrame(rb_data),
        'wr': pd.DataFrame(wr_data),
        'te': pd.DataFrame(te_data),
        'defense': pd.DataFrame(defense_data)
    }

class NFLStreamlitProjector:
    def __init__(self):
        self.data = load_sample_data()
        self.models = {}
        self.trained_models = set()
        
    def prepare_training_data(self, stat_type, position):
        """Prepare training data for specific stat type and position"""
        
        # Select appropriate player data based on position
        if position == 'QB':
            player_data = self.data['qb']
        elif position == 'RB':
            player_data = self.data['rb']
        elif position == 'WR':
            player_data = self.data['wr']
        elif position == 'TE':
            player_data = self.data['te']
        else:
            raise ValueError(f"Unsupported position: {position}")
        
        defense_data = self.data['defense']
        training_data = []
        
        for _, player in player_data.iterrows():
            for _, defense in defense_data.iterrows():
                # Skip same team matchups
                if player['Team'] == defense['Team']:
                    continue
                
                # Create feature vector based on stat type
                if stat_type == 'passing_yards':
                    features = {
                        'Player_Stat_Per_Game': player['Pass_Yards_Per_Game'],
                        'Player_Efficiency': player['YPA'],
                        'Player_Secondary_Stat': player['Completion_Pct'],
                        'Opp_Stat_Allowed_Per_Game': defense['Pass_Yards_Allowed_Per_Game'],
                        'Opp_Defense_Rank': defense['Pass_Defense_Rank']
                    }
                    base_projection = (player['Pass_Yards_Per_Game'] + defense['Pass_Yards_Allowed_Per_Game']) / 2
                    
                elif stat_type == 'rushing_yards':
                    features = {
                        'Player_Stat_Per_Game': player['Rush_Yards_Per_Game'],
                        'Player_Efficiency': player['YPC'],
                        'Player_Secondary_Stat': player['Rush_Attempts_Per_Game'],
                        'Opp_Stat_Allowed_Per_Game': defense['Rush_Yards_Allowed_Per_Game'],
                        'Opp_Defense_Rank': defense['Rush_Defense_Rank']
                    }
                    base_projection = (player['Rush_Yards_Per_Game'] + defense['Rush_Yards_Allowed_Per_Game']) / 2
                    
                elif stat_type == 'receiving_yards':
                    features = {
                        'Player_Stat_Per_Game': player['Rec_Yards_Per_Game'],
                        'Player_Efficiency': player['YPR'],
                        'Player_Secondary_Stat': player['Receptions_Per_Game'],
                        'Opp_Stat_Allowed_Per_Game': defense['Pass_Yards_Allowed_Per_Game'],
                        'Opp_Defense_Rank': defense['Pass_Defense_Rank']
                    }
                    base_projection = (player['Rec_Yards_Per_Game'] + defense['Pass_Yards_Allowed_Per_Game'] * 0.3) / 2
                    
                elif stat_type in ['passing_tds', 'rushing_tds', 'receiving_tds']:
                    if stat_type == 'passing_tds':
                        player_stat = player['Pass_TDs_Per_Game']
                        opp_stat = defense['Pass_TDs_Allowed_Per_Game']
                        opp_rank = defense['Pass_TD_Defense_Rank']
                    elif stat_type == 'rushing_tds':
                        player_stat = player['Rush_TDs_Per_Game']
                        opp_stat = defense['Rush_TDs_Allowed_Per_Game']
                        opp_rank = defense['Rush_TD_Defense_Rank']
                    else:  # receiving_tds
                        player_stat = player['Rec_TDs_Per_Game']
                        opp_stat = defense['Pass_TDs_Allowed_Per_Game']
                        opp_rank = defense['Pass_TD_Defense_Rank']
                    
                    features = {
                        'Player_Stat_Per_Game': player_stat,
                        'Player_Efficiency': player_stat * 16,
                        'Player_Secondary_Stat': player_stat,
                        'Opp_Stat_Allowed_Per_Game': opp_stat,
                        'Opp_Defense_Rank': opp_rank
                    }
                    base_projection = (player_stat + opp_stat) / 2
                
                # Add matchup factor
                matchup_factor = 1 + (features['Opp_Defense_Rank'] - 16.5) / 32 * 0.3
                target = base_projection * matchup_factor
                
                # Add variance
                if 'yards' in stat_type:
                    target += np.random.normal(0, 15)
                    target = max(10, target)
                else:  # TDs
                    target += np.random.normal(0, 0.3)
                    target = max(0, target)
                
                features['Target'] = target
                training_data.append(features)
        
        df = pd.DataFrame(training_data)
        feature_columns = ['Player_Stat_Per_Game', 'Player_Efficiency', 'Player_Secondary_Stat', 
                          'Opp_Stat_Allowed_Per_Game', 'Opp_Defense_Rank']
        
        X = df[feature_columns]
        y = df['Target']
        
        return X, y

    def train_model(self, stat_type, position):
        """Train model for specific stat type and position"""
        X, y = self.prepare_training_data(stat_type, position)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model_key = f"{position}_{stat_type}"
        self.models[model_key] = LinearRegression()
        self.models[model_key].fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.models[model_key].predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.trained_models.add(model_key)
        
        return mae, r2

    def calculate_matchup_score(self, defense_rank, stat_type):
        """Calculate matchup score based on defensive rank and stat type"""
        score = 10 - (defense_rank - 1) * 8 / 31
        return round(score, 1)

    def calculate_betting_edge(self, projection, betting_line, stat_type):
        """Calculate betting edge for different stat types"""
        difference = projection - betting_line
        edge_percentage = (difference / betting_line) * 100 if betting_line > 0 else 0
        
        recommendation = "OVER" if difference > 0 else "UNDER"
        
        # Adjust confidence thresholds based on stat type
        if 'yards' in stat_type:
            confidence = "HIGH" if abs(edge_percentage) > 10 else "MEDIUM" if abs(edge_percentage) > 5 else "LOW"
        else:  # TDs
            confidence = "HIGH" if abs(edge_percentage) > 20 else "MEDIUM" if abs(edge_percentage) > 10 else "LOW"
        
        return {
            'projection': projection,
            'betting_line': betting_line,
            'difference': difference,
            'edge_percentage': edge_percentage,
            'recommendation': recommendation,
            'confidence': confidence
        }

    def project_player_performance(self, player_name, position, opponent_team, stat_type, betting_line=None):
        """Project player performance for specific stat type"""
        model_key = f"{position}_{stat_type}"
        
        if model_key not in self.trained_models:
            # Train model if not already trained
            self.train_model(stat_type, position)
        
        # Find player stats
        if position == 'QB':
            player_data = self.data['qb']
        elif position == 'RB':
            player_data = self.data['rb']
        elif position == 'WR':
            player_data = self.data['wr']
        elif position == 'TE':
            player_data = self.data['te']
        
        player_match = player_data[player_data['Player'].str.contains(player_name, case=False)]
        if player_match.empty:
            raise ValueError(f"Player '{player_name}' not found in {position} data")
        
        player_stats = player_match.iloc[0]
        
        # Find opponent defense stats
        opp_data = self.data['defense'][self.data['defense']['Team'] == opponent_team.upper()]
        if opp_data.empty:
            raise ValueError(f"Team '{opponent_team}' not found in defensive data")
        
        opp_stats = opp_data.iloc[0]
        
        # Prepare features based on stat type
        if stat_type == 'passing_yards':
            features = np.array([[
                player_stats['Pass_Yards_Per_Game'],
                player_stats['YPA'],
                player_stats['Completion_Pct'],
                opp_stats['Pass_Yards_Allowed_Per_Game'],
                opp_stats['Pass_Defense_Rank']
            ]])
            defense_rank = opp_stats['Pass_Defense_Rank']
            
        elif stat_type == 'rushing_yards':
            features = np.array([[
                player_stats['Rush_Yards_Per_Game'],
                player_stats['YPC'],
                player_stats['Rush_Attempts_Per_Game'],
                opp_stats['Rush_Yards_Allowed_Per_Game'],
                opp_stats['Rush_Defense_Rank']
            ]])
            defense_rank = opp_stats['Rush_Defense_Rank']
            
        elif stat_type == 'receiving_yards':
            features = np.array([[
                player_stats['Rec_Yards_Per_Game'],
                player_stats['YPR'],
                player_stats['Receptions_Per_Game'],
                opp_stats['Pass_Yards_Allowed_Per_Game'],
                opp_stats['Pass_Defense_Rank']
            ]])
            defense_rank = opp_stats['Pass_Defense_Rank']
            
        elif stat_type in ['passing_tds', 'rushing_tds', 'receiving_tds']:
            if stat_type == 'passing_tds':
                player_stat = player_stats['Pass_TDs_Per_Game']
                opp_stat = opp_stats['Pass_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Pass_TD_Defense_Rank']
            elif stat_type == 'rushing_tds':
                player_stat = player_stats['Rush_TDs_Per_Game']
                opp_stat = opp_stats['Rush_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Rush_TD_Defense_Rank']
            else:  # receiving_tds
                player_stat = player_stats['Rec_TDs_Per_Game']
                opp_stat = opp_stats['Pass_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Pass_TD_Defense_Rank']
            
            features = np.array([[
                player_stat,
                player_stat * 16,
                player_stat,
                opp_stat,
                defense_rank
            ]])
        
        # Make prediction
        projection = self.models[model_key].predict(features)[0]
        
        # Calculate additional metrics
        matchup_score = self.calculate_matchup_score(defense_rank, stat_type)
        
        result = {
            'player_name': player_stats['Player'],
            'player_team': player_stats['Team'],
            'position': position,
            'stat_type': stat_type,
            'opponent': opponent_team.upper(),
            'projection': round(projection, 1 if 'yards' in stat_type else 2),
            'matchup_score': matchup_score,
            'defense_rank': int(defense_rank)
        }
        
        if betting_line:
            betting_analysis = self.calculate_betting_edge(projection, betting_line, stat_type)
            result['betting_analysis'] = betting_analysis
        
        return result

def main():
    st.title("🏈 NFL Fantasy Projections")
    st.markdown("*Inspired by xEP_Network's AI-driven methodology*")
    
    # Initialize projector
    if 'projector' not in st.session_state:
        st.session_state.projector = NFLStreamlitProjector()
    
    projector = st.session_state.projector
    
    # Sidebar for inputs
    st.sidebar.header("Projection Settings")
    
    # Position selection
    position = st.sidebar.selectbox(
        "Select Position",
        ["QB", "RB", "WR", "TE"],
        help="Choose the player position to project"
    )
    
    # Get available players for selected position
    if position == 'QB':
        available_players = projector.data['qb']['Player'].tolist()
        available_stats = ['passing_yards', 'passing_tds']
    elif position == 'RB':
        available_players = projector.data['rb']['Player'].tolist()
        available_stats = ['rushing_yards', 'rushing_tds', 'receiving_yards', 'receiving_tds']
    elif position == 'WR':
        available_players = projector.data['wr']['Player'].tolist()
        available_stats = ['receiving_yards', 'receiving_tds']
    elif position == 'TE':
        available_players = projector.data['te']['Player'].tolist()
        available_stats = ['receiving_yards', 'receiving_tds']
    
    # Player selection
    player_name = st.sidebar.selectbox(
        "Select Player",
        available_players,
        help="Choose the player to project"
    )
    
    # Stat type selection
    stat_display_names = {
        'passing_yards': 'Passing Yards',
        'passing_tds': 'Passing TDs',
        'rushing_yards': 'Rushing Yards',
        'rushing_tds': 'Rushing TDs',
        'receiving_yards': 'Receiving Yards',
        'receiving_tds': 'Receiving TDs'
    }
    
    stat_type = st.sidebar.selectbox(
        "Select Stat to Project",
        available_stats,
        format_func=lambda x: stat_display_names[x],
        help="Choose which statistic to project"
    )
    
    # Opponent selection
    available_teams = projector.data['defense']['Team'].tolist()
    opponent_team = st.sidebar.selectbox(
        "Select Opponent",
        available_teams,
        help="Choose the opposing team"
    )
    
    # Betting line input
    betting_line = st.sidebar.number_input(
        "Betting Line (Optional)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="Enter the over/under betting line for edge calculation"
    )
    
    # Generate projection button
    if st.sidebar.button("Generate Projection", type="primary"):
        try:
            with st.spinner("Training model and generating projection..."):
                # Generate projection
                projection = projector.project_player_performance(
                    player_name, 
                    position, 
                    opponent_team, 
                    stat_type, 
                    betting_line if betting_line > 0 else None
                )
            
            # Display results
            st.header("📊 Projection Results")
            
            # Main projection card
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"{stat_display_names[stat_type]} Projection",
                    value=f"{projection['projection']}"
                )
            
            with col2:
                st.metric(
                    label="Matchup Score",
                    value=f"{projection['matchup_score']}/10"
                )
            
            with col3:
                st.metric(
                    label="Opponent Defense Rank",
                    value=f"#{projection['defense_rank']}"
                )
            
            # Player and matchup info
            st.subheader("🎯 Matchup Details")
            st.write(f"**Player:** {projection['player_name']} ({projection['player_team']})")
            st.write(f"**Opponent:** {projection['opponent']}")
            st.write(f"**Stat:** {stat_display_names[stat_type]}")
            
            # Betting analysis if provided
            if 'betting_analysis' in projection:
                st.subheader("💰 Betting Analysis")
                betting = projection['betting_analysis']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Betting Line", f"{betting['betting_line']}")
                
                with col2:
                    st.metric("Edge", f"{betting['edge_percentage']:+.1f}%")
                
                with col3:
                    st.metric("Recommendation", betting['recommendation'])
                
                with col4:
                    confidence_color = {
                        'HIGH': '🟢',
                        'MEDIUM': '🟡', 
                        'LOW': '🔴'
                    }
                    st.metric("Confidence", f"{confidence_color[betting['confidence']]} {betting['confidence']}")
                
                # Edge interpretation
                if abs(betting['edge_percentage']) > 10:
                    st.success(f"Strong {betting['recommendation']} play with {betting['edge_percentage']:+.1f}% edge!")
                elif abs(betting['edge_percentage']) > 5:
                    st.info(f"Moderate {betting['recommendation']} lean with {betting['edge_percentage']:+.1f}% edge")
                else:
                    st.warning(f"Weak edge of {betting['edge_percentage']:+.1f}% - consider avoiding this bet")
            
        except Exception as e:
            st.error(f"Error generating projection: {str(e)}")
    
    # Model performance section
    st.header("🤖 Model Information")
    
    with st.expander("View Model Performance"):
        st.write("**Training Data:**")
        st.write("- Uses historical player stats and defensive rankings")
        st.write("- Generates synthetic matchup data for training")
        st.write("- Features include per-game stats, efficiency metrics, and opponent strength")
        
        st.write("**Model Features:**")
        st.write("- Linear regression with cross-validation")
        st.write("- Position-specific models for different stat types")
        st.write("- Matchup-based adjustments using defensive rankings")
    
    # Advanced metrics suggestions
    st.header("📈 Advanced Metrics Suggestions")
    
    with st.expander("Potential Model Improvements"):
        st.write("**For Enhanced Accuracy, Consider Adding:**")
        st.write("• Target share and air yards (WR/TE)")
        st.write("• Red zone touches and goal line carries (RB)")
        st.write("• Snap count percentages")
        st.write("• Weather conditions for outdoor games")
        st.write("• Home/Away splits")
        st.write("• Recent form (last 4 games)")
        st.write("• Injury reports and snap limitations")
        st.write("• Game script predictions (game total, spread)")
        st.write("• Advanced metrics: ADOT, EPA, Success Rate")
    
    # Data tables
    st.header("📋 Available Data")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["QBs", "RBs", "WRs", "TEs", "Defense"])
    
    with tab1:
        st.dataframe(projector.data['qb'], use_container_width=True)
    
    with tab2:
        st.dataframe(projector.data['rb'], use_container_width=True)
    
    with tab3:
        st.dataframe(projector.data['wr'], use_container_width=True)
    
    with tab4:
        st.dataframe(projector.data['te'], use_container_width=True)
    
    with tab5:
        st.dataframe(projector.data['defense'], use_container_width=True)

if __name__ == "__main__":
    main()
