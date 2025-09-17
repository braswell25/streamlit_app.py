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
def load_comprehensive_data():
    """Load comprehensive NFL player and team data with Week 3 2025 adjustments"""
    
    qb_data = [
        # AFC East - Updated with Week 1-2 2025 performance trends
        {'Player': 'Josh Allen', 'Team': 'BUF', 'Games': 2, 'Attempts': 72, 'Completions': 48, 
         'Pass_Yards': 634, 'Pass_TDs': 4, 'Interceptions': 0, 'YPA': 8.8, 
         'Pass_Yards_Per_Game': 317.0, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.667,
         'ADOT': 8.5, 'EPA_Per_Play': 0.22, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.14,
         'Historical_YPG': 253.3, 'Current_Form_Factor': 1.25},  # Hot start factor
        {'Player': 'Tua Tagovailoa', 'Team': 'MIA', 'Games': 2, 'Attempts': 68, 'Completions': 49, 
         'Pass_Yards': 521, 'Pass_TDs': 3, 'Interceptions': 1, 'YPA': 7.7, 
         'Pass_Yards_Per_Game': 260.5, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.721,
         'ADOT': 7.0, 'EPA_Per_Play': 0.14, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.09,
         'Historical_YPG': 272.0, 'Current_Form_Factor': 0.96},
        {'Player': 'Aaron Rodgers', 'Team': 'NYJ', 'Games': 2, 'Attempts': 58, 'Completions': 38, 
         'Pass_Yards': 449, 'Pass_TDs': 2, 'Interceptions': 1, 'YPA': 7.7, 
         'Pass_Yards_Per_Game': 224.5, 'Pass_TDs_Per_Game': 1.0, 'Completion_Pct': 0.655,
         'ADOT': 8.8, 'EPA_Per_Play': 0.11, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.15,
         'Historical_YPG': 229.2, 'Current_Form_Factor': 0.98},
        {'Player': 'Drake Maye', 'Team': 'NE', 'Games': 1, 'Attempts': 23, 'Completions': 14, 
         'Pass_Yards': 243, 'Pass_TDs': 1, 'Interceptions': 1, 'YPA': 10.6, 
         'Pass_Yards_Per_Game': 243.0, 'Pass_TDs_Per_Game': 1.0, 'Completion_Pct': 0.609,
         'ADOT': 9.2, 'EPA_Per_Play': 0.15, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.17,
         'Historical_YPG': 190.9, 'Current_Form_Factor': 1.27},  # Rookie improvement
        
        # AFC North - Updated with current season performance
        {'Player': 'Lamar Jackson', 'Team': 'BAL', 'Games': 2, 'Attempts': 56, 'Completions': 39, 
         'Pass_Yards': 512, 'Pass_TDs': 3, 'Interceptions': 0, 'YPA': 9.1, 
         'Pass_Yards_Per_Game': 256.0, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.696,
         'ADOT': 8.8, 'EPA_Per_Play': 0.19, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.16,
         'Historical_YPG': 216.4, 'Current_Form_Factor': 1.18},
        {'Player': 'Joe Burrow', 'Team': 'CIN', 'Games': 2, 'Attempts': 74, 'Completions': 48, 
         'Pass_Yards': 497, 'Pass_TDs': 5, 'Interceptions': 1, 'YPA': 6.7, 
         'Pass_Yards_Per_Game': 248.5, 'Pass_TDs_Per_Game': 2.5, 'Completion_Pct': 0.649,
         'ADOT': 8.1, 'EPA_Per_Play': 0.13, 'Red_Zone_TDs': 4, 'Deep_Ball_Pct': 0.12,
         'Historical_YPG': 263.2, 'Current_Form_Factor': 0.94},
        {'Player': 'Deshaun Watson', 'Team': 'CLE', 'Games': 2, 'Attempts': 51, 'Completions': 30, 
         'Pass_Yards': 282, 'Pass_TDs': 1, 'Interceptions': 2, 'YPA': 5.5, 
         'Pass_Yards_Per_Game': 141.0, 'Pass_TDs_Per_Game': 0.5, 'Completion_Pct': 0.588,
         'ADOT': 7.5, 'EPA_Per_Play': -0.08, 'Red_Zone_TDs': 0, 'Deep_Ball_Pct': 0.10,
         'Historical_YPG': 150.7, 'Current_Form_Factor': 0.94},
        {'Player': 'Russell Wilson', 'Team': 'PIT', 'Games': 2, 'Attempts': 62, 'Completions': 42, 
         'Pass_Yards': 542, 'Pass_TDs': 4, 'Interceptions': 1, 'YPA': 8.7, 
         'Pass_Yards_Per_Game': 271.0, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.677,
         'ADOT': 9.1, 'EPA_Per_Play': 0.17, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.16,
         'Historical_YPG': 211.5, 'Current_Form_Factor': 1.28},  # Strong start in Pittsburgh
        
        # AFC South - Updated with early season data
        {'Player': 'C.J. Stroud', 'Team': 'HOU', 'Games': 2, 'Attempts': 65, 'Completions': 42, 
         'Pass_Yards': 582, 'Pass_TDs': 3, 'Interceptions': 1, 'YPA': 9.0, 
         'Pass_Yards_Per_Game': 291.0, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.646,
         'ADOT': 9.1, 'EPA_Per_Play': 0.16, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.15,
         'Historical_YPG': 241.6, 'Current_Form_Factor': 1.20},
        {'Player': 'Anthony Richardson', 'Team': 'IND', 'Games': 1, 'Attempts': 28, 'Completions': 18, 
         'Pass_Yards': 202, 'Pass_TDs': 2, 'Interceptions': 0, 'YPA': 7.2, 
         'Pass_Yards_Per_Game': 202.0, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.643,
         'ADOT': 9.5, 'EPA_Per_Play': 0.10, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.19,
         'Historical_YPG': 139.5, 'Current_Form_Factor': 1.45},  # Returning from injury
        {'Player': 'Trevor Lawrence', 'Team': 'JAX', 'Games': 2, 'Attempts': 70, 'Completions': 40, 
         'Pass_Yards': 488, 'Pass_TDs': 2, 'Interceptions': 3, 'YPA': 7.0, 
         'Pass_Yards_Per_Game': 244.0, 'Pass_TDs_Per_Game': 1.0, 'Completion_Pct': 0.571,
         'ADOT': 8.5, 'EPA_Per_Play': 0.05, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.13,
         'Historical_YPG': 242.0, 'Current_Form_Factor': 1.01},
        {'Player': 'Will Levis', 'Team': 'TEN', 'Games': 2, 'Attempts': 63, 'Completions': 35, 
         'Pass_Yards': 392, 'Pass_TDs': 3, 'Interceptions': 2, 'YPA': 6.2, 
         'Pass_Yards_Per_Game': 196.0, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.556,
         'ADOT': 7.8, 'EPA_Per_Play': 0.01, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.11,
         'Historical_YPG': 161.5, 'Current_Form_Factor': 1.21},
        
        # AFC West - Updated with current performance
        {'Player': 'Patrick Mahomes', 'Team': 'KC', 'Games': 2, 'Attempts': 75, 'Completions': 50, 
         'Pass_Yards': 522, 'Pass_TDs': 3, 'Interceptions': 2, 'YPA': 7.0, 
         'Pass_Yards_Per_Game': 261.0, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.667,
         'ADOT': 7.7, 'EPA_Per_Play': 0.09, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.10,
         'Historical_YPG': 246.1, 'Current_Form_Factor': 1.06},
        {'Player': 'Justin Herbert', 'Team': 'LAC', 'Games': 2, 'Attempts': 69, 'Completions': 45, 
         'Pass_Yards': 478, 'Pass_TDs': 2, 'Interceptions': 1, 'YPA': 6.9, 
         'Pass_Yards_Per_Game': 239.0, 'Pass_TDs_Per_Game': 1.0, 'Completion_Pct': 0.652,
         'ADOT': 8.0, 'EPA_Per_Play': 0.07, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.11,
         'Historical_YPG': 227.6, 'Current_Form_Factor': 1.05},
        {'Player': 'Bo Nix', 'Team': 'DEN', 'Games': 2, 'Attempts': 60, 'Completions': 39, 
         'Pass_Yards': 463, 'Pass_TDs': 4, 'Interceptions': 1, 'YPA': 7.7, 
         'Pass_Yards_Per_Game': 231.5, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.650,
         'ADOT': 7.5, 'EPA_Per_Play': 0.09, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.09,
         'Historical_YPG': 222.1, 'Current_Form_Factor': 1.04},
        {'Player': 'Gardner Minshew', 'Team': 'LV', 'Games': 2, 'Attempts': 67, 'Completions': 43, 
         'Pass_Yards': 412, 'Pass_TDs': 1, 'Interceptions': 2, 'YPA': 6.1, 
         'Pass_Yards_Per_Game': 206.0, 'Pass_TDs_Per_Game': 0.5, 'Completion_Pct': 0.642,
         'ADOT': 7.0, 'EPA_Per_Play': 0.02, 'Red_Zone_TDs': 0, 'Deep_Ball_Pct': 0.08,
         'Historical_YPG': 194.4, 'Current_Form_Factor': 1.06},
        
        # NFC East - Updated with current season performance
        {'Player': 'Jalen Hurts', 'Team': 'PHI', 'Games': 2, 'Attempts': 64, 'Completions': 41, 
         'Pass_Yards': 501, 'Pass_TDs': 3, 'Interceptions': 2, 'YPA': 7.8, 
         'Pass_Yards_Per_Game': 250.5, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.641,
         'ADOT': 7.8, 'EPA_Per_Play': 0.07, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.11,
         'Historical_YPG': 226.9, 'Current_Form_Factor': 1.10},
        {'Player': 'Dak Prescott', 'Team': 'DAL', 'Games': 2, 'Attempts': 73, 'Completions': 51, 
         'Pass_Yards': 563, 'Pass_TDs': 4, 'Interceptions': 1, 'YPA': 7.7, 
         'Pass_Yards_Per_Game': 281.5, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.699,
         'ADOT': 7.6, 'EPA_Per_Play': 0.16, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.09,
         'Historical_YPG': 265.6, 'Current_Form_Factor': 1.06},
        {'Player': 'Daniel Jones', 'Team': 'NYG', 'Games': 2, 'Attempts': 63, 'Completions': 38, 
         'Pass_Yards': 421, 'Pass_TDs': 2, 'Interceptions': 1, 'YPA': 6.7, 
         'Pass_Yards_Per_Game': 210.5, 'Pass_TDs_Per_Game': 1.0, 'Completion_Pct': 0.603,
         'ADOT': 7.3, 'EPA_Per_Play': 0.01, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.09,
         'Historical_YPG': 170.4, 'Current_Form_Factor': 1.24},
        {'Player': 'Jayden Daniels', 'Team': 'WAS', 'Games': 2, 'Attempts': 61, 'Completions': 37, 
         'Pass_Yards': 535, 'Pass_TDs': 4, 'Interceptions': 1, 'YPA': 8.8, 
         'Pass_Yards_Per_Game': 267.5, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.607,
         'ADOT': 8.9, 'EPA_Per_Play': 0.11, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.14,
         'Historical_YPG': 209.9, 'Current_Form_Factor': 1.27},
        
        # NFC North - Updated with current season performance
        {'Player': 'Jared Goff', 'Team': 'DET', 'Games': 2, 'Attempts': 78, 'Completions': 52, 
         'Pass_Yards': 555, 'Pass_TDs': 4, 'Interceptions': 2, 'YPA': 7.1, 
         'Pass_Yards_Per_Game': 277.5, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.667,
         'ADOT': 7.2, 'EPA_Per_Play': 0.12, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.08,
         'Historical_YPG': 269.1, 'Current_Form_Factor': 1.03},
        {'Player': 'Jordan Love', 'Team': 'GB', 'Games': 2, 'Attempts': 71, 'Completions': 45, 
         'Pass_Yards': 548, 'Pass_TDs': 5, 'Interceptions': 1, 'YPA': 7.7, 
         'Pass_Yards_Per_Game': 274.0, 'Pass_TDs_Per_Game': 2.5, 'Completion_Pct': 0.634,
         'ADOT': 7.9, 'EPA_Per_Play': 0.10, 'Red_Zone_TDs': 4, 'Deep_Ball_Pct': 0.11,
         'Historical_YPG': 244.6, 'Current_Form_Factor': 1.12},
        {'Player': 'Sam Darnold', 'Team': 'MIN', 'Games': 2, 'Attempts': 66, 'Completions': 40, 
         'Pass_Yards': 518, 'Pass_TDs': 6, 'Interceptions': 2, 'YPA': 7.9, 
         'Pass_Yards_Per_Game': 259.0, 'Pass_TDs_Per_Game': 3.0, 'Completion_Pct': 0.606,
         'ADOT': 8.0, 'EPA_Per_Play': 0.13, 'Red_Zone_TDs': 4, 'Deep_Ball_Pct': 0.12,
         'Historical_YPG': 220.0, 'Current_Form_Factor': 1.18},
        {'Player': 'Caleb Williams', 'Team': 'CHI', 'Games': 2, 'Attempts': 77, 'Completions': 46, 
         'Pass_Yards': 481, 'Pass_TDs': 2, 'Interceptions': 1, 'YPA': 6.2, 
         'Pass_Yards_Per_Game': 240.5, 'Pass_TDs_Per_Game': 1.0, 'Completion_Pct': 0.597,
         'ADOT': 8.4, 'EPA_Per_Play': 0.05, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.13,
         'Historical_YPG': 208.3, 'Current_Form_Factor': 1.15},
        
        # NFC South - Updated with current season performance
        {'Player': 'Baker Mayfield', 'Team': 'TB', 'Games': 2, 'Attempts': 65, 'Completions': 44, 
         'Pass_Yards': 569, 'Pass_TDs': 4, 'Interceptions': 0, 'YPA': 8.8, 
         'Pass_Yards_Per_Game': 284.5, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.677,
         'ADOT': 7.8, 'EPA_Per_Play': 0.11, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.10,
         'Historical_YPG': 251.7, 'Current_Form_Factor': 1.13},
        {'Player': 'Kirk Cousins', 'Team': 'ATL', 'Games': 2, 'Attempts': 63, 'Completions': 42, 
         'Pass_Yards': 472, 'Pass_TDs': 3, 'Interceptions': 3, 'YPA': 7.5, 
         'Pass_Yards_Per_Game': 236.0, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.667,
         'ADOT': 7.5, 'EPA_Per_Play': 0.04, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.09,
         'Historical_YPG': 227.3, 'Current_Form_Factor': 1.04},
        {'Player': 'Derek Carr', 'Team': 'NO', 'Games': 2, 'Attempts': 60, 'Completions': 39, 
         'Pass_Yards': 441, 'Pass_TDs': 3, 'Interceptions': 1, 'YPA': 7.4, 
         'Pass_Yards_Per_Game': 220.5, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.650,
         'ADOT': 7.7, 'EPA_Per_Play': 0.08, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.10,
         'Historical_YPG': 208.5, 'Current_Form_Factor': 1.06},
        {'Player': 'Bryce Young', 'Team': 'CAR', 'Games': 2, 'Attempts': 68, 'Completions': 39, 
         'Pass_Yards': 408, 'Pass_TDs': 1, 'Interceptions': 2, 'YPA': 6.0, 
         'Pass_Yards_Per_Game': 204.0, 'Pass_TDs_Per_Game': 0.5, 'Completion_Pct': 0.574,
         'ADOT': 7.6, 'EPA_Per_Play': -0.10, 'Red_Zone_TDs': 0, 'Deep_Ball_Pct': 0.11,
         'Historical_YPG': 183.6, 'Current_Form_Factor': 1.11},
        
        # NFC West - Updated with current season performance
        {'Player': 'Brock Purdy', 'Team': 'SF', 'Games': 2, 'Attempts': 62, 'Completions': 43, 
         'Pass_Yards': 538, 'Pass_TDs': 3, 'Interceptions': 2, 'YPA': 8.7, 
         'Pass_Yards_Per_Game': 269.0, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.694,
         'ADOT': 8.2, 'EPA_Per_Play': 0.10, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.11,
         'Historical_YPG': 227.3, 'Current_Form_Factor': 1.18},
        {'Player': 'Kyler Murray', 'Team': 'ARI', 'Games': 2, 'Attempts': 70, 'Completions': 45, 
         'Pass_Yards': 581, 'Pass_TDs': 4, 'Interceptions': 2, 'YPA': 8.3, 
         'Pass_Yards_Per_Game': 290.5, 'Pass_TDs_Per_Game': 2.0, 'Completion_Pct': 0.643,
         'ADOT': 8.5, 'EPA_Per_Play': 0.09, 'Red_Zone_TDs': 3, 'Deep_Ball_Pct': 0.12,
         'Historical_YPG': 267.3, 'Current_Form_Factor': 1.09},
        {'Player': 'Matthew Stafford', 'Team': 'LAR', 'Games': 2, 'Attempts': 63, 'Completions': 39, 
         'Pass_Yards': 507, 'Pass_TDs': 3, 'Interceptions': 1, 'YPA': 8.0, 
         'Pass_Yards_Per_Game': 253.5, 'Pass_TDs_Per_Game': 1.5, 'Completion_Pct': 0.619,
         'ADOT': 9.0, 'EPA_Per_Play': 0.07, 'Red_Zone_TDs': 2, 'Deep_Ball_Pct': 0.14,
         'Historical_YPG': 221.3, 'Current_Form_Factor': 1.15},
        {'Player': 'Geno Smith', 'Team': 'SEA', 'Games': 2, 'Attempts': 72, 'Completions': 44, 
         'Pass_Yards': 453, 'Pass_TDs': 2, 'Interceptions': 3, 'YPA': 6.3, 
         'Pass_Yards_Per_Game': 226.5, 'Pass_TDs_Per_Game': 1.0, 'Completion_Pct': 0.611,
         'ADOT': 7.4, 'EPA_Per_Play': 0.02, 'Red_Zone_TDs': 1, 'Deep_Ball_Pct': 0.09,
         'Historical_YPG': 213.2, 'Current_Form_Factor': 1.06},
    ]
    
    rb_data = [
        # Top Tier RBs with 2025 team updates
        {'Player': 'Christian McCaffrey', 'Team': 'SF', 'Games': 2, 'Rush_Attempts': 38, 'Rush_Yards': 184, 
         'Rush_TDs': 2, 'YPC': 4.8, 'Rush_Yards_Per_Game': 92.0, 'Rush_TDs_Per_Game': 1.0, 
         'Rush_Attempts_Per_Game': 19.0, 'Rec_Yards': 89, 'Rec_TDs': 1, 'Receptions': 12, 
         'Rec_Yards_Per_Game': 44.5, 'Rec_TDs_Per_Game': 0.5, 'Target_Share': 0.19, 'Red_Zone_Touches': 6,
         'Snap_Count_Pct': 0.85, 'Goal_Line_Carries': 3, 'Broken_Tackles': 8, 'Yards_After_Contact': 3.4,
         'Historical_Rush_YPG': 91.2, 'Current_Form_Factor': 1.01},
        {'Player': 'Josh Jacobs', 'Team': 'GB', 'Games': 2, 'Rush_Attempts': 42, 'Rush_Yards': 201, 
         'Rush_TDs': 1, 'YPC': 4.8, 'Rush_Yards_Per_Game': 100.5, 'Rush_TDs_Per_Game': 0.5, 
         'Rush_Attempts_Per_Game': 21.0, 'Rec_Yards': 34, 'Rec_TDs': 0, 'Receptions': 4, 
         'Rec_Yards_Per_Game': 17.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.08, 'Red_Zone_Touches': 3,
         'Snap_Count_Pct': 0.78, 'Goal_Line_Carries': 2, 'Broken_Tackles': 7, 'Yards_After_Contact': 2.9,
         'Historical_Rush_YPG': 97.2, 'Current_Form_Factor': 1.03},  # Good fit in GB system
        {'Player': 'Saquon Barkley', 'Team': 'PHI', 'Games': 2, 'Rush_Attempts': 35, 'Rush_Yards': 198, 
         'Rush_TDs': 2, 'YPC': 5.7, 'Rush_Yards_Per_Game': 99.0, 'Rush_TDs_Per_Game': 1.0, 
         'Rush_Attempts_Per_Game': 17.5, 'Rec_Yards': 52, 'Rec_TDs': 0, 'Receptions': 6, 
         'Rec_Yards_Per_Game': 26.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.12, 'Red_Zone_Touches': 4,
         'Snap_Count_Pct': 0.82, 'Goal_Line_Carries': 3, 'Broken_Tackles': 9, 'Yards_After_Contact': 3.8,
         'Historical_Rush_YPG': 82.0, 'Current_Form_Factor': 1.21},  # Thriving in Eagles offense
        {'Player': 'Derrick Henry', 'Team': 'BAL', 'Games': 2, 'Rush_Attempts': 41, 'Rush_Yards': 189, 
         'Rush_TDs': 3, 'YPC': 4.6, 'Rush_Yards_Per_Game': 94.5, 'Rush_TDs_Per_Game': 1.5, 
         'Rush_Attempts_Per_Game': 20.5, 'Rec_Yards': 18, 'Rec_TDs': 0, 'Receptions': 3, 
         'Rec_Yards_Per_Game': 9.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.04, 'Red_Zone_Touches': 5,
         'Snap_Count_Pct': 0.71, 'Goal_Line_Carries': 4, 'Broken_Tackles': 6, 'Yards_After_Contact': 3.1,
         'Historical_Rush_YPG': 96.1, 'Current_Form_Factor': 0.98},
        
        # Second Tier RBs with 2025 team updates
        {'Player': 'Alvin Kamara', 'Team': 'NO', 'Games': 2, 'Rush_Attempts': 28, 'Rush_Yards': 105, 
         'Rush_TDs': 1, 'YPC': 3.8, 'Rush_Yards_Per_Game': 52.5, 'Rush_TDs_Per_Game': 0.5, 
         'Rush_Attempts_Per_Game': 14.0, 'Rec_Yards': 75, 'Rec_TDs': 0, 'Receptions': 12, 
         'Rec_Yards_Per_Game': 37.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.23, 'Red_Zone_Touches': 2,
         'Snap_Count_Pct': 0.87, 'Goal_Line_Carries': 0, 'Broken_Tackles': 3, 'Yards_After_Contact': 2.5,
         'Historical_Rush_YPG': 53.4, 'Current_Form_Factor': 0.98},
        {'Player': 'Austin Ekeler', 'Team': 'WAS', 'Games': 2, 'Rush_Attempts': 27, 'Rush_Yards': 91, 
         'Rush_TDs': 0, 'YPC': 3.4, 'Rush_Yards_Per_Game': 45.5, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 13.5, 'Rec_Yards': 82, 'Rec_TDs': 1, 'Receptions': 13, 
         'Rec_Yards_Per_Game': 41.0, 'Rec_TDs_Per_Game': 0.5, 'Target_Share': 0.21, 'Red_Zone_Touches': 3,
         'Snap_Count_Pct': 0.74, 'Goal_Line_Carries': 0, 'Broken_Tackles': 2, 'Yards_After_Contact': 2.3,
         'Historical_Rush_YPG': 36.9, 'Current_Form_Factor': 1.23},  # New team boost
        {'Player': 'Kenneth Walker III', 'Team': 'SEA', 'Games': 2, 'Rush_Attempts': 33, 'Rush_Yards': 135, 
         'Rush_TDs': 1, 'YPC': 4.1, 'Rush_Yards_Per_Game': 67.5, 'Rush_TDs_Per_Game': 0.5, 
         'Rush_Attempts_Per_Game': 16.5, 'Rec_Yards': 42, 'Rec_TDs': 0, 'Receptions': 5, 
         'Rec_Yards_Per_Game': 21.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.09, 'Red_Zone_Touches': 3,
         'Snap_Count_Pct': 0.67, 'Goal_Line_Carries': 1, 'Broken_Tackles': 5, 'Yards_After_Contact': 2.7,
         'Historical_Rush_YPG': 60.3, 'Current_Form_Factor': 1.12},
        {'Player': 'Bijan Robinson', 'Team': 'ATL', 'Games': 2, 'Rush_Attempts': 32, 'Rush_Yards': 148, 
         'Rush_TDs': 1, 'YPC': 4.6, 'Rush_Yards_Per_Game': 74.0, 'Rush_TDs_Per_Game': 0.5, 
         'Rush_Attempts_Per_Game': 16.0, 'Rec_Yards': 78, 'Rec_TDs': 1, 'Receptions': 9, 
         'Rec_Yards_Per_Game': 39.0, 'Rec_TDs_Per_Game': 0.5, 'Target_Share': 0.17, 'Red_Zone_Touches': 4,
         'Snap_Count_Pct': 0.73, 'Goal_Line_Carries': 1, 'Broken_Tackles': 6, 'Yards_After_Contact': 2.9,
         'Historical_Rush_YPG': 57.4, 'Current_Form_Factor': 1.29},
        
        # Additional Starting RBs with 2025 updates
        {'Player': 'Jahmyr Gibbs', 'Team': 'DET', 'Games': 2, 'Rush_Attempts': 29, 'Rush_Yards': 157, 
         'Rush_TDs': 2, 'YPC': 5.4, 'Rush_Yards_Per_Game': 78.5, 'Rush_TDs_Per_Game': 1.0, 
         'Rush_Attempts_Per_Game': 14.5, 'Rec_Yards': 51, 'Rec_TDs': 0, 'Receptions': 8, 
         'Rec_Yards_Per_Game': 25.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.11, 'Red_Zone_Touches': 4,
         'Snap_Count_Pct': 0.60, 'Goal_Line_Carries': 2, 'Broken_Tackles': 4, 'Yards_After_Contact': 3.5,
         'Historical_Rush_YPG': 55.6, 'Current_Form_Factor': 1.41},  # Exceeding expectations
        {'Player': 'Joe Mixon', 'Team': 'HOU', 'Games': 2, 'Rush_Attempts': 39, 'Rush_Yards': 162, 
         'Rush_TDs': 2, 'YPC': 4.2, 'Rush_Yards_Per_Game': 81.0, 'Rush_TDs_Per_Game': 1.0, 
         'Rush_Attempts_Per_Game': 19.5, 'Rec_Yards': 23, 'Rec_TDs': 0, 'Receptions': 4, 
         'Rec_Yards_Per_Game': 11.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.05, 'Red_Zone_Touches': 4,
         'Snap_Count_Pct': 0.71, 'Goal_Line_Carries': 1, 'Broken_Tackles': 3, 'Yards_After_Contact': 2.6,
         'Historical_Rush_YPG': 61.2, 'Current_Form_Factor': 1.32},
        {'Player': 'Jonathan Taylor', 'Team': 'IND', 'Games': 2, 'Rush_Attempts': 40, 'Rush_Yards': 178, 
         'Rush_TDs': 1, 'YPC': 4.5, 'Rush_Yards_Per_Game': 89.0, 'Rush_TDs_Per_Game': 0.5, 
         'Rush_Attempts_Per_Game': 20.0, 'Rec_Yards': 48, 'Rec_TDs': 0, 'Receptions': 7, 
         'Rec_Yards_Per_Game': 24.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.13, 'Red_Zone_Touches': 3,
         'Snap_Count_Pct': 0.75, 'Goal_Line_Carries': 1, 'Broken_Tackles': 5, 'Yards_After_Contact': 2.8,
         'Historical_Rush_YPG': 65.3, 'Current_Form_Factor': 1.36},
        {'Player': 'De\'Von Achane', 'Team': 'MIA', 'Games': 2, 'Rush_Attempts': 34, 'Rush_Yards': 195, 
         'Rush_TDs': 2, 'YPC': 5.7, 'Rush_Yards_Per_Game': 97.5, 'Rush_TDs_Per_Game': 1.0, 
         'Rush_Attempts_Per_Game': 17.0, 'Rec_Yards': 32, 'Rec_TDs': 0, 'Receptions': 5, 
         'Rec_Yards_Per_Game': 16.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.06, 'Red_Zone_Touches': 3,
         'Snap_Count_Pct': 0.63, 'Goal_Line_Carries': 1, 'Broken_Tackles': 7, 'Yards_After_Contact': 3.7,
         'Historical_Rush_YPG': 67.2, 'Current_Form_Factor': 1.45},
        
        # More RBs to reach 24 total with 2025 updates
        {'Player': 'James Cook', 'Team': 'BUF', 'Games': 2, 'Rush_Attempts': 36, 'Rush_Yards': 154, 
         'Rush_TDs': 0, 'YPC': 4.3, 'Rush_Yards_Per_Game': 77.0, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 18.0, 'Rec_Yards': 72, 'Rec_TDs': 0, 'Receptions': 7, 
         'Rec_Yards_Per_Game': 36.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.10, 'Red_Zone_Touches': 5,
         'Snap_Count_Pct': 0.66, 'Goal_Line_Carries': 2, 'Broken_Tackles': 4, 'Yards_After_Contact': 2.7,
         'Historical_Rush_YPG': 59.4, 'Current_Form_Factor': 1.30},
        {'Player': 'Kyren Williams', 'Team': 'LAR', 'Games': 2, 'Rush_Attempts': 31, 'Rush_Yards': 139, 
         'Rush_TDs': 0, 'YPC': 4.5, 'Rush_Yards_Per_Game': 69.5, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 15.5, 'Rec_Yards': 33, 'Rec_TDs': 0, 'Receptions': 5, 
         'Rec_Yards_Per_Game': 16.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.09, 'Red_Zone_Touches': 3,
         'Snap_Count_Pct': 0.69, 'Goal_Line_Carries': 1, 'Broken_Tackles': 3, 'Yards_After_Contact': 2.8,
         'Historical_Rush_YPG': 54.2, 'Current_Form_Factor': 1.28},
        {'Player': 'Rachaad White', 'Team': 'TB', 'Games': 2, 'Rush_Attempts': 30, 'Rush_Yards': 120, 
         'Rush_TDs': 0, 'YPC': 4.0, 'Rush_Yards_Per_Game': 60.0, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 15.0, 'Rec_Yards': 89, 'Rec_TDs': 0, 'Receptions': 14, 
         'Rec_Yards_Per_Game': 44.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.18, 'Red_Zone_Touches': 2,
         'Snap_Count_Pct': 0.78, 'Goal_Line_Carries': 0, 'Broken_Tackles': 2, 'Yards_After_Contact': 2.4,
         'Historical_Rush_YPG': 58.2, 'Current_Form_Factor': 1.03},
        {'Player': 'Breece Hall', 'Team': 'NYJ', 'Games': 2, 'Rush_Attempts': 28, 'Rush_Yards': 111, 
         'Rush_TDs': 0, 'YPC': 4.0, 'Rush_Yards_Per_Game': 55.5, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 14.0, 'Rec_Yards': 95, 'Rec_TDs': 0, 'Receptions': 12, 
         'Rec_Yards_Per_Game': 47.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.16, 'Red_Zone_Touches': 2,
         'Snap_Count_Pct': 0.73, 'Goal_Line_Carries': 0, 'Broken_Tackles': 3, 'Yards_After_Contact': 2.3,
         'Historical_Rush_YPG': 40.9, 'Current_Form_Factor': 1.36},
        {'Player': 'Najee Harris', 'Team': 'PIT', 'Games': 2, 'Rush_Attempts': 38, 'Rush_Yards': 155, 
         'Rush_TDs': 0, 'YPC': 4.1, 'Rush_Yards_Per_Game': 77.5, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 19.0, 'Rec_Yards': 28, 'Rec_TDs': 0, 'Receptions': 5, 
         'Rec_Yards_Per_Game': 14.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.08, 'Red_Zone_Touches': 1,
         'Snap_Count_Pct': 0.74, 'Goal_Line_Carries': 0, 'Broken_Tackles': 4, 'Yards_After_Contact': 2.5,
         'Historical_Rush_YPG': 60.9, 'Current_Form_Factor': 1.27},
        {'Player': 'Tony Pollard', 'Team': 'TEN', 'Games': 2, 'Rush_Attempts': 30, 'Rush_Yards': 128, 
         'Rush_TDs': 0, 'YPC': 4.3, 'Rush_Yards_Per_Game': 64.0, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 15.0, 'Rec_Yards': 50, 'Rec_TDs': 0, 'Receptions': 8, 
         'Rec_Yards_Per_Game': 25.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.11, 'Red_Zone_Touches': 1,
         'Snap_Count_Pct': 0.70, 'Goal_Line_Carries': 0, 'Broken_Tackles': 3, 'Yards_After_Contact': 2.6,
         'Historical_Rush_YPG': 50.2, 'Current_Form_Factor': 1.27},
        {'Player': 'Rhamondre Stevenson', 'Team': 'NE', 'Games': 2, 'Rush_Attempts': 23, 'Rush_Yards': 92, 
         'Rush_TDs': 0, 'YPC': 4.0, 'Rush_Yards_Per_Game': 46.0, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 11.5, 'Rec_Yards': 38, 'Rec_TDs': 0, 'Receptions': 6, 
         'Rec_Yards_Per_Game': 19.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.10, 'Red_Zone_Touches': 2,
         'Snap_Count_Pct': 0.61, 'Goal_Line_Carries': 0, 'Broken_Tackles': 1, 'Yards_After_Contact': 2.2,
         'Historical_Rush_YPG': 36.4, 'Current_Form_Factor': 1.26},
        {'Player': 'Nick Chubb', 'Team': 'CLE', 'Games': 1, 'Rush_Attempts': 13, 'Rush_Yards': 50, 
         'Rush_TDs': 0, 'YPC': 3.8, 'Rush_Yards_Per_Game': 50.0, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 13.0, 'Rec_Yards': 2, 'Rec_TDs': 0, 'Receptions': 1, 
         'Rec_Yards_Per_Game': 2.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.02, 'Red_Zone_Touches': 1,
         'Snap_Count_Pct': 0.54, 'Goal_Line_Carries': 0, 'Broken_Tackles': 0, 'Yards_After_Contact': 2.7,
         'Historical_Rush_YPG': 41.4, 'Current_Form_Factor': 1.21},
        {'Player': 'Alexander Mattison', 'Team': 'LV', 'Games': 2, 'Rush_Attempts': 27, 'Rush_Yards': 109, 
         'Rush_TDs': 0, 'YPC': 4.0, 'Rush_Yards_Per_Game': 54.5, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 13.5, 'Rec_Yards': 27, 'Rec_TDs': 0, 'Receptions': 4, 
         'Rec_Yards_Per_Game': 13.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.07, 'Red_Zone_Touches': 1,
         'Snap_Count_Pct': 0.63, 'Goal_Line_Carries': 0, 'Broken_Tackles': 2, 'Yards_After_Contact': 2.3,
         'Historical_Rush_YPG': 42.5, 'Current_Form_Factor': 1.28},
        {'Player': 'James Conner', 'Team': 'ARI', 'Games': 2, 'Rush_Attempts': 33, 'Rush_Yards': 156, 
         'Rush_TDs': 0, 'YPC': 4.7, 'Rush_Yards_Per_Game': 78.0, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 16.5, 'Rec_Yards': 25, 'Rec_TDs': 0, 'Receptions': 4, 
         'Rec_Yards_Per_Game': 12.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.06, 'Red_Zone_Touches': 2,
         'Snap_Count_Pct': 0.68, 'Goal_Line_Carries': 0, 'Broken_Tackles': 3, 'Yards_After_Contact': 3.0,
         'Historical_Rush_YPG': 61.2, 'Current_Form_Factor': 1.27},
        {'Player': 'Javonte Williams', 'Team': 'DEN', 'Games': 2, 'Rush_Attempts': 31, 'Rush_Yards': 127, 
         'Rush_TDs': 0, 'YPC': 4.1, 'Rush_Yards_Per_Game': 63.5, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 15.5, 'Rec_Yards': 34, 'Rec_TDs': 0, 'Receptions': 6, 
         'Rec_Yards_Per_Game': 17.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.09, 'Red_Zone_Touches': 1,
         'Snap_Count_Pct': 0.65, 'Goal_Line_Carries': 0, 'Broken_Tackles': 2, 'Yards_After_Contact': 2.5,
         'Historical_Rush_YPG': 49.9, 'Current_Form_Factor': 1.27},
        {'Player': 'Chuba Hubbard', 'Team': 'CAR', 'Games': 2, 'Rush_Attempts': 33, 'Rush_Yards': 135, 
         'Rush_TDs': 0, 'YPC': 4.1, 'Rush_Yards_Per_Game': 67.5, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 16.5, 'Rec_Yards': 18, 'Rec_TDs': 0, 'Receptions': 3, 
         'Rec_Yards_Per_Game': 9.0, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.05, 'Red_Zone_Touches': 2,
         'Snap_Count_Pct': 0.66, 'Goal_Line_Carries': 0, 'Broken_Tackles': 3, 'Yards_After_Contact': 2.4,
         'Historical_Rush_YPG': 53.1, 'Current_Form_Factor': 1.27},
        {'Player': 'David Montgomery', 'Team': 'DET', 'Games': 2, 'Rush_Attempts': 28, 'Rush_Yards': 116, 
         'Rush_TDs': 0, 'YPC': 4.1, 'Rush_Yards_Per_Game': 58.0, 'Rush_TDs_Per_Game': 0.0, 
         'Rush_Attempts_Per_Game': 14.0, 'Rec_Yards': 55, 'Rec_TDs': 0, 'Receptions': 7, 
         'Rec_Yards_Per_Game': 27.5, 'Rec_TDs_Per_Game': 0.0, 'Target_Share': 0.09, 'Red_Zone_Touches': 4,
         'Snap_Count_Pct': 0.60, 'Goal_Line_Carries': 0, 'Broken_Tackles': 2, 'Yards_After_Contact': 2.6,
         'Historical_Rush_YPG': 45.6, 'Current_Form_Factor': 1.27},
    ]
    
    # Updated WR data with Week 1-2 2025 performance and team changes
    wr_data = [
        # Elite Tier WRs with 2025 team updates
        {'Player': 'Tyreek Hill', 'Team': 'MIA', 'Games': 2, 'Receptions': 19, 'Rec_Yards': 282, 
         'Rec_TDs': 1, 'YPR': 14.8, 'Targets': 27, 'Rec_Yards_Per_Game': 141.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 9.5, 'Catch_Rate': 0.70,
         'Target_Share': 0.29, 'Air_Yards': 324, 'ADOT': 12.0, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.47,
         'Historical_Rec_YPG': 100.6, 'Current_Form_Factor': 1.40},  # Dominant start
        {'Player': 'Stefon Diggs', 'Team': 'HOU', 'Games': 2, 'Receptions': 17, 'Rec_Yards': 229, 
         'Rec_TDs': 2, 'YPR': 13.5, 'Targets': 24, 'Rec_Yards_Per_Game': 114.5, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 8.5, 'Catch_Rate': 0.71,
         'Target_Share': 0.26, 'Air_Yards': 260, 'ADOT': 11.2, 'Red_Zone_Targets': 3, 'Slot_Rate': 0.54,
         'Historical_Rec_YPG': 84.1, 'Current_Form_Factor': 1.36},  # Thriving in Houston
        {'Player': 'Davante Adams', 'Team': 'NYJ', 'Games': 2, 'Receptions': 16, 'Rec_Yards': 243, 
         'Rec_TDs': 3, 'YPR': 15.2, 'Targets': 29, 'Rec_Yards_Per_Game': 121.5, 
         'Rec_TDs_Per_Game': 1.5, 'Receptions_Per_Game': 8.0, 'Catch_Rate': 0.55,
         'Target_Share': 0.33, 'Air_Yards': 342, 'ADOT': 11.9, 'Red_Zone_Targets': 4, 'Slot_Rate': 0.40,
         'Historical_Rec_YPG': 89.2, 'Current_Form_Factor': 1.36},
        {'Player': 'CeeDee Lamb', 'Team': 'DAL', 'Games': 2, 'Receptions': 22, 'Rec_Yards': 297, 
         'Rec_TDs': 2, 'YPR': 13.5, 'Targets': 30, 'Rec_Yards_Per_Game': 148.5, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 11.0, 'Catch_Rate': 0.73,
         'Target_Share': 0.32, 'Air_Yards': 330, 'ADOT': 11.1, 'Red_Zone_Targets': 3, 'Slot_Rate': 0.50,
         'Historical_Rec_YPG': 102.9, 'Current_Form_Factor': 1.44},  # Dominating targets
        {'Player': 'A.J. Brown', 'Team': 'PHI', 'Games': 2, 'Receptions': 17, 'Rec_Yards': 233, 
         'Rec_TDs': 1, 'YPR': 13.7, 'Targets': 25, 'Rec_Yards_Per_Game': 116.5, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 8.5, 'Catch_Rate': 0.68,
         'Target_Share': 0.27, 'Air_Yards': 275, 'ADOT': 11.0, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.37,
         'Historical_Rec_YPG': 85.6, 'Current_Form_Factor': 1.36},
        {'Player': 'Ja\'Marr Chase', 'Team': 'CIN', 'Games': 2, 'Receptions': 16, 'Rec_Yards': 195, 
         'Rec_TDs': 1, 'YPR': 12.2, 'Targets': 23, 'Rec_Yards_Per_Game': 97.5, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 8.0, 'Catch_Rate': 0.70,
         'Target_Share': 0.29, 'Air_Yards': 253, 'ADOT': 11.1, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.44,
         'Historical_Rec_YPG': 71.5, 'Current_Form_Factor': 1.36},
        {'Player': 'Amon-Ra St. Brown', 'Team': 'DET', 'Games': 2, 'Receptions': 19, 'Rec_Yards': 244, 
         'Rec_TDs': 2, 'YPR': 12.8, 'Targets': 26, 'Rec_Yards_Per_Game': 122.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 9.5, 'Catch_Rate': 0.73,
         'Target_Share': 0.30, 'Air_Yards': 213, 'ADOT': 8.9, 'Red_Zone_Targets': 3, 'Slot_Rate': 0.70,
         'Historical_Rec_YPG': 94.7, 'Current_Form_Factor': 1.29},
        {'Player': 'Cooper Kupp', 'Team': 'LAR', 'Games': 1, 'Receptions': 11, 'Rec_Yards': 118, 
         'Rec_TDs': 1, 'YPR': 10.7, 'Targets': 15, 'Rec_Yards_Per_Game': 118.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 11.0, 'Catch_Rate': 0.73,
         'Target_Share': 0.22, 'Air_Yards': 129, 'ADOT': 8.9, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.74,
         'Historical_Rec_YPG': 59.2, 'Current_Form_Factor': 2.00},  # Returning strong
        
        # Second Tier WRs with 2025 updates
        {'Player': 'DeVonta Smith', 'Team': 'PHI', 'Games': 2, 'Receptions': 11, 'Rec_Yards': 172, 
         'Rec_TDs': 1, 'YPR': 15.6, 'Targets': 19, 'Rec_Yards_Per_Game': 86.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 5.5, 'Catch_Rate': 0.58,
         'Target_Share': 0.21, 'Air_Yards': 217, 'ADOT': 11.6, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.30,
         'Historical_Rec_YPG': 62.7, 'Current_Form_Factor': 1.37},
        {'Player': 'Puka Nacua', 'Team': 'LAR', 'Games': 2, 'Receptions': 13, 'Rec_Yards': 214, 
         'Rec_TDs': 1, 'YPR': 16.5, 'Targets': 25, 'Rec_Yards_Per_Game': 107.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 6.5, 'Catch_Rate': 0.52,
         'Target_Share': 0.34, 'Air_Yards': 336, 'ADOT': 13.0, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.43,
         'Historical_Rec_YPG': 87.4, 'Current_Form_Factor': 1.22},
        {'Player': 'Nico Collins', 'Team': 'HOU', 'Games': 2, 'Receptions': 11, 'Rec_Yards': 186, 
         'Rec_TDs': 1, 'YPR': 16.9, 'Targets': 18, 'Rec_Yards_Per_Game': 93.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 5.5, 'Catch_Rate': 0.61,
         'Target_Share': 0.20, 'Air_Yards': 247, 'ADOT': 13.8, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.24,
         'Historical_Rec_YPG': 86.5, 'Current_Form_Factor': 1.07},
        {'Player': 'DK Metcalf', 'Team': 'SEA', 'Games': 2, 'Receptions': 15, 'Rec_Yards': 202, 
         'Rec_TDs': 0, 'YPR': 13.5, 'Targets': 21, 'Rec_Yards_Per_Game': 101.0, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 7.5, 'Catch_Rate': 0.71,
         'Target_Share': 0.24, 'Air_Yards': 304, 'ADOT': 11.5, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.17,
         'Historical_Rec_YPG': 73.8, 'Current_Form_Factor': 1.37},
        {'Player': 'Mike Evans', 'Team': 'TB', 'Games': 2, 'Receptions': 13, 'Rec_Yards': 202, 
         'Rec_TDs': 2, 'YPR': 15.5, 'Targets': 20, 'Rec_Yards_Per_Game': 101.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 6.5, 'Catch_Rate': 0.65,
         'Target_Share': 0.24, 'Air_Yards': 250, 'ADOT': 12.6, 'Red_Zone_Targets': 3, 'Slot_Rate': 0.14,
         'Historical_Rec_YPG': 96.5, 'Current_Form_Factor': 1.05},
        {'Player': 'Chris Godwin', 'Team': 'TB', 'Games': 1, 'Receptions': 8, 'Rec_Yards': 92, 
         'Rec_TDs': 1, 'YPR': 11.5, 'Targets': 11, 'Rec_Yards_Per_Game': 92.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 8.0, 'Catch_Rate': 0.73,
         'Target_Share': 0.21, 'Air_Yards': 99, 'ADOT': 9.2, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.60,
         'Historical_Rec_YPG': 82.3, 'Current_Form_Factor': 1.12},
        {'Player': 'Calvin Ridley', 'Team': 'TEN', 'Games': 2, 'Receptions': 12, 'Rec_Yards': 163, 
         'Rec_TDs': 1, 'YPR': 13.6, 'Targets': 22, 'Rec_Yards_Per_Game': 81.5, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 6.0, 'Catch_Rate': 0.55,
         'Target_Share': 0.25, 'Air_Yards': 255, 'ADOT': 10.6, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.43,
         'Historical_Rec_YPG': 59.8, 'Current_Form_Factor': 1.36},
        {'Player': 'Jaylen Waddle', 'Team': 'MIA', 'Games': 2, 'Receptions': 11, 'Rec_Yards': 162, 
         'Rec_TDs': 1, 'YPR': 14.7, 'Targets': 16, 'Rec_Yards_Per_Game': 81.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 5.5, 'Catch_Rate': 0.69,
         'Target_Share': 0.18, 'Air_Yards': 190, 'ADOT': 11.4, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.64,
         'Historical_Rec_YPG': 63.4, 'Current_Form_Factor': 1.28},
        
        # Additional Starting WRs with 2025 updates
        {'Player': 'Garrett Wilson', 'Team': 'NYJ', 'Games': 2, 'Receptions': 15, 'Rec_Yards': 178, 
         'Rec_TDs': 1, 'YPR': 11.9, 'Targets': 24, 'Rec_Yards_Per_Game': 89.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 7.5, 'Catch_Rate': 0.63,
         'Target_Share': 0.26, 'Air_Yards': 211, 'ADOT': 9.6, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.46,
         'Historical_Rec_YPG': 64.9, 'Current_Form_Factor': 1.37},
        {'Player': 'Terry McLaurin', 'Team': 'WAS', 'Games': 2, 'Receptions': 13, 'Rec_Yards': 176, 
         'Rec_TDs': 2, 'YPR': 13.5, 'Targets': 20, 'Rec_Yards_Per_Game': 88.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 6.5, 'Catch_Rate': 0.65,
         'Target_Share': 0.31, 'Air_Yards': 284, 'ADOT': 11.7, 'Red_Zone_Targets': 3, 'Slot_Rate': 0.33,
         'Historical_Rec_YPG': 64.5, 'Current_Form_Factor': 1.37},
        {'Player': 'Tee Higgins', 'Team': 'CIN', 'Games': 1, 'Receptions': 12, 'Rec_Yards': 152, 
         'Rec_TDs': 1, 'YPR': 12.7, 'Targets': 18, 'Rec_Yards_Per_Game': 152.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 12.0, 'Catch_Rate': 0.67,
         'Target_Share': 0.22, 'Air_Yards': 189, 'ADOT': 10.6, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.27,
         'Historical_Rec_YPG': 75.9, 'Current_Form_Factor': 2.00},  # Strong performance
        {'Player': 'Marvin Harrison Jr.', 'Team': 'ARI', 'Games': 2, 'Receptions': 8, 'Rec_Yards': 128, 
         'Rec_TDs': 0, 'YPR': 16.0, 'Targets': 13, 'Rec_Yards_Per_Game': 64.0, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 4.0, 'Catch_Rate': 0.62,
         'Target_Share': 0.19, 'Air_Yards': 198, 'ADOT': 12.5, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.24,
         'Historical_Rec_YPG': 47.0, 'Current_Form_Factor': 1.36},
        {'Player': 'Brian Thomas Jr.', 'Team': 'JAX', 'Games': 2, 'Receptions': 14, 'Rec_Yards': 205, 
         'Rec_TDs': 1, 'YPR': 14.6, 'Targets': 21, 'Rec_Yards_Per_Game': 102.5, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 7.0, 'Catch_Rate': 0.67,
         'Target_Share': 0.24, 'Air_Yards': 308, 'ADOT': 12.2, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.20,
         'Historical_Rec_YPG': 75.4, 'Current_Form_Factor': 1.36},
        {'Player': 'Malik Nabers', 'Team': 'NYG', 'Games': 1, 'Receptions': 18, 'Rec_Yards': 193, 
         'Rec_TDs': 1, 'YPR': 10.7, 'Targets': 29, 'Rec_Yards_Per_Game': 193.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 18.0, 'Catch_Rate': 0.62,
         'Target_Share': 0.32, 'Air_Yards': 306, 'ADOT': 8.2, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.57,
         'Historical_Rec_YPG': 92.6, 'Current_Form_Factor': 2.00},  # Dominant performance
        {'Player': 'Drake London', 'Team': 'ATL', 'Games': 2, 'Receptions': 14, 'Rec_Yards': 171, 
         'Rec_TDs': 1, 'YPR': 12.2, 'Targets': 21, 'Rec_Yards_Per_Game': 85.5, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 7.0, 'Catch_Rate': 0.67,
         'Target_Share': 0.26, 'Air_Yards': 205, 'ADOT': 10.4, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.40,
         'Historical_Rec_YPG': 62.6, 'Current_Form_Factor': 1.36},
        {'Player': 'Courtland Sutton', 'Team': 'DEN', 'Games': 2, 'Receptions': 11, 'Rec_Yards': 170, 
         'Rec_TDs': 1, 'YPR': 15.5, 'Targets': 17, 'Rec_Yards_Per_Game': 85.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 5.5, 'Catch_Rate': 0.65,
         'Target_Share': 0.24, 'Air_Yards': 272, 'ADOT': 12.6, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.17,
         'Historical_Rec_YPG': 62.6, 'Current_Form_Factor': 1.36},
        {'Player': 'Amari Cooper', 'Team': 'BUF', 'Games': 2, 'Receptions': 11, 'Rec_Yards': 179, 
         'Rec_TDs': 0, 'YPR': 16.3, 'Targets': 16, 'Rec_Yards_Per_Game': 89.5, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 5.5, 'Catch_Rate': 0.69,
         'Target_Share': 0.19, 'Air_Yards': 286, 'ADOT': 14.9, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.27,
         'Historical_Rec_YPG': 73.5, 'Current_Form_Factor': 1.22},
    ]
    
    te_data = [
        {'Player': 'Travis Kelce', 'Team': 'KC', 'Games': 2, 'Receptions': 18, 'Rec_Yards': 270, 
         'Rec_TDs': 2, 'YPR': 15.0, 'Targets': 25, 'Rec_Yards_Per_Game': 135.0, 
         'Rec_TDs_Per_Game': 1.0, 'Receptions_Per_Game': 9.0, 'Catch_Rate': 0.72,
         'Target_Share': 0.26, 'Air_Yards': 256, 'ADOT': 8.7, 'Red_Zone_Targets': 4, 'Slot_Rate': 0.57,
         'Historical_Rec_YPG': 78.7, 'Current_Form_Factor': 1.71},
        {'Player': 'Mark Andrews', 'Team': 'BAL', 'Games': 2, 'Receptions': 14, 'Rec_Yards': 218, 
         'Rec_TDs': 1, 'YPR': 15.6, 'Targets': 20, 'Rec_Yards_Per_Game': 109.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 7.0, 'Catch_Rate': 0.70,
         'Target_Share': 0.28, 'Air_Yards': 232, 'ADOT': 11.8, 'Red_Zone_Targets': 3, 'Slot_Rate': 0.44,
         'Historical_Rec_YPG': 80.1, 'Current_Form_Factor': 1.36},
        {'Player': 'George Kittle', 'Team': 'SF', 'Games': 2, 'Receptions': 9, 'Rec_Yards': 115, 
         'Rec_TDs': 0, 'YPR': 12.8, 'Targets': 13, 'Rec_Yards_Per_Game': 57.5, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 4.5, 'Catch_Rate': 0.69,
         'Target_Share': 0.16, 'Air_Yards': 147, 'ADOT': 10.4, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.40,
         'Historical_Rec_YPG': 47.8, 'Current_Form_Factor': 1.20},
        {'Player': 'T.J. Hockenson', 'Team': 'MIN', 'Games': 2, 'Receptions': 15, 'Rec_Yards': 154, 
         'Rec_TDs': 1, 'YPR': 10.3, 'Targets': 21, 'Rec_Yards_Per_Game': 77.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 7.5, 'Catch_Rate': 0.71,
         'Target_Share': 0.24, 'Air_Yards': 168, 'ADOT': 8.5, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.50,
         'Historical_Rec_YPG': 64.0, 'Current_Form_Factor': 1.20},
        {'Player': 'Evan Engram', 'Team': 'JAX', 'Games': 2, 'Receptions': 18, 'Rec_Yards': 144, 
         'Rec_TDs': 0, 'YPR': 8.0, 'Targets': 23, 'Rec_Yards_Per_Game': 72.0, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 9.0, 'Catch_Rate': 0.78,
         'Target_Share': 0.26, 'Air_Yards': 157, 'ADOT': 7.1, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.64,
         'Historical_Rec_YPG': 56.6, 'Current_Form_Factor': 1.27},
        {'Player': 'Dallas Goedert', 'Team': 'PHI', 'Games': 2, 'Receptions': 9, 'Rec_Yards': 95, 
         'Rec_TDs': 0, 'YPR': 10.6, 'Targets': 14, 'Rec_Yards_Per_Game': 47.5, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 4.5, 'Catch_Rate': 0.64,
         'Target_Share': 0.16, 'Air_Yards': 114, 'ADOT': 8.3, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.46,
         'Historical_Rec_YPG': 37.0, 'Current_Form_Factor': 1.28},
        {'Player': 'Sam LaPorta', 'Team': 'DET', 'Games': 2, 'Receptions': 13, 'Rec_Yards': 142, 
         'Rec_TDs': 1, 'YPR': 10.9, 'Targets': 19, 'Rec_Yards_Per_Game': 71.0, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 6.5, 'Catch_Rate': 0.68,
         'Target_Share': 0.16, 'Air_Yards': 157, 'ADOT': 8.3, 'Red_Zone_Targets': 3, 'Slot_Rate': 0.54,
         'Historical_Rec_YPG': 52.3, 'Current_Form_Factor': 1.36},
        {'Player': 'Kyle Pitts', 'Team': 'ATL', 'Games': 2, 'Receptions': 9, 'Rec_Yards': 107, 
         'Rec_TDs': 0, 'YPR': 11.9, 'Targets': 15, 'Rec_Yards_Per_Game': 53.5, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 4.5, 'Catch_Rate': 0.60,
         'Target_Share': 0.19, 'Air_Yards': 131, 'ADOT': 9.2, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.37,
         'Historical_Rec_YPG': 39.2, 'Current_Form_Factor': 1.36},
        {'Player': 'David Njoku', 'Team': 'CLE', 'Games': 2, 'Receptions': 13, 'Rec_Yards': 141, 
         'Rec_TDs': 1, 'YPR': 10.8, 'Targets': 19, 'Rec_Yards_Per_Game': 70.5, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 6.5, 'Catch_Rate': 0.68,
         'Target_Share': 0.29, 'Air_Yards': 163, 'ADOT': 8.9, 'Red_Zone_Targets': 2, 'Slot_Rate': 0.43,
         'Historical_Rec_YPG': 55.1, 'Current_Form_Factor': 1.28},
        {'Player': 'Brock Bowers', 'Team': 'LV', 'Games': 2, 'Receptions': 18, 'Rec_Yards': 287, 
         'Rec_TDs': 1, 'YPR': 15.9, 'Targets': 26, 'Rec_Yards_Per_Game': 143.5, 
         'Rec_TDs_Per_Game': 0.5, 'Receptions_Per_Game': 9.0, 'Catch_Rate': 0.69,
         'Target_Share': 0.33, 'Air_Yards': 227, 'ADOT': 7.8, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.60,
         'Historical_Rec_YPG': 70.2, 'Current_Form_Factor': 2.00},  # Dominant performance
        {'Player': 'Cade Otton', 'Team': 'TB', 'Games': 2, 'Receptions': 8, 'Rec_Yards': 117, 
         'Rec_TDs': 0, 'YPR': 14.6, 'Targets': 11, 'Rec_Yards_Per_Game': 58.5, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 4.0, 'Catch_Rate': 0.73,
         'Target_Share': 0.15, 'Air_Yards': 131, 'ADOT': 11.1, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.48,
         'Historical_Rec_YPG': 43.2, 'Current_Form_Factor': 1.36},
        {'Player': 'Trey McBride', 'Team': 'ARI', 'Games': 2, 'Receptions': 15, 'Rec_Yards': 241, 
         'Rec_TDs': 0, 'YPR': 16.1, 'Targets': 21, 'Rec_Yards_Per_Game': 120.5, 
         'Rec_TDs_Per_Game': 0.0, 'Receptions_Per_Game': 7.5, 'Catch_Rate': 0.71,
         'Target_Share': 0.25, 'Air_Yards': 193, 'ADOT': 8.3, 'Red_Zone_Targets': 1, 'Slot_Rate': 0.51,
         'Historical_Rec_YPG': 60.2, 'Current_Form_Factor': 2.00},  # Dominant performance
    ]
    
    defense_data = [
        # AFC East
        {'Team': 'BUF', 'Games': 2, 'Pass_Yards_Allowed': 450, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 180, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 225.0, 'Rush_Yards_Allowed_Per_Game': 90.0,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 7, 'Rush_Defense_Rank': 3, 'Pass_TD_Defense_Rank': 3, 'Rush_TD_Defense_Rank': 3,
         'Pressure_Rate': 0.30, 'Blitz_Rate': 0.25, 'Coverage_Grade': 74.0, 'Red_Zone_TD_Rate': 0.50},
        {'Team': 'MIA', 'Games': 2, 'Pass_Yards_Allowed': 500, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 200, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 250.0, 'Rush_Yards_Allowed_Per_Game': 100.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 16, 'Rush_Defense_Rank': 6, 'Pass_TD_Defense_Rank': 10, 'Rush_TD_Defense_Rank': 6,
         'Pressure_Rate': 0.26, 'Blitz_Rate': 0.27, 'Coverage_Grade': 68.0, 'Red_Zone_TD_Rate': 0.53},
        {'Team': 'NYJ', 'Games': 2, 'Pass_Yards_Allowed': 430, 'Pass_TDs_Allowed': 1, 
         'Rush_Yards_Allowed': 170, 'Rush_TDs_Allowed': 0, 'Total_TDs_Allowed': 1,
         'Pass_Yards_Allowed_Per_Game': 215.0, 'Rush_Yards_Allowed_Per_Game': 85.0,
         'Pass_TDs_Allowed_Per_Game': 0.5, 'Rush_TDs_Allowed_Per_Game': 0.0,
         'Pass_Defense_Rank': 4, 'Rush_Defense_Rank': 1, 'Pass_TD_Defense_Rank': 1, 'Rush_TD_Defense_Rank': 1,
         'Pressure_Rate': 0.32, 'Blitz_Rate': 0.24, 'Coverage_Grade': 75.0, 'Red_Zone_TD_Rate': 0.47},
        {'Team': 'NE', 'Games': 2, 'Pass_Yards_Allowed': 550, 'Pass_TDs_Allowed': 4, 
         'Rush_Yards_Allowed': 250, 'Rush_TDs_Allowed': 3, 'Total_TDs_Allowed': 7,
         'Pass_Yards_Allowed_Per_Game': 275.0, 'Rush_Yards_Allowed_Per_Game': 125.0,
         'Pass_TDs_Allowed_Per_Game': 2.0, 'Rush_TDs_Allowed_Per_Game': 1.5,
         'Pass_Defense_Rank': 24, 'Rush_Defense_Rank': 25, 'Pass_TD_Defense_Rank': 24, 'Rush_TD_Defense_Rank': 25,
         'Pressure_Rate': 0.23, 'Blitz_Rate': 0.28, 'Coverage_Grade': 64.0, 'Red_Zone_TD_Rate': 0.63},
        
        # AFC North
        {'Team': 'BAL', 'Games': 2, 'Pass_Yards_Allowed': 440, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 175, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 220.0, 'Rush_Yards_Allowed_Per_Game': 87.5,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 5, 'Rush_Defense_Rank': 2, 'Pass_TD_Defense_Rank': 4, 'Rush_TD_Defense_Rank': 2,
         'Pressure_Rate': 0.33, 'Blitz_Rate': 0.23, 'Coverage_Grade': 76.0, 'Red_Zone_TD_Rate': 0.46},
        {'Team': 'CIN', 'Games': 2, 'Pass_Yards_Allowed': 490, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 210, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 245.0, 'Rush_Yards_Allowed_Per_Game': 105.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 14, 'Rush_Defense_Rank': 12, 'Pass_TD_Defense_Rank': 11, 'Rush_TD_Defense_Rank': 12,
         'Pressure_Rate': 0.27, 'Blitz_Rate': 0.26, 'Coverage_Grade': 70.0, 'Red_Zone_TD_Rate': 0.55},
        {'Team': 'CLE', 'Games': 2, 'Pass_Yards_Allowed': 470, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 190, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 235.0, 'Rush_Yards_Allowed_Per_Game': 95.0,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 9, 'Rush_Defense_Rank': 5, 'Pass_TD_Defense_Rank': 5, 'Rush_TD_Defense_Rank': 5,
         'Pressure_Rate': 0.29, 'Blitz_Rate': 0.27, 'Coverage_Grade': 71.0, 'Red_Zone_TD_Rate': 0.52},
        {'Team': 'PIT', 'Games': 2, 'Pass_Yards_Allowed': 450, 'Pass_TDs_Allowed': 1, 
         'Rush_Yards_Allowed': 180, 'Rush_TDs_Allowed': 0, 'Total_TDs_Allowed': 1,
         'Pass_Yards_Allowed_Per_Game': 225.0, 'Rush_Yards_Allowed_Per_Game': 90.0,
         'Pass_TDs_Allowed_Per_Game': 0.5, 'Rush_TDs_Allowed_Per_Game': 0.0,
         'Pass_Defense_Rank': 7, 'Rush_Defense_Rank': 3, 'Pass_TD_Defense_Rank': 1, 'Rush_TD_Defense_Rank': 1,
         'Pressure_Rate': 0.32, 'Blitz_Rate': 0.27, 'Coverage_Grade': 74.0, 'Red_Zone_TD_Rate': 0.47},
        
        # AFC South
        {'Team': 'HOU', 'Games': 2, 'Pass_Yards_Allowed': 460, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 185, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 230.0, 'Rush_Yards_Allowed_Per_Game': 92.5,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 8, 'Rush_Defense_Rank': 4, 'Pass_TD_Defense_Rank': 4, 'Rush_TD_Defense_Rank': 4,
         'Pressure_Rate': 0.28, 'Blitz_Rate': 0.25, 'Coverage_Grade': 72.0, 'Red_Zone_TD_Rate': 0.49},
        {'Team': 'IND', 'Games': 2, 'Pass_Yards_Allowed': 510, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 210, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 255.0, 'Rush_Yards_Allowed_Per_Game': 105.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 18, 'Rush_Defense_Rank': 14, 'Pass_TD_Defense_Rank': 12, 'Rush_TD_Defense_Rank': 14,
         'Pressure_Rate': 0.25, 'Blitz_Rate': 0.29, 'Coverage_Grade': 66.0, 'Red_Zone_TD_Rate': 0.57},
        {'Team': 'JAX', 'Games': 2, 'Pass_Yards_Allowed': 520, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 205, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 260.0, 'Rush_Yards_Allowed_Per_Game': 102.5,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 20, 'Rush_Defense_Rank': 10, 'Pass_TD_Defense_Rank': 12, 'Rush_TD_Defense_Rank': 10,
         'Pressure_Rate': 0.24, 'Blitz_Rate': 0.29, 'Coverage_Grade': 66.0, 'Red_Zone_TD_Rate': 0.56},
        {'Team': 'TEN', 'Games': 2, 'Pass_Yards_Allowed': 530, 'Pass_TDs_Allowed': 4, 
         'Rush_Yards_Allowed': 230, 'Rush_TDs_Allowed': 3, 'Total_TDs_Allowed': 7,
         'Pass_Yards_Allowed_Per_Game': 265.0, 'Rush_Yards_Allowed_Per_Game': 115.0,
         'Pass_TDs_Allowed_Per_Game': 2.0, 'Rush_TDs_Allowed_Per_Game': 1.5,
         'Pass_Defense_Rank': 22, 'Rush_Defense_Rank': 20, 'Pass_TD_Defense_Rank': 22, 'Rush_TD_Defense_Rank': 20,
         'Pressure_Rate': 0.22, 'Blitz_Rate': 0.30, 'Coverage_Grade': 63.0, 'Red_Zone_TD_Rate': 0.65},
        
        # AFC West
        {'Team': 'KC', 'Games': 2, 'Pass_Yards_Allowed': 470, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 190, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 235.0, 'Rush_Yards_Allowed_Per_Game': 95.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 9, 'Rush_Defense_Rank': 5, 'Pass_TD_Defense_Rank': 10, 'Rush_TD_Defense_Rank': 12,
         'Pressure_Rate': 0.27, 'Blitz_Rate': 0.28, 'Coverage_Grade': 71.0, 'Red_Zone_TD_Rate': 0.52},
        {'Team': 'LAC', 'Games': 2, 'Pass_Yards_Allowed': 460, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 180, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 230.0, 'Rush_Yards_Allowed_Per_Game': 90.0,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 8, 'Rush_Defense_Rank': 3, 'Pass_TD_Defense_Rank': 5, 'Rush_TD_Defense_Rank': 3,
         'Pressure_Rate': 0.31, 'Blitz_Rate': 0.26, 'Coverage_Grade': 73.0, 'Red_Zone_TD_Rate': 0.48},
        {'Team': 'DEN', 'Games': 2, 'Pass_Yards_Allowed': 420, 'Pass_TDs_Allowed': 1, 
         'Rush_Yards_Allowed': 160, 'Rush_TDs_Allowed': 0, 'Total_TDs_Allowed': 1,
         'Pass_Yards_Allowed_Per_Game': 210.0, 'Rush_Yards_Allowed_Per_Game': 80.0,
         'Pass_TDs_Allowed_Per_Game': 0.5, 'Rush_TDs_Allowed_Per_Game': 0.0,
         'Pass_Defense_Rank': 3, 'Rush_Defense_Rank': 1, 'Pass_TD_Defense_Rank': 1, 'Rush_TD_Defense_Rank': 1,
         'Pressure_Rate': 0.35, 'Blitz_Rate': 0.22, 'Coverage_Grade': 78.0, 'Red_Zone_TD_Rate': 0.42},
        {'Team': 'LV', 'Games': 2, 'Pass_Yards_Allowed': 540, 'Pass_TDs_Allowed': 4, 
         'Rush_Yards_Allowed': 240, 'Rush_TDs_Allowed': 3, 'Total_TDs_Allowed': 7,
         'Pass_Yards_Allowed_Per_Game': 270.0, 'Rush_Yards_Allowed_Per_Game': 120.0,
         'Pass_TDs_Allowed_Per_Game': 2.0, 'Rush_TDs_Allowed_Per_Game': 1.5,
         'Pass_Defense_Rank': 26, 'Rush_Defense_Rank': 22, 'Pass_TD_Defense_Rank': 26, 'Rush_TD_Defense_Rank': 22,
         'Pressure_Rate': 0.21, 'Blitz_Rate': 0.31, 'Coverage_Grade': 61.0, 'Red_Zone_TD_Rate': 0.69},
        
        # NFC East
        {'Team': 'PHI', 'Games': 2, 'Pass_Yards_Allowed': 470, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 190, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 235.0, 'Rush_Yards_Allowed_Per_Game': 95.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 11, 'Rush_Defense_Rank': 5, 'Pass_TD_Defense_Rank': 10, 'Rush_TD_Defense_Rank': 12,
         'Pressure_Rate': 0.28, 'Blitz_Rate': 0.26, 'Coverage_Grade': 70.0, 'Red_Zone_TD_Rate': 0.54},
        {'Team': 'DAL', 'Games': 2, 'Pass_Yards_Allowed': 500, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 220, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 250.0, 'Rush_Yards_Allowed_Per_Game': 110.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 16, 'Rush_Defense_Rank': 16, 'Pass_TD_Defense_Rank': 12, 'Rush_TD_Defense_Rank': 16,
         'Pressure_Rate': 0.26, 'Blitz_Rate': 0.29, 'Coverage_Grade': 68.0, 'Red_Zone_TD_Rate': 0.58},
        {'Team': 'NYG', 'Games': 2, 'Pass_Yards_Allowed': 510, 'Pass_TDs_Allowed': 4, 
         'Rush_Yards_Allowed': 230, 'Rush_TDs_Allowed': 3, 'Total_TDs_Allowed': 7,
         'Pass_Yards_Allowed_Per_Game': 255.0, 'Rush_Yards_Allowed_Per_Game': 115.0,
         'Pass_TDs_Allowed_Per_Game': 2.0, 'Rush_TDs_Allowed_Per_Game': 1.5,
         'Pass_Defense_Rank': 18, 'Rush_Defense_Rank': 20, 'Pass_TD_Defense_Rank': 18, 'Rush_TD_Defense_Rank': 20,
         'Pressure_Rate': 0.24, 'Blitz_Rate': 0.30, 'Coverage_Grade': 66.0, 'Red_Zone_TD_Rate': 0.61},
        
        # NFC North
        {'Team': 'DET', 'Games': 2, 'Pass_Yards_Allowed': 480, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 200, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 240.0, 'Rush_Yards_Allowed_Per_Game': 100.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 12, 'Rush_Defense_Rank': 6, 'Pass_TD_Defense_Rank': 10, 'Rush_TD_Defense_Rank': 6,
         'Pressure_Rate': 0.29, 'Blitz_Rate': 0.25, 'Coverage_Grade': 70.0, 'Red_Zone_TD_Rate': 0.55},
        {'Team': 'GB', 'Games': 2, 'Pass_Yards_Allowed': 450, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 180, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 225.0, 'Rush_Yards_Allowed_Per_Game': 90.0,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 7, 'Rush_Defense_Rank': 3, 'Pass_TD_Defense_Rank': 5, 'Rush_TD_Defense_Rank': 3,
         'Pressure_Rate': 0.30, 'Blitz_Rate': 0.24, 'Coverage_Grade': 73.0, 'Red_Zone_TD_Rate': 0.49},
        {'Team': 'MIN', 'Games': 2, 'Pass_Yards_Allowed': 440, 'Pass_TDs_Allowed': 1, 
         'Rush_Yards_Allowed': 170, 'Rush_TDs_Allowed': 0, 'Total_TDs_Allowed': 1,
         'Pass_Yards_Allowed_Per_Game': 220.0, 'Rush_Yards_Allowed_Per_Game': 85.0,
         'Pass_TDs_Allowed_Per_Game': 0.5, 'Rush_TDs_Allowed_Per_Game': 0.0,
         'Pass_Defense_Rank': 5, 'Rush_Defense_Rank': 1, 'Pass_TD_Defense_Rank': 1, 'Rush_TD_Defense_Rank': 1,
         'Pressure_Rate': 0.31, 'Blitz_Rate': 0.23, 'Coverage_Grade': 75.0, 'Red_Zone_TD_Rate': 0.46},
        {'Team': 'CHI', 'Games': 2, 'Pass_Yards_Allowed': 520, 'Pass_TDs_Allowed': 4, 
         'Rush_Yards_Allowed': 210, 'Rush_TDs_Allowed': 3, 'Total_TDs_Allowed': 7,
         'Pass_Yards_Allowed_Per_Game': 260.0, 'Rush_Yards_Allowed_Per_Game': 105.0,
         'Pass_TDs_Allowed_Per_Game': 2.0, 'Rush_TDs_Allowed_Per_Game': 1.5,
         'Pass_Defense_Rank': 20, 'Rush_Defense_Rank': 14, 'Pass_TD_Defense_Rank': 20, 'Rush_TD_Defense_Rank': 14,
         'Pressure_Rate': 0.25, 'Blitz_Rate': 0.28, 'Coverage_Grade': 67.0, 'Red_Zone_TD_Rate': 0.59},
        
        # NFC South
        {'Team': 'TB', 'Games': 2, 'Pass_Yards_Allowed': 460, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 180, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 230.0, 'Rush_Yards_Allowed_Per_Game': 90.0,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 11, 'Rush_Defense_Rank': 3, 'Pass_TD_Defense_Rank': 5, 'Rush_TD_Defense_Rank': 3,
         'Pressure_Rate': 0.28, 'Blitz_Rate': 0.26, 'Coverage_Grade': 71.0, 'Red_Zone_TD_Rate': 0.50},
        {'Team': 'ATL', 'Games': 2, 'Pass_Yards_Allowed': 490, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 200, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 245.0, 'Rush_Yards_Allowed_Per_Game': 100.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 14, 'Rush_Defense_Rank': 6, 'Pass_TD_Defense_Rank': 12, 'Rush_TD_Defense_Rank': 6,
         'Pressure_Rate': 0.26, 'Blitz_Rate': 0.27, 'Coverage_Grade': 69.0, 'Red_Zone_TD_Rate': 0.57},
        {'Team': 'NO', 'Games': 2, 'Pass_Yards_Allowed': 520, 'Pass_TDs_Allowed': 4, 
         'Rush_Yards_Allowed': 220, 'Rush_TDs_Allowed': 3, 'Total_TDs_Allowed': 7,
         'Pass_Yards_Allowed_Per_Game': 260.0, 'Rush_Yards_Allowed_Per_Game': 110.0,
         'Pass_TDs_Allowed_Per_Game': 2.0, 'Rush_TDs_Allowed_Per_Game': 1.5,
         'Pass_Defense_Rank': 20, 'Rush_Defense_Rank': 16, 'Pass_TD_Defense_Rank': 20, 'Rush_TD_Defense_Rank': 16,
         'Pressure_Rate': 0.23, 'Blitz_Rate': 0.29, 'Coverage_Grade': 64.0, 'Red_Zone_TD_Rate': 0.63},
        {'Team': 'CAR', 'Games': 2, 'Pass_Yards_Allowed': 530, 'Pass_TDs_Allowed': 4, 
         'Rush_Yards_Allowed': 230, 'Rush_TDs_Allowed': 3, 'Total_TDs_Allowed': 7,
         'Pass_Yards_Allowed_Per_Game': 265.0, 'Rush_Yards_Allowed_Per_Game': 115.0,
         'Pass_TDs_Allowed_Per_Game': 2.0, 'Rush_TDs_Allowed_Per_Game': 1.5,
         'Pass_Defense_Rank': 22, 'Rush_Defense_Rank': 20, 'Pass_TD_Defense_Rank': 22, 'Rush_TD_Defense_Rank': 20,
         'Pressure_Rate': 0.22, 'Blitz_Rate': 0.30, 'Coverage_Grade': 62.0, 'Red_Zone_TD_Rate': 0.67},
        
        # NFC West
        {'Team': 'SF', 'Games': 2, 'Pass_Yards_Allowed': 430, 'Pass_TDs_Allowed': 2, 
         'Rush_Yards_Allowed': 160, 'Rush_TDs_Allowed': 1, 'Total_TDs_Allowed': 3,
         'Pass_Yards_Allowed_Per_Game': 215.0, 'Rush_Yards_Allowed_Per_Game': 80.0,
         'Pass_TDs_Allowed_Per_Game': 1.0, 'Rush_TDs_Allowed_Per_Game': 0.5,
         'Pass_Defense_Rank': 3, 'Rush_Defense_Rank': 1, 'Pass_TD_Defense_Rank': 5, 'Rush_TD_Defense_Rank': 1,
         'Pressure_Rate': 0.34, 'Blitz_Rate': 0.22, 'Coverage_Grade': 77.0, 'Red_Zone_TD_Rate': 0.44},
        {'Team': 'ARI', 'Games': 2, 'Pass_Yards_Allowed': 500, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 190, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 250.0, 'Rush_Yards_Allowed_Per_Game': 95.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 16, 'Rush_Defense_Rank': 5, 'Pass_TD_Defense_Rank': 12, 'Rush_TD_Defense_Rank': 12,
         'Pressure_Rate': 0.26, 'Blitz_Rate': 0.28, 'Coverage_Grade': 68.0, 'Red_Zone_TD_Rate': 0.57},
        {'Team': 'LAR', 'Games': 2, 'Pass_Yards_Allowed': 470, 'Pass_TDs_Allowed': 3, 
         'Rush_Yards_Allowed': 180, 'Rush_TDs_Allowed': 2, 'Total_TDs_Allowed': 5,
         'Pass_Yards_Allowed_Per_Game': 235.0, 'Rush_Yards_Allowed_Per_Game': 90.0,
         'Pass_TDs_Allowed_Per_Game': 1.5, 'Rush_TDs_Allowed_Per_Game': 1.0,
         'Pass_Defense_Rank': 11, 'Rush_Defense_Rank': 3, 'Pass_TD_Defense_Rank': 10, 'Rush_TD_Defense_Rank': 3,
         'Pressure_Rate': 0.29, 'Blitz_Rate': 0.25, 'Coverage_Grade': 72.0, 'Red_Zone_TD_Rate': 0.51},
        {'Team': 'SEA', 'Games': 2, 'Pass_Yards_Allowed': 410, 'Pass_TDs_Allowed': 1, 
         'Rush_Yards_Allowed': 150, 'Rush_TDs_Allowed': 0, 'Total_TDs_Allowed': 1,
         'Pass_Yards_Allowed_Per_Game': 205.0, 'Rush_Yards_Allowed_Per_Game': 75.0,
         'Pass_TDs_Allowed_Per_Game': 0.5, 'Rush_TDs_Allowed_Per_Game': 0.0,
         'Pass_Defense_Rank': 1, 'Rush_Defense_Rank': 1, 'Pass_TD_Defense_Rank': 1, 'Rush_TD_Defense_Rank': 1,
         'Pressure_Rate': 0.36, 'Blitz_Rate': 0.21, 'Coverage_Grade': 79.0, 'Red_Zone_TD_Rate': 0.40},
    ]
    
    return {
        'qb': pd.DataFrame(qb_data),
        'rb': pd.DataFrame(rb_data),
        'wr': pd.DataFrame(wr_data),
        'te': pd.DataFrame(te_data),
        'defense': pd.DataFrame(defense_data)
    }

def get_week3_matchup_adjustments():
    """Get Week 3 specific matchup adjustments based on early season trends"""
    return {
        # Teams showing improved/declined performance through Week 2
        'hot_teams': ['BUF', 'BAL', 'PHI', 'PIT', 'HOU', 'DET', 'MIN'],  # Strong Week 1-2
        'cold_teams': ['CLE', 'CAR', 'LV', 'NYG', 'CHI'],  # Struggling early
        'defensive_surprises': {
            'improved': ['PIT', 'MIN', 'DEN'],  # Better than expected defense
            'declined': ['DAL', 'SF', 'LAR']   # Worse than expected defense
        },
        'injury_impacts': {
            'questionable': ['CMC', 'Tua', 'Watson'],  # Players with injury concerns
            'returning': ['Richardson', 'Rodgers']      # Players returning from injury
        }
    }

class AdvancedNFLProjector:
    def __init__(self):
        self.data = load_comprehensive_data()
        self.models = {}
        self.trained_models = set()
        
    def prepare_advanced_training_data(self, stat_type, position):
        """Prepare training data with advanced metrics for specific stat type and position"""
        
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
                
                if stat_type == 'passing_yards':
                    features = {
                        'Player_Stat_Per_Game': player['Pass_Yards_Per_Game'],
                        'Player_Efficiency': player['YPA'],
                        'Player_Secondary_Stat': player['Completion_Pct'],
                        'Player_Advanced_1': player['ADOT'],
                        'Player_Advanced_2': player['EPA_Per_Play'],
                        'Player_Advanced_3': player['Deep_Ball_Pct'],
                        'Opp_Stat_Allowed_Per_Game': defense['Pass_Yards_Allowed_Per_Game'],
                        'Opp_Defense_Rank': defense['Pass_Defense_Rank'],
                        'Opp_Advanced_1': defense['Pressure_Rate'],
                        'Opp_Advanced_2': defense['Coverage_Grade']
                    }
                    base_projection = (player['Pass_Yards_Per_Game'] + defense['Pass_Yards_Allowed_Per_Game']) / 2
                    
                elif stat_type == 'rushing_yards':
                    features = {
                        'Player_Stat_Per_Game': player['Rush_Yards_Per_Game'],
                        'Player_Efficiency': player['YPC'],
                        'Player_Secondary_Stat': player['Rush_Attempts_Per_Game'],
                        'Player_Advanced_1': player['Snap_Count_Pct'],
                        'Player_Advanced_2': player['Broken_Tackles'],
                        'Player_Advanced_3': player['Yards_After_Contact'],
                        'Opp_Stat_Allowed_Per_Game': defense['Rush_Yards_Allowed_Per_Game'],
                        'Opp_Defense_Rank': defense['Rush_Defense_Rank'],
                        'Opp_Advanced_1': defense['Pressure_Rate'],
                        'Opp_Advanced_2': defense['Coverage_Grade']
                    }
                    base_projection = (player['Rush_Yards_Per_Game'] + defense['Rush_Yards_Allowed_Per_Game']) / 2
                    
                elif stat_type == 'receiving_yards':
                    features = {
                        'Player_Stat_Per_Game': player['Rec_Yards_Per_Game'],
                        'Player_Efficiency': player['YPR'],
                        'Player_Secondary_Stat': player['Receptions_Per_Game'],
                        'Player_Advanced_1': player['Target_Share'],
                        'Player_Advanced_2': player['ADOT'],
                        'Player_Advanced_3': player.get('Air_Yards', player['Rec_Yards'] * 1.2),
                        'Opp_Stat_Allowed_Per_Game': defense['Pass_Yards_Allowed_Per_Game'],
                        'Opp_Defense_Rank': defense['Pass_Defense_Rank'],
                        'Opp_Advanced_1': defense['Pressure_Rate'],
                        'Opp_Advanced_2': defense['Coverage_Grade']
                    }
                    # Adjust for position-specific receiving usage
                    position_multiplier = 0.4 if position == 'RB' else 0.8 if position == 'TE' else 1.0
                    base_projection = (player['Rec_Yards_Per_Game'] + defense['Pass_Yards_Allowed_Per_Game'] * position_multiplier) / 2
                    
                elif stat_type in ['passing_tds', 'rushing_tds', 'receiving_tds']:
                    if stat_type == 'passing_tds':
                        player_stat = player['Pass_TDs_Per_Game']
                        opp_stat = defense['Pass_TDs_Allowed_Per_Game']
                        opp_rank = defense['Pass_TD_Defense_Rank']
                        advanced_1 = player['Red_Zone_TDs']
                        advanced_2 = player['EPA_Per_Play']
                    elif stat_type == 'rushing_tds':
                        player_stat = player['Rush_TDs_Per_Game']
                        opp_stat = defense['Rush_TDs_Allowed_Per_Game']
                        opp_rank = defense['Rush_TD_Defense_Rank']
                        advanced_1 = player['Goal_Line_Carries']
                        advanced_2 = player['Red_Zone_Touches']
                    else:  # receiving_tds
                        player_stat = player['Rec_TDs_Per_Game']
                        opp_stat = defense['Pass_TDs_Allowed_Per_Game']
                        opp_rank = defense['Pass_TD_Defense_Rank']
                        advanced_1 = player['Red_Zone_Targets']
                        advanced_2 = player['Target_Share']
                    
                    features = {
                        'Player_Stat_Per_Game': player_stat,
                        'Player_Efficiency': player_stat * 16,
                        'Player_Secondary_Stat': player_stat,
                        'Player_Advanced_1': advanced_1,
                        'Player_Advanced_2': advanced_2,
                        'Player_Advanced_3': player_stat * 10,
                        'Opp_Stat_Allowed_Per_Game': opp_stat,
                        'Opp_Defense_Rank': opp_rank,
                        'Opp_Advanced_1': defense['Red_Zone_TD_Rate'],
                        'Opp_Advanced_2': defense['Coverage_Grade']
                    }
                    base_projection = (player_stat + opp_stat) / 2
                
                # Enhanced matchup factor with advanced metrics
                defense_strength = (features['Opp_Defense_Rank'] - 16.5) / 32
                coverage_factor = (features['Opp_Advanced_2'] - 70) / 100
                matchup_factor = 1 + (defense_strength * 0.25) + (coverage_factor * 0.15)
                
                target = base_projection * matchup_factor
                
                # Add realistic variance
                if 'yards' in stat_type:
                    target += np.random.normal(0, 12)
                    target = max(5, target)
                else:  # TDs
                    target += np.random.normal(0, 0.25)
                    target = max(0, target)
                
                features['Target'] = target
                training_data.append(features)
        
        df = pd.DataFrame(training_data)
        feature_columns = ['Player_Stat_Per_Game', 'Player_Efficiency', 'Player_Secondary_Stat', 
                          'Player_Advanced_1', 'Player_Advanced_2', 'Player_Advanced_3',
                          'Opp_Stat_Allowed_Per_Game', 'Opp_Defense_Rank', 
                          'Opp_Advanced_1', 'Opp_Advanced_2']
        
        X = df[feature_columns]
        y = df['Target']
        
        return X, y

    def train_advanced_model(self, stat_type, position):
        """Train advanced model with enhanced features"""
        X, y = self.prepare_advanced_training_data(stat_type, position)
        
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
        
        return mae, r2, self.models[model_key].coef_

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
            confidence = "HIGH" if abs(edge_percentage) > 8 else "MEDIUM" if abs(edge_percentage) > 4 else "LOW"
        else:  # TDs
            confidence = "HIGH" if abs(edge_percentage) > 15 else "MEDIUM" if abs(edge_percentage) > 8 else "LOW"
        
        return {
            'projection': projection,
            'betting_line': betting_line,
            'difference': difference,
            'edge_percentage': edge_percentage,
            'recommendation': recommendation,
            'confidence': confidence
        }

    def project_player_performance(self, player_name, position, opponent_team, stat_type, betting_line=None):
        """Project player performance using advanced metrics"""
        model_key = f"{position}_{stat_type}"
        
        if model_key not in self.trained_models:
            # Train model if not already trained
            self.train_advanced_model(stat_type, position)
        
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
        
        if stat_type == 'passing_yards':
            features = np.array([[
                player_stats['Pass_Yards_Per_Game'],
                player_stats['YPA'],
                player_stats['Completion_Pct'],
                player_stats['ADOT'],
                player_stats['EPA_Per_Play'],
                player_stats['Deep_Ball_Pct'],
                opp_stats['Pass_Yards_Allowed_Per_Game'],
                opp_stats['Pass_Defense_Rank'],
                opp_stats['Pressure_Rate'],
                opp_stats['Coverage_Grade']
            ]])
            defense_rank = opp_stats['Pass_Defense_Rank']
            
        elif stat_type == 'rushing_yards':
            features = np.array([[
                player_stats['Rush_Yards_Per_Game'],
                player_stats['YPC'],
                player_stats['Rush_Attempts_Per_Game'],
                player_stats['Snap_Count_Pct'],
                player_stats['Broken_Tackles'],
                player_stats['Yards_After_Contact'],
                opp_stats['Rush_Yards_Allowed_Per_Game'],
                opp_stats['Rush_Defense_Rank'],
                opp_stats['Pressure_Rate'],
                opp_stats['Coverage_Grade']
            ]])
            defense_rank = opp_stats['Rush_Defense_Rank']
            
        elif stat_type == 'receiving_yards':
            features = np.array([[
                player_stats['Rec_Yards_Per_Game'],
                player_stats['YPR'],
                player_stats['Receptions_Per_Game'],
                player_stats['Target_Share'],
                player_stats['ADOT'],
                player_stats.get('Air_Yards', player_stats['Rec_Yards'] * 1.2),
                opp_stats['Pass_Yards_Allowed_Per_Game'],
                opp_stats['Pass_Defense_Rank'],
                opp_stats['Pressure_Rate'],
                opp_stats['Coverage_Grade']
            ]])
            defense_rank = opp_stats['Pass_Defense_Rank']
            
        elif stat_type in ['passing_tds', 'rushing_tds', 'receiving_tds']:
            if stat_type == 'passing_tds':
                player_stat = player_stats['Pass_TDs_Per_Game']
                opp_stat = opp_stats['Pass_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Pass_TD_Defense_Rank']
                advanced_1 = player_stats['Red_Zone_TDs']
                advanced_2 = player_stats['EPA_Per_Play']
            elif stat_type == 'rushing_tds':
                player_stat = player_stats['Rush_TDs_Per_Game']
                opp_stat = opp_stats['Rush_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Rush_TD_Defense_Rank']
                advanced_1 = player_stats['Goal_Line_Carries']
                advanced_2 = player_stats['Red_Zone_Touches']
            else:  # receiving_tds
                player_stat = player_stats['Rec_TDs_Per_Game']
                opp_stat = opp_stats['Pass_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Pass_TD_Defense_Rank']
                advanced_1 = player_stats['Red_Zone_Targets']
                advanced_2 = player_stats['Target_Share']
            
            features = np.array([[
                player_stat,
                player_stat * 16,
                player_stat,
                advanced_1,
                advanced_2,
                player_stat * 10,
                opp_stat,
                defense_rank,
                opp_stats['Red_Zone_TD_Rate'],
                opp_stats['Coverage_Grade']
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

    def calculate_week3_projection(self, player_name, opponent, stat_type, position):
        """Enhanced projection calculation for Week 3 with current season weighting"""
        
        # Get base projection from model
        model_key = f"{position}_{stat_type}"
        if model_key not in self.trained_models:
            # Train model if not already trained
            self.train_advanced_model(stat_type, position)
        
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
        opp_data = self.data['defense'][self.data['defense']['Team'] == opponent.upper()]
        if opp_data.empty:
            raise ValueError(f"Team '{opponent}' not found in defensive data")
        
        opp_stats = opp_data.iloc[0]
        
        # Apply Week 3 specific adjustments
        adjustments = get_week3_matchup_adjustments()
        
        week_3_weight = 0.35  # 35% current season, 65% historical for Week 3
        
        # Get player's current form factor
        if 'Current_Form_Factor' in player_stats:
            form_adjustment = (player_stats['Current_Form_Factor'] - 1.0) * week_3_weight
            
            if stat_type == 'passing_yards':
                base_projection = player_stats['Historical_YPG'] * (1 - week_3_weight) + player_stats['Pass_Yards_Per_Game'] * week_3_weight
                base_projection *= (1 + form_adjustment)
            elif stat_type == 'rushing_yards':
                base_projection = player_stats['Historical_Rush_YPG'] * (1 - week_3_weight) + player_stats['Rush_Yards_Per_Game'] * week_3_weight
                base_projection *= (1 + form_adjustment)
            elif stat_type == 'receiving_yards':
                base_projection = player_stats['Historical_Rec_YPG'] * (1 - week_3_weight) + player_stats['Rec_Yards_Per_Game'] * week_3_weight
                base_projection *= (1 + form_adjustment)
            elif stat_type == 'passing_tds':
                base_projection = player_stats['Pass_TDs_Per_Game']
                base_projection *= (1 + form_adjustment)
            elif stat_type == 'rushing_tds':
                base_projection = player_stats['Rush_TDs_Per_Game']
                base_projection *= (1 + form_adjustment)
            elif stat_type == 'receiving_tds':
                base_projection = player_stats['Rec_TDs_Per_Game']
                base_projection *= (1 + form_adjustment)
        else:
            if stat_type == 'passing_yards':
                base_projection = player_stats['Pass_Yards_Per_Game']
            elif stat_type == 'rushing_yards':
                base_projection = player_stats['Rush_Yards_Per_Game']
            elif stat_type == 'receiving_yards':
                base_projection = player_stats['Rec_Yards_Per_Game']
            elif stat_type == 'passing_tds':
                base_projection = player_stats['Pass_TDs_Per_Game']
            elif stat_type == 'rushing_tds':
                base_projection = player_stats['Rush_TDs_Per_Game']
            elif stat_type == 'receiving_tds':
                base_projection = player_stats['Rec_TDs_Per_Game']
        
        # Team-based adjustments
        player_team = player_stats['Team']
        if player_team in adjustments['hot_teams']:
            base_projection *= 1.08  # 8% boost for hot teams
        elif player_team in adjustments['cold_teams']:
            base_projection *= 0.92  # 8% reduction for cold teams
        
        # Defensive adjustments for opponent
        if opponent in adjustments['defensive_surprises']['improved']:
            base_projection *= 0.90  # Tougher matchup than expected
        elif opponent in adjustments['defensive_surprises']['declined']:
            base_projection *= 1.10  # Easier matchup than expected
        
        return base_projection

    def predict_stat(self, player_name, opponent, stat_type, position):
        """Predict player performance using advanced metrics"""
        model_key = f"{position}_{stat_type}"
        
        if model_key not in self.trained_models:
            # Train model if not already trained
            self.train_advanced_model(stat_type, position)
        
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
        opp_data = self.data['defense'][self.data['defense']['Team'] == opponent.upper()]
        if opp_data.empty:
            raise ValueError(f"Team '{opponent}' not found in defensive data")
        
        opp_stats = opp_data.iloc[0]
        
        if stat_type == 'passing_yards':
            features = np.array([[
                player_stats['Pass_Yards_Per_Game'],
                player_stats['YPA'],
                player_stats['Completion_Pct'],
                player_stats['ADOT'],
                player_stats['EPA_Per_Play'],
                player_stats['Deep_Ball_Pct'],
                opp_stats['Pass_Yards_Allowed_Per_Game'],
                opp_stats['Pass_Defense_Rank'],
                opp_stats['Pressure_Rate'],
                opp_stats['Coverage_Grade']
            ]])
            defense_rank = opp_stats['Pass_Defense_Rank']
            
        elif stat_type == 'rushing_yards':
            features = np.array([[
                player_stats['Rush_Yards_Per_Game'],
                player_stats['YPC'],
                player_stats['Rush_Attempts_Per_Game'],
                player_stats['Snap_Count_Pct'],
                player_stats['Broken_Tackles'],
                player_stats['Yards_After_Contact'],
                opp_stats['Rush_Yards_Allowed_Per_Game'],
                opp_stats['Rush_Defense_Rank'],
                opp_stats['Pressure_Rate'],
                opp_stats['Coverage_Grade']
            ]])
            defense_rank = opp_stats['Rush_Defense_Rank']
            
        elif stat_type == 'receiving_yards':
            features = np.array([[
                player_stats['Rec_Yards_Per_Game'],
                player_stats['YPR'],
                player_stats['Receptions_Per_Game'],
                player_stats['Target_Share'],
                player_stats['ADOT'],
                player_stats.get('Air_Yards', player_stats['Rec_Yards'] * 1.2),
                opp_stats['Pass_Yards_Allowed_Per_Game'],
                opp_stats['Pass_Defense_Rank'],
                opp_stats['Pressure_Rate'],
                opp_stats['Coverage_Grade']
            ]])
            defense_rank = opp_stats['Pass_Defense_Rank']
            
        elif stat_type in ['passing_tds', 'rushing_tds', 'receiving_tds']:
            if stat_type == 'passing_tds':
                player_stat = player_stats['Pass_TDs_Per_Game']
                opp_stat = opp_stats['Pass_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Pass_TD_Defense_Rank']
                advanced_1 = player_stats['Red_Zone_TDs']
                advanced_2 = player_stats['EPA_Per_Play']
            elif stat_type == 'rushing_tds':
                player_stat = player_stats['Rush_TDs_Per_Game']
                opp_stat = opp_stats['Rush_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Rush_TD_Defense_Rank']
                advanced_1 = player_stats['Goal_Line_Carries']
                advanced_2 = player_stats['Red_Zone_Touches']
            else:  # receiving_tds
                player_stat = player_stats['Rec_TDs_Per_Game']
                opp_stat = opp_stats['Pass_TDs_Allowed_Per_Game']
                defense_rank = opp_stats['Pass_TD_Defense_Rank']
                advanced_1 = player_stats['Red_Zone_Targets']
                advanced_2 = player_stats['Target_Share']
            
            features = np.array([[
                player_stat,
                player_stat * 16,
                player_stat,
                advanced_1,
                advanced_2,
                player_stat * 10,
                opp_stat,
                defense_rank,
                opp_stats['Red_Zone_TD_Rate'],
                opp_stats['Coverage_Grade']
            ]])
        
        # Make prediction
        model_key = f"{position}_{stat_type}"
        projection = self.models[model_key].predict(features)[0]
        
        return projection

    def get_player_data(self, player_name, position):
        """Helper function to retrieve player data"""
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
        
        player_match = player_data[player_data['Player'].str.contains(player_name, case=False)]
        if player_match.empty:
            return None
        
        return player_match.iloc[0]

    def get_player_team(self, player_name, position):
        """Helper function to retrieve player team"""
        player_data = self.get_player_data(player_name, position)
        if player_data is not None:
            return player_data['Team']
        return None

def main():
    st.title("🏈 NFL Week 3 2025 Advanced Projections")
    st.markdown("*Inspired by xEP_Network's AI-driven methodology with hybrid historical/current season data*")

    # Week 3 context sidebar
    with st.sidebar:
        st.header("⚡ Week 3 2025 Context")
        st.markdown("""
        **Data Weighting:**
        - 35% Current Season (Weeks 1-2)
        - 65% Historical Data (2024 season)
        
        **Hot Teams:** BUF, BAL, PHI, PIT, HOU
        **Cold Teams:** CLE, CAR, LV, NYG, CHI
        
        **Key Storylines:**
        - Saquon thriving in PHI
        - Russell Wilson's PIT resurgence  
        - Rookie QBs finding rhythm
        - Defensive surprises emerging
        """)
    
    # Initialize projector
    if 'projector' not in st.session_state:
        st.session_state.projector = AdvancedNFLProjector()
    
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
    if st.sidebar.button("Generate Advanced Projection", type="primary"):
        try:
            with st.spinner("Training advanced model and generating projection..."):
                # Generate projection
                #projection = projector.project_player_performance(
                #    player_name, 
                #    position, 
                #    opponent_team, 
                #    stat_type, 
                #    betting_line if betting_line > 0 else None
                #)
                projection = {}
                projection['projection'] = projector.calculate_week3_projection(player_name, opponent_team, stat_type, position)
                
                # Find opponent defense stats
                opp_data = projector.data['defense'][projector.data['defense']['Team'] == opponent_team.upper()]
                if opp_data.empty:
                    raise ValueError(f"Team '{opponent_team}' not found in defensive data")
                
                opp_stats = opp_data.iloc[0]
                
                if stat_type == 'passing_yards':
                    defense_rank = opp_stats['Pass_Defense_Rank']
                elif stat_type == 'rushing_yards':
                    defense_rank = opp_stats['Rush_Defense_Rank']
                elif stat_type == 'receiving_yards':
                    defense_rank = opp_stats['Pass_Defense_Rank']
                elif stat_type == 'passing_tds':
                    defense_rank = opp_stats['Pass_TD_Defense_Rank']
                elif stat_type == 'rushing_tds':
                    defense_rank = opp_stats['Rush_TD_Defense_Rank']
                elif stat_type == 'receiving_tds':
                    defense_rank = opp_stats['Pass_TD_Defense_Rank']
                
                projection['matchup_score'] = projector.calculate_matchup_score(defense_rank, stat_type)
                projection['defense_rank'] = defense_rank
                
                # Find player stats
                if position == 'QB':
                    player_data = projector.data['qb']
                elif position == 'RB':
                    player_data = projector.data['rb']
                elif position == 'WR':
                    player_data = projector.data['wr']
                elif position == 'TE':
                    player_data = projector.data['te']
                
                player_match = player_data[player_data['Player'].str.contains(player_name, case=False)]
                if player_match.empty:
                    raise ValueError(f"Player '{player_name}' not found in {position} data")
                
                player_stats = player_match.iloc[0]
                
                projection['player_name'] = player_stats['Player']
                projection['player_team'] = player_stats['Team']
                projection['opponent'] = opponent_team.upper()
                projection['stat_type'] = stat_type
                
                if betting_line:
                    betting_analysis = projector.calculate_betting_edge(projection['projection'], betting_line, stat_type)
                    projection['betting_analysis'] = betting_analysis
            
            # Display results
            st.header("📊 Advanced Projection Results")
            
            # Main projection card
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"{stat_display_names[stat_type]} Projection",
                    value=f"{projection['projection']:.1f}"
                )
            
            with col2:
                st.metric(
                    label="Matchup Score",
                    value=f"{projection['matchup_score']:.1f}/10"
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
            st.write(f"**Stat:** {stat_display_names[stat_type]}**")
            
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
                if abs(betting['edge_percentage']) > 8:
                    st.success(f"Strong {betting['recommendation']} play with {betting['edge_percentage']:+.1f}% edge!")
                elif abs(betting['edge_percentage']) > 4:
                    st.info(f"Moderate {betting['recommendation']} lean with {betting['edge_percentage']:+.1f}% edge")
                else:
                    st.warning(f"Weak edge of {betting['edge_percentage']:+.1f}% - consider avoiding this bet")
            
        except Exception as e:
            st.error(f"Error generating projection: {str(e)}")
    
    # Model performance section
    st.header("🤖 Advanced Model Information")
    
    with st.expander("View Enhanced Model Features"):
        st.write("**Advanced Training Features:**")
        st.write("• **QB**: ADOT, EPA per play, deep ball percentage, red zone TDs")
        st.write("• **RB**: Snap count %, broken tackles, yards after contact, goal line carries")
        st.write("• **WR/TE**: Target share, air yards, ADOT, red zone targets, slot rate")
        st.write("• **Defense**: Pressure rate, coverage grade, red zone TD rate, blitz rate")
        
        st.write("**Model Improvements:**")
        st.write("• 10 features per model (vs 5 in basic version)")
        st.write("• Position-specific advanced metrics")
        st.write("• Enhanced matchup scoring with coverage grades")
        st.write("• Improved variance modeling for realistic projections")
    
    # Advanced metrics suggestions
    st.header("📈 Future Enhancement Opportunities")
    
    with st.expander("Next-Level Analytics"):
        st.write("**Weather & Game Environment:**")
        st.write("• Wind speed and direction for passing games")
        st.write("• Temperature effects on ball handling")
        st.write("• Dome vs outdoor stadium factors")
        
        st.write("**Game Script Predictions:**")
        st.write("• Vegas totals and spread integration")
        st.write("• Pace of play adjustments")
        st.write("• Garbage time probability")
        
        st.write("**Injury & Load Management:**")
        st.write("• Snap count trends")
        st.write("• Injury report severity scoring")
        st.write("• Rest vs division rivals")
        
        st.write("**Advanced Situational Stats:**")
        st.write("• Third down conversion rates")
        st.write("• Two-minute drill efficiency")
        st.write("• Play action vs standard dropback splits")
    
    # Data tables
    st.header("📋 Comprehensive Player Database")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["QBs", "RBs", "WRs", "TEs", "Defense"])
    
    with tab1:
        st.subheader("Quarterback Stats with Advanced Metrics")
        st.dataframe(projector.data['qb'], use_container_width=True)
    
    with tab2:
        st.subheader("Running Back Stats with Advanced Metrics")
        st.dataframe(projector.data['rb'], use_container_width=True)
    
    with tab3:
        st.subheader("Wide Receiver Stats with Advanced Metrics")
        st.dataframe(projector.data['wr'], use_container_width=True)
    
    with tab4:
        st.subheader("Tight End Stats with Advanced Metrics")
        st.dataframe(projector.data['te'], use_container_width=True)
    
    with tab5:
        st.subheader("Defensive Stats with Advanced Metrics")
        st.dataframe(projector.data['defense'], use_container_width=True)

if __name__ == "__main__":
    main()
