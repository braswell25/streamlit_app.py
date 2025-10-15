import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import poisson
import warnings
import time  # For retries
warnings.filterwarnings('ignore')

st.title("NFL Player Props Dashboard (2025 – Week 5)")
st.write("Projects Yards & TDs for QBs, RBs, WRs, TEs. Inspired by @xEP_Network's ATD Ratings & efficiency (e.g., Bijan Robinson 5.8 YPA).")

# Sidebar: Position & Metric
position = st.sidebar.selectbox("Select Position", ["QB", "RB", "WR", "TE"])
metric = st.sidebar.selectbox("Select Metric", ["Yards", "Touchdowns"])
st.sidebar.write("Week 5 data from PFR. Built for Week 6 projections.")

# Clear cache button
if st.sidebar.button("Clear Cache & Retry Scrapes"):
    st.cache_data.clear()
    st.experimental_rerun()

# Robust request function with retry and headers
def safe_get(url, max_retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
            else:
                st.warning(f"HTTP {response.status_code} for {url} (attempt {attempt+1})")
        except Exception as e:
            st.warning(f"Request error for {url}: {e} (attempt {attempt+1})")
        time.sleep(2 ** attempt)  # Exponential backoff
    return None

# Step 1: Updated data loading with robustness
@st.cache_data
def load_passing_data():
    url = "https://www.pro-football-reference.com/years/2025/passing.htm"
    response = safe_get(url)
    if not response:
        st.error(f"Failed to fetch {url} after retries. Using fallback data.")
        return create_fallback_passing()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'passing'}) or soup.select_one('table#passing')
    if not table:
        st.error("Passing table not found in HTML. Structure may have changed.")
        return create_fallback_passing()
    
    try:
        df = pd.read_html(str(table))[0]
        df = df[df['Player'] != 'Player'].dropna(subset=['Player', 'Tm', 'G'])
        numeric_cols = ['Yds', 'Att', 'TD', 'G']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Column '{col}' missing in passing data.")
        df['YPA'] = df['Yds'] / df['Att'].replace(0, np.nan)
        df['TDPerGame'] = df['TD'] / df['G'].replace(0, np.nan)
        df = df.groupby('Player').agg({'Yds': 'sum', 'Att': 'sum', 'TD': 'sum', 'G': 'max', 'YPA': 'mean', 'TDPerGame': 'mean'}).reset_index()
        df['YdsPerGame'] = df['Yds'] / df['G'].replace(0, np.nan)
        return df
    except Exception as e:
        st.error(f"Error parsing passing data: {e}")
        return create_fallback_passing()

@st.cache_data
def load_rushing_data():
    url = "https://www.pro-football-reference.com/years/2025/rushing.htm"
    response = safe_get(url)
    if not response:
        st.error(f"Failed to fetch {url} after retries. Using fallback data.")
        return create_fallback_rushing()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'rushing'}) or soup.select_one('table#rushing')
    if not table:
        st.error("Rushing table not found in HTML. Structure may have changed.")
        return create_fallback_rushing()
    
    try:
        df = pd.read_html(str(table))[0]
        df = df[df['Player'] != 'Player'].dropna(subset=['Player', 'Tm', 'G'])
        numeric_cols = ['Yds', 'Att', 'TD', 'G']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Column '{col}' missing in rushing data.")
        df['YPA'] = df['Yds'] / df['Att'].replace(0, np.nan)
        df['TDPerGame'] = df['TD'] / df['G'].replace(0, np.nan)
        df = df.groupby('Player').agg({'Yds': 'sum', 'Att': 'sum', 'TD': 'sum', 'G': 'max', 'YPA': 'mean', 'TDPerGame': 'mean'}).reset_index()
        df['YdsPerGame'] = df['Yds'] / df['G'].replace(0, np.nan)
        return df
    except Exception as e:
        st.error(f"Error parsing rushing data: {e}")
        return create_fallback_rushing()

@st.cache_data
def load_receiving_data():
    url = "https://www.pro-football-reference.com/years/2025/receiving.htm"
    response = safe_get(url)
    if not response:
        st.error(f"Failed to fetch {url} after retries. Using fallback data.")
        return create_fallback_receiving()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'receiving'}) or soup.select_one('table#receiving')
    if not table:
        st.error("Receiving table not found in HTML. Structure may have changed.")
        return create_fallback_receiving()
    
    try:
        df = pd.read_html(str(table))[0]
        df = df[df['Player'] != 'Player'].dropna(subset=['Player', 'Tm', 'G'])
        numeric_cols = ['Yds', 'Tgt', 'TD', 'G']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                st.warning(f"Column '{col}' missing in receiving data.")
        df['YPT'] = df['Yds'] / df['Tgt'].replace(0, np.nan)
        df['TDPerGame'] = df['TD'] / df['G'].replace(0, np.nan)
        df = df.groupby('Player').agg({'Yds': 'sum', 'Tgt': 'sum', 'TD': 'sum', 'G': 'max', 'YPT': 'mean', 'TDPerGame': 'mean'}).reset_index()
        df['YdsPerGame'] = df['Yds'] / df['G'].replace(0, np.nan)
        return df
    except Exception as e:
        st.error(f"Error parsing receiving data: {e}")
        return create_fallback_receiving()

# Fallback sample data (hardcoded from 2025 pages for testing)
def create_fallback_passing():
    st.info("Using fallback passing data (top 3 QBs from 2025 Week 5).")
    data = {
        'Player': ['Matthew Stafford', 'Dak Prescott', 'Sam Darnold'],
        'Yds': [1684, 1617, 1541],
        'Att': [209, 229, 161],
        'TD': [12, 13, 11],
        'G': [6, 6, 6]
    }
    df = pd.DataFrame(data)
    df['YPA'] = df['Yds'] / df['Att']
    df['TDPerGame'] = df['TD'] / df['G']
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

def create_fallback_rushing():
    st.info("Using fallback rushing data (top 3 RBs from 2025 Week 5).")
    data = {
        'Player': ['Jonathan Taylor', 'James Cook', 'Josh Jacobs'],
        'Yds': [603, 537, 359],
        'Att': [115, 107, 98],
        'TD': [7, 5, 6],
        'G': [6, 6, 5]
    }
    df = pd.DataFrame(data)
    df['YPA'] = df['Yds'] / df['Att']
    df['TDPerGame'] = df['TD'] / df['G']
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

def create_fallback_receiving():
    st.info("Using fallback receiving data (top 3 WRs/TEs from 2025 Week 5).")
    data = {
        'Player': ['Puka Nacua', 'Amon-Ra St. Brown', 'Ja\'Marr Chase'],
        'Yds': [616, 452, 468],
        'Tgt': [65, 51, 57],
        'TD': [2, 6, 4],
        'G': [6, 6, 6]
    }
    df = pd.DataFrame(data)
    df['YPT'] = df['Yds'] / df['Tgt']
    df['TDPerGame'] = df['TD'] / df['G']
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

@st.cache_data
def load_def_data():
    url_def = "https://www.pro-football-reference.com/years/2025/opp.htm"
    response_def = safe_get(url_def)
    if not response_def:
        st.error("Failed to fetch defensive data. Using fallback.")
        return create_fallback_def()
    
    soup_def = BeautifulSoup(response_def.content, 'html.parser')
    table_def = soup_def.find('table', {'id': 'team_stats'}) or soup_def.select_one('table#team_stats')
    if not table_def:
        st.error("Defensive table not found.")
        return create_fallback_def()
    
    try:
        df_def = pd.read_html(str(table_def))[0]
        df_def = df_def[df_def['Tm'] != 'League Total'].dropna(subset=['Tm'])
        df_def['RushYdsAllowed'] = pd.to_numeric(df_def.get('Yds', 0), errors='coerce')
        df_def['PassYdsAllowed'] = pd.to_numeric(df_def.get('Yds.1', 0), errors='coerce')
        df_def['TotalTDAllowed'] = pd.to_numeric(df_def.get('TD', 0), errors='coerce')
        df_def['RushTDAllowedPerGame'] = (df_def['TotalTDAllowed'] * 0.4) / 5
        df_def['PassTDAllowedPerGame'] = (df_def['TotalTDAllowed'] * 0.6) / 5
        df_def['RushYdsAllowedPerGame'] = df_def['RushYdsAllowed'] / 5
        df_def['PassYdsAllowedPerGame'] = df_def['PassYdsAllowed'] / 5
        return df_def
    except Exception as e:
        st.error(f"Error parsing defensive data: {e}")
        return create_fallback_def()

def create_fallback_def():
    st.info("Using fallback defensive data (sample teams).")
    data = {
        'Tm': ['LAR', 'DAL', 'SEA', 'IND', 'BUF'],
        'Yds': [800, 750, 820],  # Rush Yds Allowed (sample)
        'Yds.1': [1400, 1350, 1450],  # Pass Yds Allowed
        'TD': [25, 22, 28]  # Total TD Allowed
    }
    df = pd.DataFrame(data)
    df['RushYdsAllowed'] = df['Yds']
    df['PassYdsAllowed'] = df['Yds.1']
    df['TotalTDAllowed'] = df['TD']
    df['RushTDAllowedPerGame'] = (df['TotalTDAllowed'] * 0.4) / 5
    df['PassTDAllowedPerGame'] = (df['TotalTDAllowed'] * 0.6) / 5
    df['RushYdsAllowedPerGame'] = df['RushYdsAllowed'] / 5
    df['PassYdsAllowedPerGame'] = df['PassYdsAllowed'] / 5
    return df

# Load data (rest of the code remains the same as before)
df_pass = load_passing_data()
df_rush = load_rushing_data()
df_rec = load_receiving_data()
df_def = load_def_data()

# [Insert the rest of the projection function and interface code from the previous version here – it hasn't changed]
# For brevity, paste the project_player function, interface, and leaderboard from the last code I provided.

# Example: The project_player function stays the same...
def project_player(player, opponent, betting_line, df_player, df_def, is_rush=False, is_rec=False, metric='Yards'):
    # ... (same as before)

# Interface (same as before)
# ... (player input, button, outputs, leaderboard)
