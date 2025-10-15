import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import poisson
import warnings
import time

warnings.filterwarnings('ignore')

st.title("NFL Player Props Dashboard (2025 – Week 5)")
st.write("Projects Yards & TDs for QBs, RBs, WRs, TEs. Inspired by @xEP_Network's ATD Ratings & efficiency (e.g., Bijan Robinson 5.8 YPA, Sam Darnold 11.2% CPOE).")
st.write("Week 6 projections using Week 5 data from Pro Football Reference.")

# Sidebar: Position & Metric
position = st.sidebar.selectbox("Select Position", ["QB", "RB", "WR", "TE"])
metric = st.sidebar.selectbox("Select Metric", ["Yards", "Touchdowns"])
st.sidebar.write("Data from PFR. If errors persist, click 'Clear Cache & Retry'.")

# Clear cache button
if st.sidebar.button("Clear Cache & Retry Scrapes"):
    st.cache_data.clear()
    st.experimental_rerun()

# Robust request function
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
        time.sleep(2 ** attempt)
    return None

# Data loading with fallbacks
@st.cache_data
def load_passing_data():
    url = "https://www.pro-football-reference.com/years/2025/passing.htm"
    response = safe_get(url)
    if not response:
        st.error(f"Failed to fetch {url}. Using fallback data.")
        return create_fallback_passing()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'passing'}) or soup.select_one('table#passing')
    if not table:
        st.error("Passing table not found. Structure may have changed.")
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
        st.error(f"Failed to fetch {url}. Using fallback data.")
        return create_fallback_rushing()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'rushing'}) or soup.select_one('table#rushing')
    if not table:
        st.error("Rushing table not found.")
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
        st.error(f"Failed to fetch {url}. Using fallback data.")
        return create_fallback_receiving()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'receiving'}) or soup.select_one('table#receiving')
    if not table:
        st.error("Receiving table not found.")
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

@st.cache_data
def load_def_data():
    url_def = "https://www.pro-football-reference.com/years/2025/opp.htm"
    response_def = safe_get(url_def)
    if not response_def:
        st.error("Failed to fetch defensive data.")
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

# Fallback data (based on 2025 Week 5 PFR snapshots)
def create_fallback_passing():
    st.info("Using fallback passing data (top QBs from 2025 Week 5).")
    data = {
        'Player': ['Sam Darnold', 'Jared Goff', 'Drake Maye'],
        'Yds': [1541, 1612, 1350],
        'Att': [161, 211, 184],
        'TD': [11, 10, 8],
        'G': [5, 5, 5]
    }
    df = pd.DataFrame(data)
    df['YPA'] = df['Yds'] / df['Att']
    df['TDPerGame'] = df['TD'] / df['G']
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

def create_fallback_rushing():
    st.info("Using fallback rushing data (top RBs from 2025 Week 5).")
    data = {
        'Player': ['Jonathan Taylor', 'Bijan Robinson', 'James Cook'],
        'Yds': [603, 580, 537],
        'Att': [115, 100, 107],
        'TD': [7, 4, 5],
        'G': [5, 5, 5]
    }
    df = pd.DataFrame(data)
    df['YPA'] = df['Yds'] / df['Att']
    df['TDPerGame'] = df['TD'] / df['G']
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

def create_fallback_receiving():
    st.info("Using fallback receiving data (top WRs/TEs from 2025 Week 5).")
    data = {
        'Player': ['Jaxon Smith-Njigba', 'Trey McBride', 'Puka Nacua'],
        'Yds': [696, 550, 616],
        'Tgt': [93, 78, 65],
        'TD': [3, 3, 2],
        'G': [5, 5, 5]
    }
    df = pd.DataFrame(data)
    df['YPT'] = df['Yds'] / df['Tgt']
    df['TDPerGame'] = df['TD'] / df['G']
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

def create_fallback_def():
    st.info("Using fallback defensive data (sample teams).")
    data = {
        'Tm': ['Atlanta Falcons', 'New Orleans Saints', 'Los Angeles Chargers'],
        'Yds': [600, 550, 580],  # Rush Yds Allowed
        'Yds.1': [1100, 1050, 1200],  # Pass Yds Allowed
        'TD': [10, 12, 11]  # Total TDs Allowed
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

# Load data
df_pass = load_passing_data()
df_rush = load_rushing_data()
df_rec = load_receiving_data()
df_def = load_def_data()

# Projection function
def project_player(player, opponent, betting_line, df_player, df_def, is_rush=False, is_rec=False, metric='Yards'):
    player_data = df_player[df_player['Player'].str.contains(player, case=False, na=False)]
    if player_data.empty:
        return f"Error: No 2025 Week 5 data for {player}"
    
    def_yards_metric = 'RushYdsAllowedPerGame' if is_rush else 'PassYdsAllowedPerGame'
    def_td_metric = 'RushTDAllowedPerGame' if is_rush else 'PassTDAllowedPerGame'
    opp_data = df_def[df_def['Tm'].str.contains(opponent, case=False, na=False)]
    if opp_data.empty:
        return f"Error: No data for opponent {opponent}"
    
    eff_col = 'YPA' if not is_rec else 'YPT'
    target = 'YdsPerGame' if metric == 'Yards' else 'TDPerGame'
    def_metric = def_yards_metric if metric == 'Yards' else def_td_metric
    
    X = np.array([[player_data[eff_col].iloc[0], player_data[target].iloc[0], 
                   opp_data[def_metric].iloc[0]]])
    y = player_data[target].values
    
    model = LinearRegression()
    model.fit(X, y)
    projection = model.predict(X)[0]
    
    edge = ((projection - betting_line) / betting_line) * 100
    
    df_def_copy = df_def.copy()
    df_def_copy['DefRank'] = df_def_copy[def_metric].rank(ascending=True, pct=True)
    opp_rank = df_def_copy[df_def_copy['Tm'].str.contains(opponent, case=False, na=False)]['DefRank'].iloc[0]
    matchup_score = 10 - (opp_rank * 9)
    
    td_prob = poisson.cdf(0, projection) if metric == 'Touchdowns' else None
    td_prob = (1 - td_prob) * 100 if td_prob is not None else None
    
    flag = ""
    if is_rush and player_data['YPA'].iloc[0] > 5.0:
        flag = " (xEP Efficiency Alert!)"
    elif is_rec and player_data['YPT'].iloc[0] > 7.0:
        flag = " (xEP YAC Threat!)"
    elif not is_rush and not is_rec and player_data['YPA'].iloc[0] > 8.0:
        flag = " (xEP Deep Threat!)"
    if metric == 'Touchdowns' and (td_prob or 0) > 70:
        flag += " (xEP TD Threat!)"
    
    return {
        'player': player,
        'opponent': opponent,
        'projected_value': round(projection, 1),
        'edge_percent': round(edge, 1),
        'matchup_score': round(matchup_score, 1),
        'flag': flag,
        'td_prob': round(td_prob, 1) if td_prob is not None else None
    }

# Interface
player = st.text_input(f"Enter {position} Name (e.g., {'Jonathan Taylor' if position=='RB' else 'Jaxon Smith-Njigba' if position=='WR' else 'Trey McBride' if position=='TE' else 'Sam Darnold'})", 
                       {'RB': 'Jonathan Taylor', 'WR': 'Jaxon Smith-Njigba', 'TE': 'Trey McBride', 'QB': 'Sam Darnold'}.get(position, 'Player'))
opponent = st.text_input("Enter Opponent Team (e.g., Atlanta Falcons)", "Atlanta Falcons")
betting_line = st.number_input(f"O/U {'Yards' if metric=='Yards' else 'TDs'}", 
                              value=100.0 if metric=='Yards' and position in ['RB','WR','TE'] else 250.0 if metric=='Yards' else 0.5, 
                              step=5.0 if metric=='Yards' else 0.5)

df_player = df_pass if position == 'QB' else df_rush if position == 'RB' else df_rec
is_rush = position == 'RB'
is_rec = position in ['WR', 'TE']

if st.button(f"Generate Week 6 {position} {metric} Projection"):
    result = project_player(player, opponent, betting_line, df_player, df_def, is_rush, is_rec, metric)
    if isinstance(result, dict):
        st.subheader(f"Week 6 {metric} Projection for {result['player']} ({position}) vs {result['opponent']}{result['flag']}")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"Projected {metric}", result['projected_value'])
        col2.metric("Edge vs O/U", f"{result['edge_percent']}%")
        col3.metric("Matchup Score", f"{result['matchup_score']}/10")
        if metric == 'Touchdowns':
            col2.metric("P(TD≥1)", f"{result['td_prob']}%")
        if result['edge_percent'] > 10:
            st.success(f"xEP-Style Value Bet: {player} O{betting_line} {'Yards' if metric=='Yards' else 'TDs'}!")
        if result['matchup_score'] > 7:
            st.info(f"Favorable matchup—weak {'rush' if is_rush else 'pass'} defense per Week 5.")
    else:
        st.error(result)

# xEP Leaderboard (Week 5 insights)
st.subheader("xEP Refresh: Top Players Through Week 5")
col_a, col_b = st.columns(2)
with col_a:
    st.write("**RBs (YPA / Rush TD)**: Bijan Robinson (5.8 / ~0.8), Rico Dowdle (5.8 / ~0.6), Travis Etienne (5.3 / ~0.7)")
    st.write("**WRs/TEs (YPT / Rec TD)**: J. Smith-Njigba (~7.5 / ~0.5), Trey McBride (~7.2 / ~0.6)")
with col_b:
    st.write("**QBs (CPOE% / Pass TD)**: Jared Goff (11.5% / ~2.0), Sam Darnold (11.2% / ~1.8), Drake Maye (9.7% / ~1.5)")
    st.write("Full insights: @xEP_Network or xep.ai.")
