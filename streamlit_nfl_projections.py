import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

st.title("NFL Player Props Dashboard (2025 – Week 5)")
st.write("Projects Yards & TDs for QBs, RBs, WRs, TEs. Inspired by @xEP_Network's ATD Ratings & efficiency (e.g., Bijan Robinson 5.8 YPA).")

# Sidebar: Position & Metric
position = st.sidebar.selectbox("Select Position", ["QB", "RB", "WR", "TE"])
metric = st.sidebar.selectbox("Select Metric", ["Yards", "Touchdowns"])
st.sidebar.write("Week 5 data from PFR. Built for Week 6 projections.")

# Step 1: Data loading functions (cached)
@st.cache_data
def load_passing_data():
    url = "https://www.pro-football-reference.com/years/2025/passing.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'passing'})
    if not table:
        st.error("Passing data scrape failed.")
        return pd.DataFrame()
    df = pd.read_html(str(table))[0]
    df = df[df['Player'] != 'Player'].dropna(subset=['Player', 'Tm', 'G', 'Yds', 'Att', 'TD'])
    df['Yds'] = pd.to_numeric(df['Yds'], errors='coerce')
    df['Att'] = pd.to_numeric(df['Att'], errors='coerce')
    df['TD'] = pd.to_numeric(df['TD'], errors='coerce')
    df['G'] = pd.to_numeric(df['G'], errors='coerce')
    df['YPA'] = df['Yds'] / df['Att']
    df['TDPerGame'] = df['TD'] / df['G']
    df = df.groupby('Player').agg({'Yds': 'sum', 'Att': 'sum', 'TD': 'sum', 'G': 'max', 'YPA': 'mean', 'TDPerGame': 'mean'}).reset_index()
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

@st.cache_data
def load_rushing_data():
    url = "https://www.pro-football-reference.com/years/2025/rushing.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'rushing'})
    if not table:
        st.error("Rushing data scrape failed.")
        return pd.DataFrame()
    df = pd.read_html(str(table))[0]
    df = df[df['Player'] != 'Player'].dropna(subset=['Player', 'Tm', 'G', 'Yds', 'Att', 'TD'])
    df['Yds'] = pd.to_numeric(df['Yds'], errors='coerce')
    df['Att'] = pd.to_numeric(df['Att'], errors='coerce')
    df['TD'] = pd.to_numeric(df['TD'], errors='coerce')
    df['G'] = pd.to_numeric(df['G'], errors='coerce')
    df['YPA'] = df['Yds'] / df['Att']
    df['TDPerGame'] = df['TD'] / df['G']
    df = df.groupby('Player').agg({'Yds': 'sum', 'Att': 'sum', 'TD': 'sum', 'G': 'max', 'YPA': 'mean', 'TDPerGame': 'mean'}).reset_index()
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

@st.cache_data
def load_receiving_data():
    url = "https://www.pro-football-reference.com/years/2025/receiving.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'receiving'})
    if not table:
        st.error("Receiving data scrape failed.")
        return pd.DataFrame()
    df = pd.read_html(str(table))[0]
    df = df[df['Player'] != 'Player'].dropna(subset=['Player', 'Tm', 'G', 'Yds', 'Tgt', 'TD'])
    df['Yds'] = pd.to_numeric(df['Yds'], errors='coerce')
    df['Tgt'] = pd.to_numeric(df['Tgt'], errors='coerce')
    df['TD'] = pd.to_numeric(df['TD'], errors='coerce')
    df['G'] = pd.to_numeric(df['G'], errors='coerce')
    df['YPT'] = df['Yds'] / df['Tgt']
    df['TDPerGame'] = df['TD'] / df['G']
    df = df.groupby('Player').agg({'Yds': 'sum', 'Tgt': 'sum', 'TD': 'sum', 'G': 'max', 'YPT': 'mean', 'TDPerGame': 'mean'}).reset_index()
    df['YdsPerGame'] = df['Yds'] / df['G']
    return df

@st.cache_data
def load_def_data():
    url_def = "https://www.pro-football-reference.com/years/2025/opp.htm"
    response_def = requests.get(url_def)
    soup_def = BeautifulSoup(response_def.content, 'html.parser')
    table_def = soup_def.find('table', {'id': 'team_stats'})
    if not table_def:
        st.error("Defensive data scrape failed.")
        return pd.DataFrame()
    df_def = pd.read_html(str(table_def))[0]
    df_def = df_def[df_def['Tm'] != 'League Total'].dropna(subset=['Tm'])
    df_def['RushYdsAllowed'] = pd.to_numeric(df_def['Yds'], errors='coerce')
    df_def['PassYdsAllowed'] = pd.to_numeric(df_def['Yds.1'], errors='coerce')
    # Approximate TDs allowed (PFR doesn't split pass/rush TDs; assume ratio from total)
    df_def['TotalTDAllowed'] = pd.to_numeric(df_def.get('TD', 0), errors='coerce')  # Total TDs allowed
    df_def['RushTDAllowedPerGame'] = (df_def['TotalTDAllowed'] * 0.4) / 5  # ~40% rush TDs (industry avg)
    df_def['PassTDAllowedPerGame'] = (df_def['TotalTDAllowed'] * 0.6) / 5  # ~60% pass TDs
    df_def['RushYdsAllowedPerGame'] = df_def['RushYdsAllowed'] / 5
    df_def['PassYdsAllowedPerGame'] = df_def['PassYdsAllowed'] / 5
    return df_def

# Load data
df_pass = load_passing_data()
df_rush = load_rushing_data()
df_rec = load_receiving_data()
df_def = load_def_data()

# Step 2: Generalized projection function
def project_player(player, opponent, betting_line, df_player, df_def, is_rush=False, is_rec=False, metric='Yards'):
    player_data = df_player[df_player['Player'].str.contains(player, case=False, na=False)]
    if player_data.empty:
        return f"Error: No 2025 Week 5 data for {player}"
    
    # Select defense metric
    def_yards_metric = 'RushYdsAllowedPerGame' if is_rush else 'PassYdsAllowedPerGame'
    def_td_metric = 'RushTDAllowedPerGame' if is_rush else 'PassTDAllowedPerGame'
    opp_data = df_def[df_def['Tm'].str.contains(opponent, case=False, na=False)]
    if opp_data.empty:
        return f"Error: No data for opponent {opponent}"
    
    # Yards or TDs
    if metric == 'Yards':
        eff_col = 'YPA' if not is_rec else 'YPT'
        target = 'YdsPerGame'
        def_metric = def_yards_metric
    else:  # TDs
        eff_col = 'YPA' if not is_rec else 'YPT'
        target = 'TDPerGame'
        def_metric = def_td_metric
    
    # Features: Efficiency (YPA/YPT), Target (Yds/TD per game), Opp Defense
    X = np.array([[player_data[eff_col].iloc[0], player_data[target].iloc[0], 
                   opp_data[def_metric].iloc[0]]])
    y = player_data[target].values
    
    model = LinearRegression()
    model.fit(X, y)
    projection = model.predict(X)[0]
    
    # Edge % (for TDs, use O/U 0.5 for ATD prop)
    edge = ((projection - betting_line) / betting_line) * 100
    
    # Matchup Score (1-10, higher = weaker D)
    df_def_copy = df_def.copy()
    df_def_copy['DefRank'] = df_def_copy[def_metric].rank(ascending=True, pct=True)
    opp_rank = df_def_copy[df_def_copy['Tm'].str.contains(opponent, case=False, na=False)]['DefRank'].iloc[0]
    matchup_score = 10 - (opp_rank * 9)
    
    # TD Probability (xEP-style ATD)
    td_prob = poisson.cdf(0, projection) if metric == 'Touchdowns' else None  # P(TD=0)
    td_prob = (1 - td_prob) * 100 if td_prob is not None else None  # P(TD≥1)
    
    # xEP-Style Flag
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

# Step 3: Interface
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

# Step 4: xEP Leaderboard
st.subheader("xEP Refresh: Top Players Through Week 5")
col_a, col_b = st.columns(2)
with col_a:
    st.write("**RBs (YPA / Rush TD)**: Bijan Robinson (5.8 / ~0.8), Rico Dowdle (5.8 / ~0.6), Travis Etienne (5.3 / ~0.7)")
    st.write("**WRs/TEs (YPT / Rec TD)**: J. Smith-Njigba (~7.5 / ~0.5), Trey McBride (~7.2 / ~0.6)")
with col_b:
    st.write("**QBs (CPOE% / Pass TD)**: Jared Goff (11.5% / ~2.0), Sam Darnold (11.2% / ~1.8), Drake Maye (9.7% / ~1.5)")
    st.write("Full insights: @xEP_Network or xep.ai.")
