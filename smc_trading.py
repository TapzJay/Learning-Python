import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define settings as a dictionary to control the script's behavior and appearance
settings = {
    'BULLISH_LEG': 1,           # Indicates a bullish swing direction
    'BEARISH_LEG': 0,           # Indicates a bearish swing direction
    'BULLISH_BIAS': 1,          # Bias for bullish structures/order blocks
    'BEARISH_BIAS': -1,         # Bias for bearish structures/order blocks
    'GREEN': '#089981',         # Color for bullish indicators
    'RED': '#F23645',           # Color for bearish indicators
    'show_structure': True,     # Toggle to display BOS/CHoCH structures
    'show_swing_order_blocks': True,  # Toggle to display swing order blocks
    'swings_length': 50,        # Lookback period for swing detection
    'internal_order_blocks_size': 5,  # Size for internal order blocks (not implemented here)
    'swing_order_blocks_size': 5,     # Size for swing order blocks (not implemented here)
}

# Define classes to store pivot and order block data
class Pivot:
    def __init__(self, level, timestamp, index):
        """Initialize a Pivot object.
        
        Args:
            level (float): Price level of the pivot
            timestamp (datetime): Timestamp of the pivot
            index (int): Index position in the data
        """
        self.level = level
        self.timestamp = timestamp
        self.index = index
        self.crossed = False  # Tracks if the pivot has been crossed by price

class OrderBlock:
    def __init__(self, high, low, timestamp, bias):
        """Initialize an OrderBlock object.
        
        Args:
            high (float): High price of the order block
            low (float): Low price of the order block
            timestamp (datetime): Starting timestamp of the order block
            bias (int): Bullish (1) or Bearish (-1) bias
        """
        self.high = high
        self.low = low
        self.timestamp = timestamp
        self.bias = bias

# Define functions to process the data
def leg(data, size):
    """Determine the direction of the swing (leg) at each bar.
    
    Args:
        data (pd.DataFrame): OHLC data with 'High' and 'Low' columns
        size (int): Lookback period for swing detection
    
    Returns:
        list: List of leg values (1 for bullish, 0 for bearish)
    """
    leg_values = [settings['BEARISH_LEG']] * len(data)
    for n in range(size, len(data)):
        # Ensure scalar comparisons
        high_n_size = data['High'].iloc[n - size]
        high_max = data['High'].iloc[n - size + 1:n + 1].max()
        low_n_size = data['Low'].iloc[n - size]
        low_min = data['Low'].iloc[n - size + 1:n + 1].min()
        
        new_leg_high = high_n_size > high_max
        new_leg_low = low_n_size < low_min
        
        if new_leg_high:
            leg_values[n] = settings['BEARISH_LEG']  # Bearish leg starts after a high
        elif new_leg_low:
            leg_values[n] = settings['BULLISH_LEG']  # Bullish leg starts after a low
        else:
            leg_values[n] = leg_values[n - 1]
    return leg_values

def get_current_structure(data, size):
    """Identify pivot points based on swing changes.
    
    Args:
        data (pd.DataFrame): OHLC data
        size (int): Lookback period for swings
    
    Returns:
        list: List of Pivot objects
    """
    leg_values = leg(data, size)
    pivots = []
    for n in range(size, len(data)):
        if leg_values[n] != leg_values[n - 1]:
            if leg_values[n] == settings['BULLISH_LEG']:
                pivot_level = data['Low'].iloc[n - size]  # Pivot low
                pivots.append(Pivot(pivot_level, data.index[n - size], n - size))
            elif leg_values[n] == settings['BEARISH_LEG']:
                pivot_level = data['High'].iloc[n - size]  # Pivot high
                pivots.append(Pivot(pivot_level, data.index[n - size], n - size))
    return pivots

def process_data(data, settings):
    """Process the data to detect structures and order blocks.
    
    Args:
        data (pd.DataFrame): OHLC data
        settings (dict): Configuration settings
    
    Returns:
        tuple: (list of structures, list of order blocks)
    """
    pivots = get_current_structure(data, settings['swings_length'])
    structures = []  # Stores BOS and CHoCH events
    order_blocks = []  # Stores order block ranges
    trend_bias = 0  # 0: neutral, 1: bullish, -1: bearish

    for i in range(settings['swings_length'], len(data)):
        # Find the most recent pivot high and low before the current bar
        pivot_highs = [p for p in pivots if p.level == data['High'].iloc[p.index] and p.index < i]
        pivot_lows = [p for p in pivots if p.level == data['Low'].iloc[p.index] and p.index < i]
        current_swing_high = max(pivot_highs, key=lambda p: p.index) if pivot_highs else None
        current_swing_low = min(pivot_lows, key=lambda p: p.index) if pivot_lows else None

        close = data['Close'].iloc[i]
        
        # Bullish structure detection (Break of Structure or Change of Character)
        if current_swing_high and close > current_swing_high.level and not current_swing_high.crossed:
            tag = 'CHoCH' if trend_bias == settings['BEARISH_BIAS'] else 'BOS'
            trend_bias = settings['BULLISH_BIAS']
            current_swing_high.crossed = True
            structures.append((tag, current_swing_high.level, data.index[i], settings['BULLISH_BIAS']))
            
            if settings['show_swing_order_blocks']:
                # Find the bar with the lowest low between pivot and current bar
                slice_data = data.iloc[current_swing_high.index:i]
                min_low_idx = slice_data['Low'].idxmin()
                ob_bar = data.loc[min_low_idx]
                ob = OrderBlock(ob_bar['High'], ob_bar['Low'], min_low_idx, settings['BULLISH_BIAS'])
                order_blocks.append(ob)
        
        # Bearish structure detection
        if current_swing_low and close < current_swing_low.level and not current_swing_low.crossed:
            tag = 'CHoCH' if trend_bias == settings['BULLISH_BIAS'] else 'BOS'
            trend_bias = settings['BEARISH_BIAS']
            current_swing_low.crossed = True
            structures.append((tag, current_swing_low.level, data.index[i], settings['BEARISH_BIAS']))
            
            if settings['show_swing_order_blocks']:
                # Find the bar with the highest high between pivot and current bar
                slice_data = data.iloc[current_swing_low.index:i]
                max_high_idx = slice_data['High'].idxmax()
                ob_bar = data.loc[max_high_idx]
                ob = OrderBlock(ob_bar['High'], ob_bar['Low'], max_high_idx, settings['BEARISH_BIAS'])
                order_blocks.append(ob)

    return structures, order_blocks

def plot_smart_money_concepts(data, settings):
    """Plot the price data with SMC structures and order blocks.
    
    Args:
        data (pd.DataFrame): OHLC data
        settings (dict): Configuration settings
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data.index, data['Close'], label='Close', color='black')

    structures, order_blocks = process_data(data, settings)

    # Plot structures (BOS and CHoCH)
    for tag, level, timestamp, bias in structures:
        color = settings['GREEN'] if bias == settings['BULLISH_BIAS'] else settings['RED']
        ax.axhline(y=level, color=color, linestyle='--', label=tag)
        ax.text(timestamp, level, tag, color=color)

    # Plot order blocks as horizontal spans
    for ob in order_blocks:
        color = settings['GREEN'] if ob.bias == settings['BULLISH_BIAS'] else settings['RED']
        ax.axhspan(ob.low, ob.high, xmin=data.index.get_loc(ob.timestamp) / len(data.index),
                   facecolor=color, alpha=0.2)

    plt.legend()
    plt.title('Smart Money Concepts Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

# Fetch data from Yahoo Finance
ticker = "EURUSD=X"
data = yf.download(ticker, start="2024-06-01", end="2025-03-01", interval="1d")

# Simplify MultiIndex columns if single ticker
if isinstance(data.columns, pd.MultiIndex):
    tickers = set(data.columns.get_level_values(1))
    if len(tickers) == 1:
        data.columns = data.columns.droplevel(1)
    else:
        raise ValueError("Multiple tickers detected. This code assumes a single ticker.")
else:
    pass

# Select the desired columns
data = data[['Open', 'High', 'Low', 'Close']]

# Execute the plotting function
plot_smart_money_concepts(data, settings)