'''
This file contains a simple animation demo using mplfinance.

Imagine that we make an API request where we receive the latest asset data; up to the present moment,
create the Range Chart from it, and save it in a file.
Then we make a websocket connection to this same asset and use
the previous Range Chart Data as a starting point, without creating the chart from scratch!

Useful when you need large initial data to calculate/plot your technical indicators.
'''

import mplfinance as mpf
import pandas as pd
from matplotlib import animation

from rangedf import RangeWS, Range

df_ticks = pd.read_parquet('data/US30_T1_cT.parquet')
df_ticks.rename(columns={'bid': 'close'}, inplace=True)
df_ticks['timestamp'] = pd.DatetimeIndex(df_ticks.index).asi8 // 10 ** 6  # Datetime to Timestamp (ns to ms)

df_GET = df_ticks.loc[(df_ticks.index <= '2025-05-15 10:50')]
df_ticks = df_ticks.loc[(df_ticks.index >= '2025-05-15 10:50')]

# API request
r = Range(df_GET, range_size=15)
ext_df = r.to_rws()  # Save this

# Load the file and chosen its Range Mode
r = RangeWS(external_df=ext_df, external_mode='normal')
initial_df = r.initial_df
# if you need multiple dataframes of different modes as initial_df
# just add 's' to get this function:
# r.initial_dfs('normal')
# r.initial_dfs('nongap')
# etc...

fig, axes = mpf.plot(initial_df, returnfig=True, volume=True,
                    figsize=(11, 8), panel_ratios=(2, 1),
                    title='\nUS30', type='candle', style='yahoo')
ax1 = axes[0]
ax2 = axes[2]

mpf.plot(initial_df,type='candle',ax=ax1,volume=ax2,axtitle='range: normal')

def animate(ival):

    if (0 + ival) >= len(df_ticks):
        print('no more data to plot')
        ani.event_source.interval *= 3
        if ani.event_source.interval > 12000:
            exit()
        return

    timestamp = df_ticks['timestamp'].iat[(0 + ival)]
    price = df_ticks['close'].iat[(0 + ival)]

    r.add_prices(timestamp, price)

    df_range = r.range_animate(max_len=1000, keep=500)

    ax1.clear()
    ax2.clear()

    mpf.plot(df_range, type='candle', ax=ax1, volume=ax2, axtitle='range: normal')


ani = animation.FuncAnimation(fig, animate, interval=80)
mpf.show()