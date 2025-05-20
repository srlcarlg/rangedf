"""
rangedf
=====
Transform Tick Data into OHLCV Range Dataframe!
"""
import gc

import numpy as np
import pandas as pd
import mplfinance as mpf

_MODE_dict = ['normal', 'nongap']


class Range:
    def __init__(self, df: pd.DataFrame, range_size: float, keep_inner_gap: bool = False, add_columns: list = None, show_progress: bool = False):
        """
        Create Range OHLCV dataframe with existing Ticks data.

        Usage
        ------
        >> from rangedf import Range \n
        >> r = Range(df_ticks, range_size) \n
        >> df = r.rangedf() \n

        Parameters
        ----------
        df : dataframe
            Only two columns are required:

            * "close": Mandatory.
            * "datetime": If is not present, the index will be used.

        range_size : float
            Cannot be less than or equal to 0.00...
        keep_inner_gap : bool
            if True, a price expansion of the range bar will occur if any gaps happens during its formation
            Useful to reduce the gaps between bars as much as possible
        add_columns : list
            A list of strings(column names) to be added to the final result, such as spread, quantity, etc.
        show_progress : bool
            A self-explanatory percentage number from 0 to 100;
            The performance will be affected by 2x if it's True
        """

        if range_size is None or range_size <= 0:
            raise ValueError("range_size cannot be 'None' or '<= 0'")
        if 'datetime' not in df.columns:
            df["datetime"] = df.index
        if 'close' not in df.columns:
            raise ValueError("Column 'close' doesn't exist!")
        if add_columns is not None:
            if not set(add_columns).issubset(df.columns):
                raise ValueError(f"One or more of {add_columns} columns don't exist!")

        self._range_size = range_size
        self._custom_columns = add_columns
        self._df_len = len(df["close"])
        self._keep_inner_gap = keep_inner_gap
        self._show_progress = show_progress

        first_close = df["close"].iat[0]
        initial_price = (first_close // range_size) * range_size
        # Range Single Data
        self._rsd = {
            "origin_index": [0],
            "date": [df["datetime"].iat[0]],
            "price": [initial_price],
            "direction": [0],
            "wick": [initial_price],
            "close": [initial_price],
            "volume": [1],
        }
        if add_columns is not None:
            for name in add_columns:
                self._rsd.update({
                    name: [df[name].iat[0]]
                })

        self._wick_min_i = initial_price
        self._wick_max_i = initial_price
        self._volume_i = 1
        self._open_price_i = 0.0

        for i in range(1, self._df_len):
            self._add_prices(i, df)

    def _add_prices(self, i, df):
        """
        Determine if there are new bricks to add according to the current (loop) price relative to the previous range.

        Here, the 'Range Single Data' is constructed.
        """
        df_close = df["close"].iat[i]
        self._open_price_i = df_close if self._open_price_i == 0.0 else self._open_price_i
        self._wick_min_i = df_close if df_close < self._wick_min_i else self._wick_min_i
        self._wick_max_i = df_close if df_close > self._wick_max_i else self._wick_max_i
        self._volume_i += 1

        current_direction = 1 if df_close > self._open_price_i else -1 if df_close < self._open_price_i else 0
        if (self._wick_max_i - self._wick_min_i) <= self._range_size:
            return

        range_no_gap = self._wick_min_i + self._range_size if current_direction > 0 else self._wick_max_i - self._range_size

        # keep_inner_gap = True, a price expansion of the range will occur if a gap occurs, there's no limit for this gap
        # keep_inner_gap = False, the range bar will always have the same range price, the current tick (this gap)
        #  will be the next range-bar's open price
        self._add_rangebar_loop(i, df, range_no_gap if self._keep_inner_gap is False else df_close, current_direction)

        if self._show_progress:
            print(f"\r {round(float((i + 1) / self._df_len * 100), 2)}%", end='')

    def _add_rangebar_loop(self, i, df, close_price, current_direction):
        wick = self._wick_min_i if current_direction > 0 else self._wick_max_i
        to_add = [i, df["datetime"].iat[i], self._open_price_i, current_direction, wick, close_price, self._volume_i]
        for name, add in zip(list(self._rsd.keys()), to_add):
            self._rsd[name].append(add)
        if self._custom_columns is not None:
            for name in self._custom_columns:
                self._rsd[name].append(df[name].iat[i])

        self._volume_i = 1
        self._open_price_i = df["close"].iat[i]
        self._wick_min_i = df["close"].iat[i]
        self._wick_max_i = df["close"].iat[i]

    def plot(self, mode: str = "normal", volume: bool = True, df: pd.DataFrame = None, add_plots: [] = None):
        """
        Redundant function only to plot the Range Chart with fewer lines of code. \n
        If parameter "df" is used: the "add_plots" is mandatory. \n
        If parameter "df" is empty: only the 'range_df' of the current instance will be plotted.

        Notes
        -----
        This function is equivalent to: \n
        > mpf.plot(df, type='candle', volume=True, style="charles") \n
        > mpf.plot(df, type='candle', volume=True, style="charles", addplot=[]) \n
        > mpf.show() \n

        Parameters
        ----------
        mode : str
            The method for building the range dataframe, described in the function 'range_df'.
        volume : bool
            Plot with Volume or not.
        df : dataframe
            Modified external dataframe, usually with new columns for plotting indicators, signals, etc.
        add_plots : list
            A list with instances of mpf.make_addplot().
        """
        if df is not None and add_plots is None:
            raise ValueError("If 'df' parameter is used, 'add_plots' is mandatory!")

        if df is not None:
            mpf.plot(df, type='candle', style="charles", volume=volume, addplot=add_plots,
                     title=f"\n range: {mode} \range size: {self._range_size}")
        else:
            df_range = self.range_df(mode)
            mpf.plot(df_range, type='candle', style="charles", volume=volume,
                     title=f"\n range: {mode} \range size: {self._range_size}")

        return mpf.show()

    def range_df(self, mode: str = "normal"):
        """
        Transforms 'Range Single Data' into OHLCV Dataframe.

        Parameters
        ----------
        mode : str
            The method for building the range dataframe, there are 2 modes available:

              * "normal" : Standard Range. (default)
              * "nongap": Standard Range but the OPEN will have the same value as the respective wick.

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

        dates = self._rsd["date"]
        open_prices = self._rsd["price"]
        directions = self._rsd["direction"]
        wicks = self._rsd["wick"]
        closes = self._rsd["close"]
        volumes = self._rsd["volume"]

        df_dict = {
            "datetime": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        if self._custom_columns is not None:
            for name in self._custom_columns:
                df_dict.update({
                    name: []
                })

        index = 0
        for open_price, direction, date, wick, close, volume in zip(open_prices, directions, dates, wicks, closes, volumes):
            if direction != 0:
                df_dict["datetime"].append(date)
                df_dict["close"].append(close)
                df_dict["volume"].append(volume)

            # Current Range (UP)
            if direction == 1.0:
                df_dict["high"].append(close)
                df_dict["open"].append(wick if mode == "nongap" else open_price)
                df_dict["low"].append(wick)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(self._rsd[name][index])

            # Current Range (DOWN)
            elif direction == -1.0:
                df_dict["low"].append(close)
                df_dict["open"].append(wick if mode == "nongap" else open_price)
                df_dict["high"].append(wick)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(self._rsd[name][index])
            # BEGIN OF DICT
            else:
                df_dict["datetime"].append(np.NaN)
                df_dict["low"].append(np.NaN)
                df_dict["close"].append(np.NaN)
                df_dict["high"].append(np.NaN)
                df_dict["open"].append(np.NaN)
                df_dict["volume"].append(np.NaN)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(np.NaN)
            index += 1


        df = pd.DataFrame(df_dict)
        # Removing the first 2 lines of DataFrame that are the beginning of respective loops (df_dict and self._rsd)
        df.drop(df.head(2).index, inplace=True)
        # Setting Index
        df.index = pd.DatetimeIndex(df["datetime"])
        df.drop(columns=['datetime'], inplace=True)

        return df

    def to_rws(self, use_iloc: int = None):
        """
        Transforms 'Range Single Data' into a Dataframe,
        which can be used as initial data in the 'RangeWS' class. \n
        The DatetimeIndex will be converted to Timestamp (from nanoseconds to milliseconds)

        Parameters
        ----------
        use_iloc : int
            * If positive: First nº rows will be returned
            * If negative: Last nº rows will be returned

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        rws = pd.DataFrame(self._rsd)
        rws.index = pd.DatetimeIndex(rws["date"]).asi8 // 10 ** 6  # Datetime to Timestamp (ns to ms)
        rws.index.name = 'timestamp'
        rws['timestamp'] = rws.index
        rws['range_size'] = self._range_size

        if use_iloc is not None:
            if use_iloc < 0:
                return rws.iloc[use_iloc:]
            else:
                return rws.iloc[:use_iloc]
        else:
            return rws


class RangeWS:

    def __init__(self, ws_timestamp: int = None, ws_price: float = None,
                 range_size: float = None, keep_inner_gap: bool = False,
                 external_df: pd.DataFrame = None, external_mode: str = 'normal'):
        """
        Create real-time Range charts, usually over a WebSocket connection.

        Usage
        -----
        >> from rangedf import RangeWS \n
        >> r = RangeWS(your combination) \n
        >> # At every price change \n
        >> r.add_prices(ws_timestamp, ws_price) \n
        >> df = r.range_animate() \n

        Notes
        -----
        Only the following combinations are possible: \n
        > RangeWS(ws_timestamp, ws_price, range_size) \n
        > RangeWS(external_df, external_mode) \n

        Parameters
        ----------
        ws_timestamp : int
            Timestamp in milliseconds.
        ws_price : float
            Self-explanatory.
        range_size : float
            Cannot be less than or equal to 0.00...
        keep_inner_gap : bool
            if True, a price expansion of the range bar will occur if any gaps happens during its formation
            Useful to reduce the gaps between bars as much as possible
        external_df : dataframe
            The dataframe made from Range.to_rws()
        external_mode : str
            The method for building the external range df, described in the 'Range.range_df()'.
        """
        if external_df is None:
            if range_size is None or range_size <= 0:
                raise ValueError("range_size cannot be 'None' or '<= 0'")
            if ws_price is None:
                raise ValueError("ws_price cannot be 'None'")
            if ws_timestamp is None:
                raise ValueError("ws_timestamp cannot be 'None'")

        self._range_size = range_size if external_df is None else external_df['range_size'].iat[0]

        initial_price = 0.0
        if external_df is None:
            initial_price = (ws_price // self._range_size) * self._range_size
            self._rsd = {
                "timestamp": [ws_timestamp],
                "price": [initial_price],
                "direction": [0],
                "wick": [initial_price],
                "close": [initial_price],
                "volume": [1],
            }
        else:
            self._rsd = {
                "timestamp": external_df['timestamp'].to_list(),
                "price": external_df['price'].to_list(),
                "direction": external_df['direction'].to_list(),
                "wick": external_df['wick'].to_list(),
                "close": external_df['close'].to_list(),
                "volume": external_df['volume'].to_list(),
            }

        if external_df is None:
            initial_df = {
                "timestamp": [ws_timestamp],
                "open": [initial_price],
                "high": [initial_price],
                "low": [initial_price],
                "close": [initial_price],
                "volume": [1]
            }
            initial_df = pd.DataFrame(initial_df, columns=["timestamp", "open", "high", "low", "close", "volume"])
            initial_df.index = pd.DatetimeIndex(
                pd.to_datetime(initial_df["timestamp"].values.astype(np.int64), unit="ms"))
            initial_df.drop(columns=['timestamp'], inplace=True)
        else:
            initial_df = self._range_df(external_mode)

        self.initial_df = initial_df
        self._keep_inner_gap = keep_inner_gap
        # For loop
        self._volume_i = 1
        self._wick_min_i = initial_price if external_df is None else external_df['price'].iat[-1]
        self._wick_max_i = initial_price if external_df is None else external_df['price'].iat[-1]
        self._open_price_i = 0.0

        self._ws_timestamp = ws_timestamp
        self._ws_price = ws_price

    def initial_dfs(self, mode: str = 'normal'):
        return self._range_df(mode)

    def add_prices(self, ws_timestamp: int, ws_price: float):
        """
        Determine if there are new bricks to add according to the current price relative to the previous range.

        Must be called at every price change.

        Here, the 'Range Single Data' is constructed.

        Parameters
        ----------
        ws_timestamp : int
            Timestamp in milliseconds.
        ws_price : float
            Self-explanatory.
        """
        self._ws_timestamp = ws_timestamp
        self._ws_price = ws_price

        self._open_price_i = ws_price if self._open_price_i == 0.0 else self._open_price_i
        self._wick_min_i = ws_price if ws_price < self._wick_min_i else self._wick_min_i
        self._wick_max_i = ws_price if ws_price > self._wick_max_i else self._wick_max_i
        self._volume_i += 1

        current_direction = 1 if ws_price > self._open_price_i else -1 if ws_price < self._open_price_i else 0
        if (self._wick_max_i - self._wick_min_i) <= self._range_size:
            return

        range_no_gap = self._wick_min_i + self._range_size if current_direction > 0 else self._wick_max_i - self._range_size
        # keep_inner_gap = True, a price expansion of the range will occur if a 'small' gap occurs
        # keep_inner_gap = False, the range bar will always have the same range price, the current tick (this gap)
        #  will be the next range-bar's open price
        self._add_rangebar_loop(ws_timestamp, ws_price, range_no_gap if self._keep_inner_gap is False else ws_price, current_direction)


    def _add_rangebar_loop(self, ws_timestamp, ws_price, close_price, current_direction):
        wick = self._wick_min_i if current_direction > 0 else self._wick_max_i
        to_add = [ws_timestamp, self._open_price_i, current_direction, wick, close_price, self._volume_i]
        for name, add in zip(list(self._rsd.keys()), to_add):
            self._rsd[name].append(add)

        self._volume_i = 1
        self._open_price_i = ws_price
        self._wick_min_i = ws_price
        self._wick_max_i = ws_price

    def _range_df(self, mode: str = "normal"):
        """
        Transforms 'Range Single Data' into OHLCV Dataframe.
        """
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

        timestamps = self._rsd["timestamp"]
        open_prices = self._rsd["price"]
        directions = self._rsd["direction"]
        wicks = self._rsd["wick"]
        closes = self._rsd["close"]
        volumes = self._rsd["volume"]

        df_dict = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }

        for open_price, direction, timestamp, wick, close, volume in zip(open_prices, directions, timestamps, wicks, closes,
                                                                    volumes):
            if direction != 0:
                df_dict["timestamp"].append(timestamp)
                df_dict["close"].append(close)
                df_dict["volume"].append(volume)

            # Current Range (UP)
            if direction == 1.0:
                df_dict["high"].append(close)
                df_dict["open"].append(wick if mode == "nongap" else open_price)
                df_dict["low"].append(wick)

            # Current Range (DOWN)
            elif direction == -1.0:
                df_dict["low"].append(close)
                df_dict["open"].append(wick if mode == "nongap" else open_price)
                df_dict["high"].append(wick)

            # BEGIN OF DICT
            else:
                df_dict["timestamp"].append(np.NaN)
                df_dict["low"].append(np.NaN)
                df_dict["close"].append(np.NaN)
                df_dict["high"].append(np.NaN)
                df_dict["open"].append(np.NaN)
                df_dict["volume"].append(np.NaN)

        df = pd.DataFrame(df_dict)
        # Removing the first 2 lines of DataFrame that are the beginning of respective loops (df_dict and self._rsd)
        df.drop(df.head(2).index, inplace=True)
        # Setting Index
        df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"].values.astype(np.int64), unit="ms"))
        df.index.name = 'datetime'
        df.drop(columns=['timestamp'], inplace=True)

        return df

    def range_animate(self, mode: str = 'normal', max_len: int = 500, keep: int = 250):
        """
        Should be called after 'add_prices(ws_timestamp, ws_price)'

        Parameters
        ----------
        mode : str
            The method for building the range dataframe, described in the Range.range_df().
        max_len : int
            Once reached, the 'Single Range Data' values will be deleted.
        keep : int
            Keep last nº values after deletion.

        Returns
        -------
        x : dataframe
            pandas.Dataframe
        """
        range_df = self._range_df(mode)

        ws_timestamp = self._ws_timestamp
        ws_price = self._ws_price

        raw_ws = {
            "timestamp": [ws_timestamp],
            "open": [ws_price],
            "high": [ws_price],
            "low": [ws_price],
            "close": [ws_price],
            "volume": self._volume_i
        }
        length = len(range_df)
        if length < 1:
            raw_ws["open"][-1] = self.initial_df["close"].iat[-1]
            raw_ws["high"][-1] = self._wick_max_i
            raw_ws["low"][-1] = self._wick_min_i

            df_ws = pd.DataFrame(raw_ws)
            df_ws.index = pd.DatetimeIndex(pd.to_datetime(df_ws["timestamp"].values.astype(np.int64), unit="ms"))
            df_ws.index.name = 'datetime'
            df_ws.drop(columns=['timestamp'], inplace=True)

            return pd.concat([self.initial_df, df_ws])

        # Forming wick
        raw_ws["high"][-1] = self._wick_max_i if mode != 'nongap' else ws_price
        raw_ws["low"][-1] = self._wick_min_i if mode != 'nongap' else ws_price

        nongap_rule = mode in ['nongap']
        current_range_open = self._open_price_i if self._open_price_i != 0 else ws_price

        if ws_price > current_range_open:
            raw_ws["open"][-1] = self._wick_min_i if nongap_rule else current_range_open
        else:
            raw_ws["open"][-1] = self._wick_max_i if nongap_rule else current_range_open

        df_ws = pd.DataFrame(raw_ws)
        df_ws.index = pd.DatetimeIndex(pd.to_datetime(df_ws["timestamp"].values.astype(np.int64), unit="ms"))
        df_ws.index.name = 'datetime'
        df_ws.drop(columns=['timestamp'], inplace=True)

        if length >= max_len:
            # Cleaning dictionary, keeping keys and nº last values/bricks
            for value in self._rsd.values():
                del value[:-keep]
            gc.collect()

        return pd.concat([range_df, df_ws])
