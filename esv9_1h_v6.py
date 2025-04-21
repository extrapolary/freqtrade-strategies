# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# Chandelier
# from finta import TA as fta

# This class is a sample. Feel free to customize it.
class esv9_1h_v6(IStrategy):
    """
    This is a strategy made by Phil 

    This is a medium time frame swing strategy that buys when a hyperoptimized
    ema slope crosses above a certain threshold, and sells when the prices closes below
    another hyperopt'd ema, or on the roi table or trailing stoploss (parameter file)

    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    max_open_trades = 3

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.29700000000000004,
        "205": 0.198,
        "310": 0.07,
        "934": 0
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    # stoploss = -0.315

    stoploss = -0.05
    use_custom_stoploss = False
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        if current_profit > 0.02 and current_profit < 0.05:
            return 0.0

        if current_profit < 0.05:
            return -1 # return a value bigger than the inital stoploss to keep using the inital stoploss

        # After reaching the desired offset, allow the stoploss to trail by half the profit
        desired_stoploss = current_profit - 0.01

        # Use a minimum of 2.5% and a maximum of 5%
        # return max(min(desired_stoploss, 0.05), 0.025)
        return desired_stoploss 
    
    # Optimal timeframe for the strategy.
    timeframe = '1h'


    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_ema = IntParameter(low=3,high=50,default=9,space='buy',optimize=True, load=True)
    buy_ema_slope = DecimalParameter(low=-0.5, high=5.0, default=0.225, space='buy', optimize=True, load=True)
    # buy_aroonosc = IntParameter(low=-100,high=100,default=50,space='buy',optimize=True, load=True)

    buy_rsi_low = IntParameter(low=20, high=100, default=30, space='buy', optimize=True, load=True)
    buy_rsi_high = IntParameter(low=20, high=100, default=60, space='buy', optimize=True, load=True)
    #buy_rsi45_slope = DecimalParameter(low=0.000, high=0.500, default=0.010, space='buy', optimize=True, load=True)
    #buy_ema_momentum = DecimalParameter(low=0, high=5, default=0.04, space='buy', optimize=True, load=True)
    #buy_sma_slope = DecimalParameter(low=-0.01, high=2.5, default=0, space='buy', optimize=True, load=True)   # sma slope must be greater than value
    #buy_ema_slope_difference


    #sell_ema_momentum = DecimalParameter(low=-0.2, high=0.01, default=-0.01, space='sell', optimize=True, load=True)
    # sell_ema9_slope = DecimalParameter(low=-0.05,high=0.02, default = 0.00, space='sell',optimize=True, load=True)
    sell_ema = IntParameter(low=3,high=50,default=9,space='sell',optimize=True, load=True)
    
    #sell_rsi = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)

    #buy_uo = IntParameter(low=1, high=50, default=30, space='buy', optimize=True, load=True)
    #sell_uo = IntParameter(low=50, high=100, default=70, space='sell', optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 250

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        slopelength = 3 # had okay results with 5, trying 3

        # Calculate all ema_short values
        for val in self.buy_ema.range:
            dataframe[f'ema_{val}'] = ta.EMA(dataframe, timeperiod=val)
            dataframe[f'ema_{val}_slope'] = (ta.LINEARREG(dataframe[f'ema_{val}'], slopelength) / ta.SMA(dataframe[f'ema_{val}'], timeperiod=slopelength) -1) * 100 # this is a percent slope so it scales with mulitple stocks

        dataframe[f'ema_{self.sell_ema.value}'] = ta.EMA(dataframe, timeperiod=self.sell_ema.value)

        # Momentum Indicators
        # ------------------------------------
        
        # [dataframe['chandelier_short'], dataframe['chandelier_long']] = fta.CHANDELIER(dataframe)

        # chandelier from google's code
        dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'], length=22)
        dataframe['highest_high_22'] = dataframe['high'].rolling(window=22).max()
        dataframe['lowest_low_22'] = dataframe['low'].rolling(window=22).min()
        dataframe['chandelier_long'] = dataframe['highest_high_22'] - (dataframe['atr'] * 3)
        dataframe['chandelier_short'] = dataframe['lowest_low_22'] + (dataframe['atr'] * 3)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        dataframe['close'] = heikinashi['close']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['rsi14'] = ta.RSI(dataframe, timeperiod=14) 
        # dataframe['rsi14_slope'] = (ta.LINEARREG(dataframe['rsi14'], slopelength) / ta.SMA(dataframe['rsi14'], timeperiod=slopelength) -1) * 100

        # dataframe['rsi45'] = ta.RSI(dataframe, timeperiod=45) 
        # dataframe['rsi45_slope'] = (ta.LINEARREG(dataframe['rsi45'], slopelength) / ta.SMA(dataframe['rsi45'], timeperiod=slopelength) -1) * 100
        
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema9slope'] = (ta.LINEARREG(dataframe['ema9'], slopelength) / ta.SMA(dataframe['ema9'], timeperiod=slopelength) -1) * 100 # this is a percent slope so it scales with mulitple stocks
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['ema200slope'] = (ta.LINEARREG(dataframe['ema200'], slopelength) / ta.SMA(dataframe['ema200'], timeperiod=slopelength) -1) * 100 # this is a percent slope so it scales with mulitple stocks

        # dataframe['ema12'] = ta.EMA(dataframe, timeperiod=12)
        # dataframe['ema12_slope'] = (ta.LINEARREG(dataframe['ema12'], slopelength) / ta.SMA(dataframe['ema12'], timeperiod=slopelength) - 1) * 100
        dataframe['sma20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['sma20_slope'] = (ta.LINEARREG(dataframe['sma20'], slopelength) / ta.SMA(dataframe['sma20'], timeperiod=slopelength) - 1) * 100
        # dataframe['ema_momentum'] = dataframe['ema12_slope'] + dataframe['sma20_slope']
        # dataframe['ema_difference'] = dataframe['ema12'] - dataframe['sma20']
        # dataframe['ema_slope_difference'] = dataframe['ema12_slope'] - dataframe['sma20_slope']

        dataframe.loc[(qtpylib.crossed_above(dataframe['ema20'], dataframe['ema200'])),'ema cross'] = dataframe['close'] # for plotting purposes

        # dataframe['obv'] = ta.OBV(dataframe)
        # dataframe['obv_percent_change'] = (dataframe['obv'] - dataframe['obv'].shift(1)) / dataframe['obv'].shift(1)


        # ADX
        dataframe['adx'] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        [dataframe['aroondown'], dataframe['aroonup']] = ta.AROON(dataframe['high'], dataframe['low'], timeperiod=14)
        dataframe['aroonosc'] = ta.AROONOSC(dataframe['high'],dataframe['low'], timeperiod=14)

        # # Awesome Oscillator 
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # # Ultimate Oscillator
        dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)
        # dataframe['cci_slope'] = ta.LINEARREG(dataframe['cci'])

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_slope'] = ta.LINEARREG(dataframe['rsi'])


        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch['slowd']
        # dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        # stoch_fast = ta.STOCHF(dataframe)
        # dataframe['fastd'] = stoch_fast['fastd']
        # dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        #RSI
        # dataframe['s_rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        #StochRSI 
        # period = 14
        # smoothD = 3
        # SmoothK = 3
        # stochrsi  = (dataframe['s_rsi'] - dataframe['s_rsi'].rolling(period).min()) / (dataframe['s_rsi'].rolling(period).max() - dataframe['s_rsi'].rolling(period).min())
        # dataframe['srsi_k'] = stochrsi.rolling(SmoothK).mean() * 100
        # dataframe['srsi_d'] = dataframe['srsi_k'].rolling(smoothD).mean()

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Momentum
        # dataframe['momentum'] = ta.MOM(dataframe)
        # dataframe['momentum_slope'] = ta.LINEARREG(dataframe['momentum'])

        # MFI
        # dataframe['mfi'] = ta.MFI(dataframe)

        # Williams Percent R
        # dataframe['williams'] = ta.WILLR(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        # dataframe['bbema_diff'] = (dataframe['ema9'] - dataframe['bb_middleband']) / dataframe['ema9']


        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) /
        #     dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)

        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # dataframe['ema18'] = ta.EMA(dataframe, timeperiod = 18)
        # dataframe['ema78'] = ta.EMA(dataframe, timeperiod = 78)

        # # # SMA - Simple Moving Average
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)

        # dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        # dataframe['sar'] = ta.SAR(dataframe)

        # # TEMA - Triple Exponential Moving Average
        # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        # # Cycle Indicator
        # # ------------------------------------
        # # Hilbert Transform Indicator - SineWave
        # hilbert = ta.HT_SINE(dataframe)
        # dataframe['htsine'] = hilbert['sine']
        # dataframe['htleadsine'] = hilbert['leadsine']

        # # Pattern Recognition - Bullish candlestick patterns

        # dataframe['hikkake'] = ta.CDLHIKKAKE(dataframe)

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """


        # ---------------------------------------------------------
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        conditions = []
        conditions.append(qtpylib.crossed_above(
                dataframe[f'ema_{self.buy_ema.value}_slope'], self.buy_ema_slope.value
            ))
        
        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)
        conditions.append(dataframe['rsi'] > self.buy_rsi_low.value)
        conditions.append(dataframe['rsi'] < self.buy_rsi_high.value)
        # conditions.append(dataframe['aroonosc'] > self.buy_aroonosc.value)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        conditions = []
        conditions.append(qtpylib.crossed_below(
                dataframe['close'], dataframe[f'ema_{self.sell_ema.value}']
            ))

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'exit_long'] = 1
        
        return dataframe
