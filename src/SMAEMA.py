
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import tpqoa
pd.options.display.max_rows = 400
# from scipy.optimize import brute
# plt.style.use("seaborn")


class SMAEMA():
    ''' Class for the vectorized backtesting of SMA/EMA-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    SMA: int
        time window in days for SMA
    EMA: int
        time window in days for EMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade


    Methods
    =======
    get_data:
        retrieves and prepares the data

    set_parameters:
        sets one or two new SMA/EMA parameters

    test_strategy:
        runs the backtest for the SMA/EMA-based strategy

    plot_results:
        plots the performance of the strategy compared to buy and hold

    update_and_run:
        updates EMA parameters and returns the negative absolute performance (for minimization algorithm)

    optimize_parameters:
        implements a brute force optimization for the two SAM/EMA parameters
    '''

    def __init__(self, symbol, SMA, EMA, start, end, tc):
        self.symbol = symbol
        self.SMA = SMA
        self.EMA = EMA
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None

    def __repr__(self):
        return "SMAEMABacktester(symbol = {}, SMA = {}, EMA = {}, start = {}, end = {})".format(self.symbol, self.SMA, self.EMA, self.start, self.end)

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument=self.symbol, start=self.start,
                              end=self.end, granularity="H1", price="A")
        raw = raw[["c"]]
        raw.rename(columns={"c": "Price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        raw = raw.dropna()
        raw["SMA"] = raw["Price"].rolling(self.SMA).mean()
        raw["EMA"] = raw["Price"].ewm(
            span=self.EMA, min_periods=self.EMA).mean()
        pd.options.display.max_rows = 100
        print(raw)

    """def set_parameters(self, SMA = None, EMA = None):
        ''' Updates SMA/EMA parameters and resp. time series.
        '''
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["Price"].rolling(self.SMA).mean() 
        if EMA is not None:
            self.EMA = EMA
            self.data["EMA"] = self.data["Price"].ewm(span = self.EMA, min_periods = self.EMA).mean()
            
    """

    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        # do i need to copy the df?
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument=self.symbol, start=self.start,
                              end=self.end, granularity="H1", price="A")
        raw = raw[["c"]]
        raw.rename(columns={"c": "Price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        raw = raw.dropna()
        raw["SMA"] = raw["Price"].rolling(self.SMA).mean()
        raw["EMA"] = raw["Price"].ewm(
            span=self.EMA, min_periods=self.EMA).mean()
        raw["Position"] = np.where(raw["EMA"] > raw["SMA"], 1, -1)
        raw["Strategy"] = raw["Position"].shift(
            1) * raw["Returns"].round(decimals=6)
        raw.dropna(inplace=True)

        # determine when a trade takes place
        raw["Trades"] = raw.Position.diff().fillna(0).abs()

        # subtract transaction costs from return when trade takes place
        raw.Strategy = raw.Strategy - raw.Trades * self.tc
        raw["CReturns"] = raw["Returns"].cumsum().apply(np.exp)
        raw["CStrategy"] = raw["Strategy"].cumsum().apply(np.exp)
        # self.results = print(raw)

        # absolute performance of the strategy
        perf = raw["CStrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - raw["CReturns"].iloc[-1]
        pd.options.display.max_rows = 100
        print(raw.head(100))
        print([round(perf, 6), round(outperf, 6)])
        # do I have to print everything to the console?

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument=self.symbol, start=self.start,
                              end=self.end, granularity="H1", price="A")
        raw = raw[["c"]]
        raw.rename(columns={"c": "Price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        raw = raw.dropna()
        raw["SMA"] = raw["Price"].rolling(self.SMA).mean()
        raw["EMA"] = raw["Price"].ewm(
            span=self.EMA, min_periods=self.EMA).mean()
        raw["Position"] = np.where(raw["EMA"] > raw["SMA"], 1, -1)
        raw["Strategy"] = raw["Position"].shift(1) * raw["Returns"]
        raw.dropna(inplace=True)
        raw["Trades"] = raw.Position.diff().fillna(0).abs()
        raw.Strategy = raw.Strategy - raw.Trades * self.tc
        raw["CReturns"] = raw["Returns"].cumsum().apply(np.exp)
        raw["CStrategy"] = raw["Strategy"].cumsum().apply(np.exp)

        # Matplotlib chart
        plt.title("{} | SMA = {} | EMA = {} | TC = {}".format(
            self.symbol, self.SMA, self.EMA, self.tc))
        plt.plot((raw[["CReturns", "CStrategy"]].to_numpy()))
        plt.show()

    def set_parameters(self, SMA=None, EMA=None):
        ''' Updates SMA/EMA parameters and resp. time series.
        '''
        # will set_params change universally?
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument=self.symbol, start=self.start,
                              end=self.end, granularity="H1", price="A")
        raw = raw[["c"]]
        raw.rename(columns={"c": "Price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        raw = raw.dropna()
        raw["SMA"] = raw["Price"].rolling(self.SMA).mean()
        raw["EMA"] = raw["Price"].ewm(
            span=self.EMA, min_periods=self.EMA).mean()
        raw["Position"] = np.where(raw["EMA"] > raw["SMA"], 1, -1)
        raw["Strategy"] = raw["Position"].shift(1) * raw["Returns"].astype(int)
        raw.dropna(inplace=True)
        raw["Trades"] = raw.Position.diff().fillna(0).abs()
        raw.Strategy = raw.Strategy - raw.Trades * self.tc
        raw["CReturns"] = raw["Returns"].cumsum().apply(np.exp)
        raw["CStrategy"] = raw["Strategy"].cumsum().apply(np.exp)

        if SMA is not None and EMA is not None:
            self.SMA = SMA
            raw["SMA"] = raw["Price"].rolling(self.SMA).mean()
            self.EMA = EMA
            raw["EMA"] = raw["Price"].ewm(
                span=self.EMA, min_periods=self.EMA).mean()
            print([self.SMA, self.EMA])
        else:
            self.SMA = self.SMA
            self.EMA = self.EMA
            print([self.SMA, self.EMA])

    def update_and_run(self):
        ''' Updates SMA/EMA parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        SMAEMA: tuple
            SMA/EMA parameter tuple
        '''
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument=self.symbol, start=self.start,
                              end=self.end, granularity="H1", price="A")
        raw = raw[["c"]]
        raw.rename(columns={"c": "Price"}, inplace=True)
        raw["Returns"] = np.log(raw / raw.shift(1))
        raw = raw.dropna()
        raw["SMA"] = raw["Price"].rolling(self.SMA).mean()
        raw["EMA"] = raw["Price"].ewm(
            span=self.EMA, min_periods=self.EMA).mean()
        raw["Position"] = np.where(raw["EMA"] > raw["SMA"], 1, -1)
        raw["Strategy"] = raw["Position"].shift(1) * raw["Returns"]
        raw.dropna(inplace=True)
        raw["Trades"] = raw.Position.diff().fillna(0).abs()
        raw.Strategy = raw.Strategy - raw.Trades * self.tc
        raw["CReturns"] = raw["Returns"].cumsum().apply(np.exp)
        raw["CStrategy"] = raw["Strategy"].cumsum().apply(np.exp)

        # unable to get this to work
        self.set_parameters(self)
        ts = self.test_strategy()
        print(ts[0][0])
    """ def optimize_parameters(self, SMA_range, EMA_range):
        ''' Finds global maximum given the SMA/EMA parameter ranges.

        Parameters
        ==========
        SMA_range, EMA_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (SMA_range, EMA_range), finish=None)
        return opt, -self.update_and_run(opt) """


# df = SMAEMA("USD_CAD", 50, 50, "2023-01-01", "2023-09-01", .00007)
# df.data
df = SMAEMA("EUR_USD", 43, 36, "2020-01-01", "2023-09-01", .00007)
# df.test_strategy()


# print(pd.set_option("display.max_rows", 1200))

df.plot_results()
# df.update_and_run(SMAEMA)

# df.set_parameters(75, 75)


# df.update_and_run()
