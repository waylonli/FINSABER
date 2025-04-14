import pandas as pd
import numpy as np
import ta

class StrategyAgents:
    """
    Our trading strategies expertly leverage technical indicators across four key areas: (1) Trend, for following market direction; (2) Momentum, to identify strong market moves; (3) Volatility, for detecting price fluctuations; and (4) Volume, to gauge trading activity.
    These strategies encompass Trend Following, Overbought & Oversold conditions, Support & Resistance, and Pattern Finding techniques, providing traders with precise tools for a range of market conditions.
    """

    def data_process(self, data):
        """
        Method to process the input dataframe.
        Renames columns to standard names.
        """
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'adj_close'])
        else:
            df = data
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        return df

    def wrapper(self,strategy, data,params=None):
        """
        Wrapper method to run the strategy and return the results.
        """
        if strategy == 0:
            return self.strategy0_Buy_and_Hold(data,params)
        if strategy == 1:
            if params:
                return self.strategy1_MACD(data,params)
            else:
                return self.strategy1_MACD(data)
        elif strategy == 2:
            if params:
                return self.strategy2_KDJ_RSI(data,params)
            else:
                return self.strategy2_KDJ_RSI(data)
        elif strategy == 3:
            if params:
                return self.strategy3_Stochastic_Bollinger(data,params)
            else:
                return self.strategy3_Stochastic_Bollinger(data)
        elif strategy == 4:
            if params:
                return self.strategy4_Mean_Reversion(data,params)
            else:
                return self.strategy4_Mean_Reversion(data)
        elif strategy == 5:
            if params:
                return self.strategy5_Mean_Reversion_ATR(data,params)
            else:
                return self.strategy5_Mean_Reversion_ATR(data)
        elif strategy == 6:
            if params:
                return self.strategy6_Pattern_Zigzag(data,params)
            else:
                return self.strategy6_Pattern_Zigzag(data)
        else:
            return None, None

    def strategy0_Buy_and_Hold(self, data, params=None):
        """
        Method to implement the Buy and Hold strategy.
        Returns a list of signals and explanations based on the Buy and Hold strategy.

        >> Summary:
        [Name] : Buy and Hold Strategy
        [Type] : None
        [Description] : This strategy is the simplest of all, and is often used as a benchmark to compare other strategies against. It simply buys the asset and holds it for the entire duration of the backtest. This strategy is useful for comparing the performance of other strategies, and is often used as a baseline to determine if a strategy is profitable.
        """
        signals = ['HOLD'] * len(data)
        if params["FirstDay"] == True:
            signals[-1] = 'BUY'
        explanations = ['Buy and hold strategy, buy at the start and hold until the end.'] * len(data)
        return signals, explanations

    def strategy1_MACD(self, data, params={'short_window': 7}):
        """
        Method to implement the MACD strategy.
        Returns a list of signals and explanations based on the MACD strategy.

        >> Summary:
        [Name] : MACD Crossover Strategy
        [Type] : Trend Following
        [Description] : This strategy uses the Moving Average Convergence Divergence (MACD) indicator to generate trading signals. A BUY signal is generated when the MACD line crosses above the signal line, indicating a bullish momentum. A SELL signal is generated when the MACD line crosses below the signal line, indicating a bearish momentum. In real life, this means buying when the momentum is expected to go up and selling when it's expected to go down. HOLD signal is given when there's no clear trend.
        """
        short_window = params['short_window']
        long_window = 14
        data = self.data_process(data)
        macd_line, signal_line = self.calculate_macd(data, short_window, long_window)
        signals = []
        explanations = []
        for i in range(len(data)):
            if i == 0:
                signals.append('HOLD')
                explanations.append('Initial state, no data to compare.')
            elif macd_line[i] > signal_line[i] and macd_line[i-1] < signal_line[i-1]:
                signals.append('BUY')
                explanations.append('MACD line crossed above the signal line, indicating bullish momentum. This is a BUY signal.')
            elif macd_line[i] < signal_line[i] and macd_line[i-1] > signal_line[i-1]:
                signals.append('SELL')
                explanations.append('MACD line crossed below the signal line, indicating bearish momentum. This is a SELL signal.')
            else:
                signals.append('HOLD')
                explanations.append('No clear trend, HOLD position.')
        
        return signals, explanations

    def strategy2_KDJ_RSI(self, data, params={"ilong" : 9, "isig" : 3,  "rsiOverbought" : 60, "rsiOversold" : 40}):
        """
        Method to implement the KDJ strategy with an RSI filter.
        Returns a list of signals and explanations based on the KDJ and RSI values.

        >> Summary:
        [Name] : KDJ with RSI Filter Strategy
        [Type] : Overbought & Oversold
        [Short Description] : This strategy uses the KDJ (Stochastic Oscillator) and RSI (Relative Strength Index) indicators to generate trading signals. A BUY signal is generated when the J line of the KDJ crosses above the D line, indicating bullish momentum, and the RSI is below a specified oversold threshold. A SELL signal is generated when the J line of the KDJ crosses below the D line, indicating bearish momentum, and the RSI is above a specified overbought threshold. In real life, this means buying when the price is considered too low and likely to increase, and selling when the price is considered too high and likely to decrease. A HOLD signal is given when there's no clear trend or the conditions for buy/sell signals are not met.
        """
        data = self.data_process(data)

        ilong = params["ilong"]
        isig = params["isig"]
        rsiPeriod = 14
        rsiOverbought = params["rsiOverbought"]
        rsiOversold = params["rsiOversold"]
        useRsiFilter = True

        k, d, j = self.calculate_kdj(data, ilong, isig)
        rsi = self.calculate_rsi(data, rsiPeriod)

        signals = []
        explanations = []
        for i in range(len(data)):
            if i < max(ilong, isig, rsiPeriod):
                signals.append('HOLD')
                explanations.append('Initial state, not enough data to generate signals.')
            elif (j[i] > d[i] and j[i-1] < d[i-1]) and (not useRsiFilter or rsi[i] < rsiOversold):
                signals.append('BUY')
                explanations.append('J line crossed above the D line indicating a bullish signal, and RSI is below the oversold threshold. This is a BUY signal.')
            elif (j[i] < d[i] and j[i-1] > d[i-1]) and (not useRsiFilter or rsi[i] > rsiOverbought):
                signals.append('SELL')
                explanations.append('J line crossed below the D line indicating a bearish signal, and RSI is above the overbought threshold. This is a SELL signal.')
            else:
                signals.append('HOLD')
                explanations.append('No clear trend, HOLD position.')

        return signals, explanations
    
    def strategy3_Stochastic_Bollinger(self, data, params={"std_dev":2, "overbought":80, "oversold":20}):
        """
        Stochastic Oscillator and Bollinger Bands strategy: Buy when the price is below the lower Bollinger Band and the Stochastic Oscillator is below 20. Sell when the price is above the upper Bollinger Band and the Stochastic Oscillator is above 80.
        Returns a list of signals and explanations based on the Stochastic Oscillator and Bollinger Bands values.
        
        >> Summary:
        [Name] : Stochastic Oscillator and Bollinger Bands Strategy
        [Type] : Overbought & Oversold, Support & Resistance
        [Description] : This strategy combines the StochasticVMA Oscillator and Bollinger Bands to generate trading signals. A BUY signal is generated when the price is below the lower Bollinger Band and the Stochastic Oscillator is below a specified oversold threshold. This indicates that the price is considered too low and likely to increase. A SELL signal is generated when the price is above the upper Bollinger Band and the Stochastic Oscillator is above a specified overbought threshold. This indicates that the price is considered too high and likely to decrease. In real life, this means buying when the asset is undervalued and selling when it's overvalued. A HOLD signal is given when there's no clear trend or the conditions for buy/sell signals are not met.
        """
        data = self.data_process(data)

        lookback = 14
        sma_period = 14
        std_dev = params["std_dev"]
        overbought = params["overbought"]
        oversold = params["oversold"]


        # Calculate Bollinger Bands
        sma = data['Close'].rolling(window=sma_period).mean()
        std_deviation = data['Close'].rolling(window=sma_period).std()
        lower_bb = sma - std_dev * std_deviation
        upper_bb = sma + std_dev * std_deviation

        # Calculate Stochastic Oscillator
        lowest = data['Low'].rolling(window=lookback).min()
        highest = data['High'].rolling(window=lookback).max()
        stochastic_oscillator = 100 * ((data['Close'] - lowest) / (highest - lowest))

        signals = []
        explanations = []
        for i in range(len(data)):
            if i < max(lookback, sma_period):
                signals.append('HOLD')
                explanations.append('Initial state, not enough data to generate signals.')
            elif stochastic_oscillator[i] < oversold and data['Close'].iloc[i] < lower_bb.iloc[i]:
                signals.append('BUY')
                explanations.append('The price is below the lower Bollinger Band and the Stochastic Oscillator is below the oversold threshold. This is a BUY signal.')
            elif stochastic_oscillator[i] > overbought and data['Close'].iloc[i] > upper_bb.iloc[i]:
                signals.append('SELL')
                explanations.append('The price is above the upper Bollinger Band and the Stochastic Oscillator is above the overbought threshold. This is a SELL signal.')
            else:
                signals.append('HOLD')
                explanations.append('No clear trend, HOLD position.')

        return signals, explanations

    def strategy4_Mean_Reversion(self, data, params={"z_score_threshold":1.0}):
        """
        Mean Reversion strategy: Buy when the z-score is below -1.0 (indicating the price is below the mean by a certain threshold). Sell when the z-score is above 1.0 (indicating the price is above the mean by a certain threshold).
        Returns a list of signals and explanations based on the z-score values.

        >> Summary:
        [Name] : Mean Reversion Strategy
        [Type] : Mean Reversion
        [Description] : This strategy assumes that the price will revert to its mean over time. A BUY signal is generated when the z-score is below -1.0, indicating that the price is significantly below the mean and likely to increase. A SELL signal is generated when the z-score is above 1.0, indicating that the price is significantly above the mean and likely to decrease. A HOLD signal is given when the z-score is between -1.0 and 1.0, indicating no significant deviation from the mean.
        """
        data = self.data_process(data)
        lookback = 14
        z_score_threshold = params["z_score_threshold"]
        # Calculate mean and standard deviation of closing prices over the lookback period
        mean = data['Close'].rolling(window=lookback).mean()
        std_dev = data['Close'].rolling(window=lookback).std()

        # Calculate z-score for each day
        z_score = (data['Close'] - mean) / std_dev

        signals = []
        explanations = []
        for i in range(len(data)):
            if i < lookback:
                signals.append('HOLD')
                explanations.append('Initial state, not enough data to generate signals.')
            elif z_score[i] < -z_score_threshold:
                signals.append('BUY')
                explanations.append(
                    'The z-score is below -1.0, indicating that the price is significantly below the mean. This is a BUY signal.')
            elif z_score[i] > z_score_threshold:
                signals.append('SELL')
                explanations.append(
                    'The z-score is above 1.0, indicating that the price is significantly above the mean. This is a SELL signal.')
            else:
                signals.append('HOLD')
                explanations.append(
                    'The z-score is between -1.0 and 1.0, indicating no significant deviation from the mean. HOLD position.')

        return signals, explanations

    def strategy5_Mean_Reversion_ATR(self,data, params:{"atr_length":7, "atr_multiplier":2, "len_volat":7, "len_drift":7,"multiple_std":1}):
        """
        Mean Reversion and ATR Strategy: Buy when ATR indicates high market volatility and the trend is upward, and sell when the price falls below the trailing stop loss calculated using the ATR.
        >> Summary:
        [Name] : Mean Reversion and ATR Strategy
        [Type] : Volatility & Trend Analysis, Mean Reversion, Risk Management
        [Description] : This strategy combines Average True Range (ATR) analysis with log-normal trend predictions to identify trading opportunities in volatile markets. A BUY signal is generated when the ATR indicates significant market volatility and trend analysis confirms an upward movement, suggesting a potential profitable entry. Conversely, a SELL signal is generated when the price drops below the trailing stop loss, set based on the ATR, to manage risks and prevent large losses. The strategy capitalizes on the market's mean-reverting nature during volatile conditions, aiming to enter trades when the price is expected to bounce back and exit to cut losses or lock in profits as needed.
        """
        # Calculate ATR
        data = self.data_process(data)
        atr_length = params["atr_length"]
        atr_multiplier = params["atr_multiplier"]
        len_volat = params["len_volat"]
        len_drift = params["len_drift"]
        multiple_std = params["multiple_std"]

        data['atr'] = self.calculate_ATR(data, atr_length)

        # Trailing Stop Loss
        data['trailing_stop'] = data['Low'] - (data['atr'] * atr_multiplier)
        data['trailing_stop_max'] = data['trailing_stop'].cummax()

        # ATR Divergence Signal
        data['avg_atr'] = data['atr'].rolling(window=len_volat).mean()
        data['std_atr'] = data['atr'].rolling(window=len_volat).std()
        # data['signal_diverted_atr'] = ((data['atr'] > (data['avg_atr'] + data['std_atr'])) |
        #                                (data['atr'] < (data['avg_atr'] - data['std_atr'])))
        data['avg_atr'] = data['atr'].rolling(window=len_volat).mean()
        data['std_atr'] = data['atr'].rolling(window=len_volat).std()

        # Define the signal: true if ATR deviates from the mean more than 'multiple_std' standard deviations
        data['signal_diverted_atr'] = ((data['atr'] > (data['avg_atr'] + multiple_std * data['std_atr'])) |
                                       (data['atr'] < (data['avg_atr'] - multiple_std * data['std_atr'])))
        # Lognorm Returns for Trend Prediction
        data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['drift'] = data['log_return'].rolling(window=len_drift).mean() - \
                        0.5 * data['log_return'].rolling(window=len_drift).std() ** 2
        data['signal_uptrend'] = (data['drift'] > data['drift'].shift(1)) & \
                                 (data['drift'] > data['drift'].shift(2)) | \
                                 (data['drift'] > 0)

        # Combined Signal for Entry
        data['entry_signal'] = data['signal_diverted_atr'] & data['signal_uptrend']

        # Determine Buy/Sell Signals
        data['position'] = 0  # No position initially
        data['position'] = np.where(data['entry_signal'], 1, 0)  # 1 for Buy, 0 for Hold
        # Implementing Trailing Stop Loss Logic
        data['trailing_stop_max'] = data['trailing_stop'].cummax()
        # for index where data['Low'] < data['trailing_stop_max'], set position to -1
        data.loc[data['Low'] < data['trailing_stop_max'], 'position'] = -1
        # Position Logic
        holding_explanation= "Maintaining current position as neither buy nor sell conditions have been met based on ATR divergence and trend prediction signals. Awaiting clearer market signals aligned with strategy criteria before making a trade decision."
        buying_explanation="Executing a buy order as current ATR indicates significant divergence from its rolling mean, suggesting increased volatility, and lognormal returns confirm an upward trend. This alignment with both ATR and trend conditions matches our strategy's criteria for a potential profitable entry."
        selling_explanation="Initiating a sell order as the current low price has fallen below the maximum trailing stop loss level. This indicates a potential reversal or significant decrease in the asset's price, aligning with our strategy's risk management rules to protect from larger losses."
        signals=['HOLD'] * len(data)
        explanations= [holding_explanation] * len(data)
        buy_signals = data[data['position']==1]
        sell_signals = data[data['position']==-1]
        buy_signal_indices = buy_signals.index.tolist()
        sell_signal_indices = sell_signals.index.tolist()
        for idx in buy_signal_indices:
            signals[idx] = 'BUY'
            explanations[idx] = buying_explanation
        for idx in sell_signal_indices:
            signals[idx] = 'SELL'
            explanations[idx] = selling_explanation
        return signals, explanations


    def strategy6_Pattern_Zigzag(self,data, percentage_change=5):
        """
            Zigzag Pattern Trading Strategy: Utilize the zigzag pattern to identify potential buy and sell signals based on trend reversals.

            >> Summary:
            [Name]: Zigzag Pattern Trading Strategy
            [Type]: Pattern Finding, Trend Reversal Identification
            [Description]: This strategy leverages the Zigzag pattern to decipher market trends and reversals. A BUY signal is generated when there's an upward movement in the zigzag pattern, indicating a potential bullish reversal or continuation of an upward trend. Conversely, a SELL signal is triggered when the zigzag pattern moves downward, suggesting a bearish reversal or the continuation of a downward trend. The strategy is particularly effective in filtering out market noise and focusing on significant movements. The percentage change parameter allows for customization of the strategy's sensitivity to market movements, enabling adaptability to different market conditions and volatility levels.
        """

        data = self.data_process(data)
        # Calculate Zigzag
        data = self.calculate_zigzag(data, percentage_change)
        # Position Logic
        holding_explanation = "We hold our position as the current market trend does not exhibit a significant zigzag pattern indicating a trend reversal or continuation. This suggests that the market is in a consolidation phase, and it is prudent to await clearer signals before making a trading decision."
        buying_explanation = "A Buy signal is generated as the latest data point in the zigzag pattern is higher than the previous one, indicating an upward trend. This suggests a potential bullish reversal or continuation of an existing upward trend, making it an opportune moment to enter or add to a long position."
        selling_explanation = "A Sell signal is issued when the latest zigzag point is lower than the preceding one, signaling a downward trend. This indicates a potential bearish reversal or the continuation of an existing downward trend, suggesting it may be an appropriate time to exit or shorten the position."
        signals = 'Hold' * len(data)
        explanation = [holding_explanation] * len(data)
        for i in range(1, len(data)):
            if pd.notna(data['Zigzag'].iloc[i]):
                if pd.isna(data['Zigzag'].iloc[i - 1]) or data['Zigzag'].iloc[i] > data['Zigzag'].iloc[i - 1]:
                    signals[i] = 'Buy'
                    explanation[i]=buying_explanation
                else:
                    signals[i] = 'Sell'
                    explanation[i]=selling_explanation
        return signals, explanations

    def calculate_zigzag(self,data, percentage_change=5):

        data['Zigzag'] = np.nan
        last_peak = last_trough = None
        last_price = data['Close'].iloc[0]

        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            change_percent = ((current_price - last_price) / last_price) * 100

            if change_percent >= percentage_change:  # Potential peak
                if last_trough is not None or last_peak is None:
                    last_peak = i
                    data['Zigzag'].iloc[last_peak]= current_price
                    last_trough = None
                elif change_percent <= -percentage_change: # Potential trough
                    if last_peak is not None or last_trough is None:
                        last_trough = i
                        data['Zigzag'].iloc[last_trough] = current_price
                        last_peak = None
                        last_price = current_price
        return data

    def calculate_ATR(self, data, atr_length=14):
        return ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=atr_length)

    def calculate_ema(self, data, window):
        """
        Helper method to calculate the Exponential Moving Average (EMA).
        """
        return data.ewm(span=window, adjust=False).mean()

    def calculate_macd(self, data, short_window, long_window):
        """
        Helper method to calculate the Moving Average Convergence Divergence (MACD).
        """
        short_ema = self.calculate_ema(data['Close'], short_window)
        long_ema = self.calculate_ema(data['Close'], long_window)
        macd_line = short_ema - long_ema
        signal_line = self.calculate_ema(macd_line, 9)
        return macd_line, signal_line
    
    def calculate_kdj(self, data, ilong, isig):
        """
        Helper method to calculate the KDJ (K Percent, D Percent, J Percent) lines.
        """
        # Calculate high and low for the given period
        high = data['High'].rolling(window=ilong).max()
        low = data['Low'].rolling(window=ilong).min()

        # Calculate RSV (Raw Stochastic Value)
        rsv = (data['Close'] - low) / (high - low) * 100

        # Calculate K, D, and J lines
        k = rsv.rolling(window=isig).mean()
        d = k.rolling(window=isig).mean()
        j = 3 * k - 2 * d

        return k, d, j

    def calculate_rsi(self, data, rsiPeriod):
        """
        Helper method to calculate the RSI (Relative Strength Index).
        """
        return ta.momentum.RSIIndicator(data['Close'], window=rsiPeriod).rsi()
    

if __name__ == "__main__":
    data = pd.read_csv("AAPL.csv")

    agent = StrategyAgents()
    print("running strategy 1")
    res1 = agent.strategy1_MACD(data)
    print("running strategy 2")
    res2 = agent.strategy2_KDJ_RSI(data)
    print("running strategy 3")
    res3 = agent.strategy3_Stochastic_Bollinger(data)
    print("running strategy 4")
    res4 = agent.strategy4_Mean_Reversion_ATR(data)
    print("running strategy 5")
    res5 = agent.strategy5_Pattern_Zigzag(data)