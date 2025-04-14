import numpy as np

"""
Given an array of prices, an initial cash, and a fee_ratio, there are three states BUY, SELL, and HOLD when you buy and sell.
There are three states: BUY, SELL, and HOLD, where BUY and SELL incur transaction fees, and HOLD means that no action is taken and no transaction fees are incurred.

Cash is the amount of cash you currently hold, position is the number of stocks you currently hold, price is the current stock price, and value is your current total value.

The BUY, HOLD and SELL, and VALUE functions are defined as follows:
def buy(cash, position, price, fee_ratio).
    # No cash or not enough to cover the fee to buy
    buy_position = int(np.floor(cash / (price * (1 + fee_ratio))))
    position += buy_position
    cash -= buy_position * price * (1 + fee_ratio)
    return cash, position

def sell(cash, position, price, fee_ratio).
    # No position, can't sell
    cash += position * price * (1 - fee_ratio)
    position = 0
    return cash, position

def hold(cash, position, price, fee_ratio).
    # do nothing
    return cash, position

def value(cash, position, price).
    # Return the current total value
    return cash + position * price
    
Requires an unlimited number of transactions, the maximum value, and the corresponding sequence of transactions [BUY, SELL, HOLD, ...].
The complete code for implementing python using dynamic programming using the function implementation defined above

Here are a few sample inputs and outputs to ensure that your inputs and outputs are correct and meet the requirements.
1.
input：
prices = [1, 2, 3, 4, 5, 6]
cash = 1000
fee_ratio = 0.001
output：
final_value = 5994.001, final_actions = ['BUY', 'HOLD', 'HOLD', 'HOLD', 'HOLD', 'HOLD']
action: BUY price: 1 cash: 0.00100000000009004 position: 999 value: 999.0010000000001
action: HOLD price: 2 cash: 0.00100000000009004 position: 999 value: 1998.0010000000002
action: HOLD price: 3 cash: 0.00100000000009004 position: 999 value: 2997.001
action: HOLD price: 4 cash: 0.00100000000009004 position: 999 value: 3996.001
action: HOLD price: 5 cash: 0.00100000000009004 position: 999 value: 4995.001
action: HOLD price: 6 cash: 0.00100000000009004 position: 999 value: 5994.001

2.
input：
prices = [6, 5, 4, 3, 2, 1]
cash = 1000
fee_ratio = 0.001
output：
final_value = 1000, final_actions = ['HOLD', 'HOLD', 'HOLD', 'HOLD', 'HOLD', 'HOLD']
action: HOLD price: 6 cash: 1000 position: 0 value: 1000
action: HOLD price: 5 cash: 1000 position: 0 value: 1000
action: HOLD price: 4 cash: 1000 position: 0 value: 1000
action: HOLD price: 3 cash: 1000 position: 0 value: 1000
action: HOLD price: 2 cash: 1000 position: 0 value: 1000
action: HOLD price: 1 cash: 1000 position: 0 value: 1000

3.
input:
prices = [1, 3, 6, 3, 5, 1]
cash = 1000
fee_ratio = 0.001
output:
final_value = 9960.055000000002, final_actions = ['BUY', 'HOLD', 'SELL', 'BUY', 'SELL', 'HOLD']
action: BUY price: 1 cash: 0.00100000000009004 position: 999 value: 999.0010000000001
action: HOLD price: 3 cash: 0.00100000000009004 position: 999 value: 2997.001
action: SELL price: 6 cash: 5988.0070000000005 position: 0 value: 5988.0070000000005
action: BUY price: 3 cash: 0.02500000000145519 position: 1994 value: 5982.0250000000015
action: SELL price: 5 cash: 9960.055000000002 position: 0 value: 9960.055000000002
action: HOLD price: 1 cash: 9960.055000000002 position: 0 value: 9960.055000000002
"""

def buy(cash, position, price, fee_ratio):
    buy_position = int(np.floor(cash / (price * (1 + fee_ratio))))
    position += buy_position
    cash -= buy_position * price * (1 + fee_ratio)
    return cash, position

def sell(cash, position, price, fee_ratio):
    cash += position * price * (1 - fee_ratio)
    position = 0
    return cash, position

def noop(cash, position, price, fee_ratio):
    return cash, position

def value(cash, position, price):
    return cash + position * price


def max_profit_with_actions(prices, cash, fee_ratio):
    """
    This function computes the maximum profit that can be made by buying and selling a stock with a fee.
    The function also returns the sequence of actions that leads to the maximum profit.
    Args:
        prices (list): A list of prices of the stock.
        cash (int): The initial amount of cash.
        fee_ratio (float): The fee ratio when buying or selling a stock.
    """

    n = len(prices)
    if n == 0:
        return 0, []

    # Initialize the arrays for DP, actions, cash, and position
    dp = [[0 for _ in range(2)] for _ in range(n)]  # 0: Not holding, 1: Holding
    actions = [['HOLD' for _ in range(2)] for _ in range(n)]
    cashs = [[cash for _ in range(2)] for _ in range(n)]
    positions = [[0 for _ in range(2)] for _ in range(n)]

    # Initialize for the first day
    cashs[0][1], positions[0][1] = buy(cash, 0, prices[0], fee_ratio)
    actions[0][1] = 'BUY'
    dp[0][1] = value(cashs[0][1], positions[0][1], prices[0])

    for i in range(1, n):
        # If not holding stock
        # Option 1: Continue not holding (noop from not holding)
        cash_noop, position_noop = noop(cashs[i-1][0], positions[i-1][0], prices[i], fee_ratio)
        value_noop = value(cash_noop, position_noop, prices[i])

        # Option 2: Sell (from holding)
        cash_sell, position_sell = sell(cashs[i-1][1], positions[i-1][1], prices[i], fee_ratio)
        value_sell = value(cash_sell, position_sell, prices[i])

        # Choose the better option
        if value_noop > value_sell:
            dp[i][0] = value_noop
            cashs[i][0], positions[i][0] = cash_noop, position_noop
            actions[i][0] = 'HOLD'
        else:
            dp[i][0] = value_sell
            cashs[i][0], positions[i][0] = cash_sell, position_sell
            actions[i][0] = 'SELL'

        # If holding stock
        # Option 1: Continue holding (noop from holding)
        cash_noop_holding, position_noop_holding = noop(cashs[i-1][1], positions[i-1][1], prices[i], fee_ratio)
        value_noop_holding = value(cash_noop_holding, position_noop_holding, prices[i])

        # Option 2: Buy (from not holding)
        cash_buy, position_buy = buy(cashs[i-1][0], positions[i-1][0], prices[i], fee_ratio)
        value_buy = value(cash_buy, position_buy, prices[i])

        # Choose the better option
        if value_noop_holding > value_buy:
            dp[i][1] = value_noop_holding
            cashs[i][1], positions[i][1] = cash_noop_holding, position_noop_holding
            actions[i][1] = 'HOLD'
        else:
            dp[i][1] = value_buy
            cashs[i][1], positions[i][1] = cash_buy, position_buy
            actions[i][1] = 'BUY'

    # Backtrack to find the action sequence
    final_value = max(dp[n-1])
    is_holding = dp[n-1].index(final_value)
    final_actions = []

    for i in range(n-1, -1, -1):
        final_actions.append(actions[i][is_holding])
        is_holding = 1 - is_holding if actions[i][is_holding] in ['BUY', 'SELL'] else is_holding

    return final_value, list(reversed(final_actions))


def max_profit_with_actions_threshold(prices, cash, fee_ratio, max_count_sell):
    n = len(prices)
    if n == 0:
        return 0, []

    # Initialize the arrays for DP, actions, cash, and position
    # dp[i][j][0/1] defines the max_profit from 1 to i, under hold or not hold, and using j times SELL operation.

    dp = [[[0 for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]  # 0: Not holding, 1: Holding
    actions = [[['HOLD' for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]
    cashs = [[[cash for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]
    positions = [[[0 for _ in range(2)] for _ in range(max_count_sell + 1)] for _ in range(n)]

    # Initialize for the first day
    cashs[0][0][1], positions[0][0][1] = buy(cash, 0, prices[0], fee_ratio)
    actions[0][0][1] = 'BUY'
    dp[0][0][1] = value(cashs[0][0][1], positions[0][0][1], prices[0])

    for i in range(1, n):
        for j in range(0, max_count_sell + 1):
            # If not holding stock
            # Option 1: Continue not holding (noop from not holding)
            cash_noop, position_noop = noop(cashs[i-1][j][0], positions[i-1][j][0], prices[i], fee_ratio)
            value_noop = value(cash_noop, position_noop, prices[i])

            if j == 0:
                dp[i][j][0] = value_noop
                cashs[i][j][0], positions[i][j][0] = cash_noop, position_noop
                actions[i][j][0] = 'HOLD'
            else:
                # Option 2: Sell (from holding)
                cash_sell, position_sell = sell(cashs[i-1][j-1][1], positions[i-1][j-1][1], prices[i], fee_ratio)
                value_sell = value(cash_sell, position_sell, prices[i])

                # Choose the better option
                if value_noop > value_sell:
                    dp[i][j][0] = value_noop
                    cashs[i][j][0], positions[i][j][0] = cash_noop, position_noop
                    actions[i][j][0] = 'HOLD'
                else:
                    dp[i][j][0] = value_sell
                    cashs[i][j][0], positions[i][j][0] = cash_sell, position_sell
                    actions[i][j][0] = 'SELL'

            # If holding stock
            # Option 1: Continue holding (noop from holding)
            cash_noop_holding, position_noop_holding = noop(cashs[i-1][j][1], positions[i-1][j][1], prices[i], fee_ratio)
            value_noop_holding = value(cash_noop_holding, position_noop_holding, prices[i])

            # Option 2: Buy (from not holding)
            cash_buy, position_buy = buy(cashs[i-1][j][0], positions[i-1][j][0], prices[i], fee_ratio)
            value_buy = value(cash_buy, position_buy, prices[i])

            # Choose the better option
            if value_noop_holding > value_buy:
                dp[i][j][1] = value_noop_holding
                cashs[i][j][1], positions[i][j][1] = cash_noop_holding, position_noop_holding
                actions[i][j][1] = 'HOLD'
            else:
                dp[i][j][1] = value_buy
                cashs[i][j][1], positions[i][j][1] = cash_buy, position_buy
                actions[i][j][1] = 'BUY'

    # Backtrack to find the action sequence
    final_value = max(dp[n-1][max_count_sell])
    is_holding = dp[n-1][max_count_sell].index(final_value)
    final_actions = []
    count_sell = max_count_sell

    for i in range(n-1, -1, -1):
        now_action = actions[i][count_sell][is_holding]
        final_actions.append(now_action)
        is_holding = 1 - is_holding if now_action in ['BUY', 'SELL'] else is_holding
        count_sell = count_sell - 1 if now_action in ['SELL'] else count_sell

    return final_value, list(reversed(final_actions))

if __name__ == '__main__':

    examples = [
        [1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1],
        [1, 3, 6, 3, 5, 4],
        [1, 3, 6, 3, 5, 1]
    ]
    threshold = 0.5

    for prices in examples:
        cash = 1000
        fee_ratio = 0.001
        position = 0
        max_profit, actions = max_profit_with_actions(prices, cash, fee_ratio)
        print("max_profit:", max_profit, "actions:", actions)
        for action, price in zip(actions, prices):
            if action == 'BUY':
                cash, position = buy(cash, position, price, fee_ratio)
            elif action == 'SELL':
                cash, position = sell(cash, position, price, fee_ratio)
            else:
                cash, position = noop(cash, position, price, fee_ratio)
            value_ = cash + position * price
            print("action:", action, "price:", price, "cash:", cash, "position:", position, "value:", value_)
        print()

        #========================================= Threshold ===========================================#

        count_sell = 0                                      # number of 'SELL' operations
        for action, price in zip(actions, prices):
            if action == 'SELL':
                count_sell += 1
        max_count_sell = int(count_sell * threshold)        # limited_number of 'SELL' operations
        
        cash = 1000
        fee_ratio = 0.001
        position = 0
        max_profit, actions = max_profit_with_actions_threshold(prices, cash, fee_ratio, max_count_sell)
        print(f"================== max_count_sell : {max_count_sell} ==================")
        print("max_profit:", max_profit, "actions:", actions)
        for action, price in zip(actions, prices):
            if action == 'BUY':
                cash, position = buy(cash, position, price, fee_ratio)
            elif action == 'SELL':
                cash, position = sell(cash, position, price, fee_ratio)
            else:
                cash, position = noop(cash, position, price, fee_ratio)
            value_ = cash + position * price
            print("action:", action, "price:", price, "cash:", cash, "position:", position, "value:", value_)
        print("=========================================================")
        print()

