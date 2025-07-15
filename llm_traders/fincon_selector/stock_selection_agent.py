"""
Stock selection agent for FINCON.
Constructs portfolios using diversification methods.
"""

import logging
import pickle
from datetime import datetime
import json
import numpy as np
import pandas as pd
from scipy import stats

from llm_traders.fincon_selector.agents.base_agent import BaseAgent

class StockSelectionAgent(BaseAgent):
    """
    Stock selection agent for FINCON.
    
    Specializes in stock selection and portfolio construction, implementing
    risk diversification methods in quantitative finance.
    """
    
    def __init__(self, agent_id, target_symbols):
        """
        Initialize stock selection agent.
        
        Args:
            agent_id (str): Unique identifier for the agent
            target_symbols (list): List of stock symbols to analyze
        """
        super().__init__(agent_id, "stock_selection_agent", target_symbols)
        
        # Track selected portfolios
        self.portfolio_history = []
        self.correlation_matrices = {}
        
    def process(self, market_data, num_stocks=3, additional_data=None):
        """
        Process market data and select portfolio stocks.
        
        Args:
            market_data (dict): Dictionary of market data for each symbol
            num_stocks (int): Number of stocks to select for portfolio
            additional_data (dict, optional): Additional data for selection (news sentiment, etc.)
            
        Returns:
            dict: Portfolio selection and analysis
        """
        if not market_data or len(market_data) < 2:
            return {
                "timestamp": datetime.now().isoformat(),
                "message": "Insufficient market data for portfolio selection.",
                "selected_stocks": [],
                "weights": {},
                "rationale": "Need at least 2 stocks for diversification."
            }
        
        # Convert market data to returns
        returns_data = {}
        for symbol, data in market_data.items():
            if "adjusted_close" in data.columns:
                # Make sure the data is sorted by date and indexed by date
                returns = data.sort_values("date").set_index("date")["adjusted_close"].pct_change().dropna()
                returns_data[symbol] = returns
                
        # Check if we have sufficient data for correlation analysis
        if len(returns_data) < 2:
            return {
                "timestamp": datetime.now().isoformat(),
                "message": "Insufficient returns data for portfolio selection.",
                "selected_stocks": [],
                "weights": {},
                "rationale": "Need returns data for at least 2 stocks."
            }
            
        # Calculate correlation matrix
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        
        # Store correlation matrix
        timestamp = datetime.now().isoformat()
        self.correlation_matrices[timestamp] = correlation_matrix
        
        # Calculate performance metrics for each stock
        performance_metrics = self._calculate_performance_metrics(returns_data)
        
        # Select stocks based on correlation and performance
        selected_stocks = self._select_portfolio_stocks(
            correlation_matrix, 
            performance_metrics, 
            num_stocks, 
            additional_data
        )
        
        # Calculate suggested weights
        weights = self._calculate_portfolio_weights(
            selected_stocks, 
            correlation_matrix, 
            performance_metrics
        )
        
        # Generate portfolio analysis
        analysis = self._generate_portfolio_analysis(
            selected_stocks,
            weights,
            correlation_matrix,
            performance_metrics,
            additional_data
        )
        
        # Create portfolio selection result
        result = {
            "timestamp": timestamp,
            "selected_stocks": selected_stocks,
            "weights": weights,
            "correlation_matrix": correlation_matrix.to_dict(),
            "performance_metrics": performance_metrics,
            "rationale": analysis["rationale"],
            "sector_allocation": analysis["sector_allocation"],
            "risk_assessment": analysis["risk_assessment"],
            "expected_performance": analysis["expected_performance"]
        }
        
        # Store portfolio selection in working memory
        self.working_memory.add_item(result, "portfolio_selection")
        
        # Add to portfolio history
        self.portfolio_history.append(result)
        
        # Store in procedural memory
        analysis_text = json.dumps({
            "timestamp": timestamp,
            "selected_stocks": selected_stocks,
            "rationale": analysis["rationale"]
        })
        analysis_embedding = self._generate_embedding(analysis_text)
        memory_id = self.add_to_procedural_memory(
            analysis_text,
            "portfolio_selection",
            analysis_embedding
        )
        
        self.logger.info(f"Generated portfolio selection with ID {memory_id}")
        return result
        
    def _calculate_performance_metrics(self, returns_data):
        """
        Calculate performance metrics for each stock.
        
        Args:
            returns_data (dict): Dictionary of returns series for each symbol
            
        Returns:
            dict: Performance metrics
        """
        metrics = {}
        
        for symbol, returns in returns_data.items():
            # Skip if insufficient data
            if len(returns) < 30:
                continue
                
            # Calculate metrics
            metrics[symbol] = {
                "mean_return": float(returns.mean()),
                "annualized_return": float(returns.mean() * 252),
                "volatility": float(returns.std()),
                "annualized_volatility": float(returns.std() * np.sqrt(252)),
                "sharpe_ratio": float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0,
                "skewness": float(stats.skew(returns)),
                "kurtosis": float(stats.kurtosis(returns)),
                "var_95": float(np.percentile(returns, 5)),
                "cvar_95": float(returns[returns <= np.percentile(returns, 5)].mean())
            }
            
            # Add recent performance
            if len(returns) >= 30:
                metrics[symbol]["recent_return_1m"] = float(returns.iloc[-30:].mean() * 30)
            if len(returns) >= 90:
                metrics[symbol]["recent_return_3m"] = float(returns.iloc[-90:].mean() * 90)
                
        return metrics
        
    def _select_portfolio_stocks(self, correlation_matrix, performance_metrics, num_stocks, additional_data=None):
        """
        Select stocks for portfolio based on correlation and performance.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            performance_metrics (dict): Performance metrics for each stock
            num_stocks (int): Number of stocks to select
            additional_data (dict, optional): Additional data for selection
            
        Returns:
            list: Selected stock symbols
        """
        if len(correlation_matrix) <= num_stocks:
            # If we have fewer stocks than requested, return all
            return list(correlation_matrix.columns)
            
        # Calculate average correlation for each stock
        avg_correlation = {}
        for symbol in correlation_matrix.columns:
            avg_correlation[symbol] = correlation_matrix[symbol].mean()
            
        # Calculate combined score for each stock
        scores = {}
        for symbol, metrics in performance_metrics.items():
            if symbol not in avg_correlation:
                continue
                
            # Calculate score based on Sharpe ratio and low correlation
            score = metrics.get("sharpe_ratio", 0) * (1 - avg_correlation[symbol])
            scores[symbol] = score
            
        # Incorporate additional data if provided
        if additional_data:
            # Adjust scores based on sentiment
            if "sentiment" in additional_data:
                for symbol, sentiment in additional_data["sentiment"].items():
                    if symbol in scores:
                        sentiment_score = 0.0
                        if sentiment == "POSITIVE":
                            sentiment_score = 0.2
                        elif sentiment == "NEGATIVE":
                            sentiment_score = -0.2
                            
                        scores[symbol] += sentiment_score
                        
        # Select top stocks based on score
        selected_stocks = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)[:num_stocks]
        
        # Check if the selected portfolio has acceptable diversification
        portfolio_correlation = correlation_matrix.loc[selected_stocks, selected_stocks]
        avg_portfolio_correlation = portfolio_correlation.values.mean()
        
        # If average correlation is too high, try to improve diversification
        if avg_portfolio_correlation > 0.7 and len(correlation_matrix) > num_stocks:
            self.logger.info(f"High portfolio correlation ({avg_portfolio_correlation:.2f}), attempting to improve diversification")
            
            # Try a different selection approach based on minimizing total correlation
            selected_stocks = self._select_diverse_portfolio(correlation_matrix, performance_metrics, num_stocks)
            
        return selected_stocks
        
    def _select_diverse_portfolio(self, correlation_matrix, performance_metrics, num_stocks):
        """
        Select a diverse portfolio by minimizing total correlation.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            performance_metrics (dict): Performance metrics for each stock
            num_stocks (int): Number of stocks to select
            
        Returns:
            list: Selected stock symbols
        """
        # Start with the stock with highest Sharpe ratio
        sharpe_ratios = {symbol: metrics.get("sharpe_ratio", 0) for symbol, metrics in performance_metrics.items()}
        best_stock = max(sharpe_ratios.keys(), key=lambda s: sharpe_ratios[s])
        selected_stocks = [best_stock]
        
        # Iteratively add stocks with lowest correlation to already selected stocks
        available_stocks = set(correlation_matrix.columns) - set(selected_stocks)
        
        while len(selected_stocks) < num_stocks and available_stocks:
            # Calculate average correlation with already selected stocks for each available stock
            avg_correlations = {}
            for stock in available_stocks:
                corr_values = [correlation_matrix.loc[stock, selected] for selected in selected_stocks]
                avg_correlations[stock] = sum(corr_values) / len(corr_values)
                
            # Find stock with lowest average correlation
            min_corr_stock = min(avg_correlations.keys(), key=lambda s: avg_correlations[s])
            
            # Add to selected stocks
            selected_stocks.append(min_corr_stock)
            available_stocks.remove(min_corr_stock)
            
        return selected_stocks
        
    def _calculate_portfolio_weights(self, selected_stocks, correlation_matrix, performance_metrics):
        """
        Calculate portfolio weights using mean-variance optimization.
        
        Args:
            selected_stocks (list): Selected stock symbols
            correlation_matrix (pd.DataFrame): Correlation matrix
            performance_metrics (dict): Performance metrics for each stock
            
        Returns:
            dict: Portfolio weights
        """
        # If no stocks selected, return empty weights
        if not selected_stocks:
            return {}
            
        # For single stock, return 100% weight
        if len(selected_stocks) == 1:
            return {selected_stocks[0]: 1.0}
            
        # Extract expected returns and covariance matrix for selected stocks
        expected_returns = np.array([performance_metrics[s]["mean_return"] for s in selected_stocks])
        sub_correlation = correlation_matrix.loc[selected_stocks, selected_stocks]
        volatilities = np.array([performance_metrics[s]["volatility"] for s in selected_stocks])
        covariance = np.outer(volatilities, volatilities) * sub_correlation.values
        
        # Simple risk parity weights (inverse volatility weighting)
        # This is a simplified approach compared to actual mean-variance optimization
        inv_vol = 1.0 / np.array([performance_metrics[s]["volatility"] for s in selected_stocks])
        weights = inv_vol / inv_vol.sum()
        
        # Create weights dictionary
        weights_dict = {stock: float(weight) for stock, weight in zip(selected_stocks, weights)}
        
        return weights_dict
        
    def _generate_portfolio_analysis(self, selected_stocks, weights, correlation_matrix, performance_metrics, additional_data=None):
        """
        Generate analysis for selected portfolio.
        
        Args:
            selected_stocks (list): Selected stock symbols
            weights (dict): Portfolio weights
            correlation_matrix (pd.DataFrame): Correlation matrix
            performance_metrics (dict): Performance metrics for each stock
            additional_data (dict, optional): Additional data for analysis
            
        Returns:
            dict: Portfolio analysis
        """
        if not selected_stocks:
            return {
                "rationale": "No stocks selected.",
                "sector_allocation": {},
                "risk_assessment": "N/A",
                "expected_performance": {
                    "expected_return": 0.0,
                    "expected_volatility": 0.0,
                    "expected_sharpe": 0.0
                }
            }
            
        # Get sector information if available
        sector_allocation = {}
        if additional_data and "sectors" in additional_data:
            sectors = additional_data["sectors"]
            for stock in selected_stocks:
                sector = sectors.get(stock, "Unknown")
                if sector not in sector_allocation:
                    sector_allocation[sector] = 0.0
                sector_allocation[sector] += weights.get(stock, 1.0 / len(selected_stocks))
                
        # Calculate portfolio expected return and volatility
        expected_return = sum(weights.get(s, 0) * performance_metrics[s]["mean_return"] for s in selected_stocks)
        
        # Calculate portfolio volatility using correlation matrix
        if len(selected_stocks) > 1:
            # Extract weights vector
            weight_vector = np.array([weights.get(s, 0) for s in selected_stocks])
            
            # Extract submatrix of correlation matrix for selected stocks
            sub_correlation = correlation_matrix.loc[selected_stocks, selected_stocks]
            
            # Extract volatilities
            volatilities = np.array([performance_metrics[s]["volatility"] for s in selected_stocks])
            
            # Calculate covariance matrix
            covariance = np.outer(volatilities, volatilities) * sub_correlation.values
            
            # Calculate portfolio volatility
            portfolio_volatility = np.sqrt(weight_vector.T @ covariance @ weight_vector)
        else:
            # For single stock, volatility is simply the stock's volatility times weight
            portfolio_volatility = performance_metrics[selected_stocks[0]]["volatility"] * weights.get(selected_stocks[0], 1.0)
            
        # Calculate portfolio Sharpe ratio
        portfolio_sharpe = expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Calculate average correlation within portfolio
        if len(selected_stocks) > 1:
            portfolio_correlations = correlation_matrix.loc[selected_stocks, selected_stocks].values
            # Exclude diagonal elements (self-correlations)
            non_diag_indices = ~np.eye(len(selected_stocks), dtype=bool)
            avg_correlation = portfolio_correlations[non_diag_indices].mean()
        else:
            avg_correlation = 0.0
            
        # Determine risk assessment
        if portfolio_volatility < 0.01:
            risk_assessment = "LOW"
        elif portfolio_volatility < 0.02:
            risk_assessment = "MEDIUM"
        else:
            risk_assessment = "HIGH"
            
        # Generate rationale using LLM
        rationale = self._generate_rationale_text(
            selected_stocks, 
            weights, 
            performance_metrics, 
            portfolio_volatility, 
            expected_return, 
            portfolio_sharpe,
            avg_correlation
        )
        
        return {
            "rationale": rationale,
            "sector_allocation": sector_allocation,
            "risk_assessment": risk_assessment,
            "expected_performance": {
                "expected_return": float(expected_return * 252),  # Annualized
                "expected_volatility": float(portfolio_volatility * np.sqrt(252)),  # Annualized
                "expected_sharpe": float(portfolio_sharpe * np.sqrt(252))  # Annualized
            }
        }
        
    def _generate_rationale_text(self, selected_stocks, weights, performance_metrics, portfolio_volatility, expected_return, portfolio_sharpe, avg_correlation):
        """
        Generate rationale text for portfolio selection using LLM.
        
        Args:
            selected_stocks (list): Selected stock symbols
            weights (dict): Portfolio weights
            performance_metrics (dict): Performance metrics for each stock
            portfolio_volatility (float): Portfolio volatility
            expected_return (float): Expected portfolio return
            portfolio_sharpe (float): Portfolio Sharpe ratio
            avg_correlation (float): Average correlation between stocks
            
        Returns:
            str: Rationale text
        """
        # Build prompt
        prompt = f"""You are a portfolio analyst specializing in stock selection and portfolio construction.

Your task is to explain the rationale behind the following portfolio selection:

Selected Stocks: {', '.join(selected_stocks)}

Portfolio Weights:
"""

        # Add weights
        for stock, weight in weights.items():
            prompt += f"- {stock}: {weight:.2f}\n"
            
        # Add performance metrics
        prompt += "\nPerformance Metrics:\n"
        for stock in selected_stocks:
            metrics = performance_metrics[stock]
            prompt += f"\n{stock}:\n"
            prompt += f"- Annualized Return: {metrics['annualized_return']:.2%}\n"
            prompt += f"- Annualized Volatility: {metrics['annualized_volatility']:.2%}\n"
            prompt += f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            
        # Add portfolio metrics
        prompt += f"\nPortfolio Metrics:\n"
        prompt += f"- Expected Annualized Return: {expected_return * 252:.2%}\n"
        prompt += f"- Expected Annualized Volatility: {portfolio_volatility * np.sqrt(252):.2%}\n"
        prompt += f"- Expected Sharpe Ratio: {portfolio_sharpe * np.sqrt(252):.2f}\n"
        prompt += f"- Average Correlation Between Stocks: {avg_correlation:.2f}\n"
        
        # Add instructions
        prompt += """
Based on this information, please provide a concise rationale for this portfolio selection. 
Explain why these stocks were selected and how they work together as a portfolio. 
Focus on diversification benefits, risk-return characteristics, and overall portfolio strategy.
Keep your explanation under 200 words.
"""

        # Get rationale from LLM
        rationale = self._get_llm_response(prompt)
        
        return rationale
        
    def receive_feedback(self, feedback):
        """
        Receive and process feedback from manager agent.
        
        Args:
            feedback (dict): Feedback from manager agent
            
        Returns:
            bool: True if feedback was processed successfully
        """
        if not feedback:
            return False
            
        # Update memory based on feedback
        importance_change = feedback.get("importance_change", 0.0)
        
        # Get recent memory events
        recent_events = self.procedural_memory.get_events_by_type("portfolio_selection")
        if not recent_events:
            return False
            
        # Update importance of most recent event
        recent_events.sort(key=lambda x: x["timestamp"], reverse=True)
        if recent_events:
            most_recent_event = recent_events[0]
            self.procedural_memory.update_importance(most_recent_event["id"], importance_change)
            
            self.logger.info(f"Updated importance of event {most_recent_event['id']} by {importance_change}")
            
        return True
        
    def update_beliefs(self, beliefs_update):
        """
        Update agent's beliefs based on feedback from risk control.
        
        Args:
            beliefs_update (dict): Updated beliefs
            
        Returns:
            bool: True if beliefs were updated successfully
        """
        relevant_aspects = ["historical_momentum", "market_data_analysis", "other_aspects"]
        present_aspects = [aspect for aspect in relevant_aspects if aspect in beliefs_update]
        
        if not beliefs_update or not present_aspects:
            return False
            
        # Create belief update prompt
        belief_text = ""
        for aspect in present_aspects:
            belief_text += f"{aspect.replace('_', ' ').title()}: {beliefs_update[aspect]}\n\n"
        
        prompt = f"""You are a portfolio analyst specializing in stock selection. You have received updated investment beliefs:

{belief_text}

Based on this feedback, update your approach to portfolio selection. Consider:
1. How should you adjust your stock selection criteria?
2. What aspects of correlation and diversification should you emphasize more?
3. How should your portfolio weighting strategy change?

Provide a concise summary of how you will adjust your portfolio selection approach.
"""
        # Get response from LLM
        response = self._get_llm_response(prompt)
        
        # Store updated belief in procedural memory
        belief_text = f"Updated Belief - Portfolio Selection: {belief_text}\n\nImplementation Plan: {response}"
        belief_embedding = self._generate_embedding(belief_text)
        
        memory_id = self.add_to_procedural_memory(
            belief_text,
            "belief_update",
            belief_embedding,
            importance=2.0  # High importance for belief updates
        )
        
        self.logger.info(f"Updated portfolio selection beliefs with ID {memory_id}")
        return True


if __name__ == "__main__":
    import pandas as pd
    stock_data = pd.read_csv("data/all_sp500_prices_2000_2024_delisted_include.csv")

    # Group by symbol, create a dict: key is symbol, value is DataFrame
    market_data = {sym: sub_df.reset_index(drop=True) for sym, sub_df in stock_data.groupby('symbol')}

    selection_agent = StockSelectionAgent("stock_selection_agent", target_symbols=["AAPL", "MSFT", "NFLX", "AMZN", "TSLA"])
    print(selection_agent.process(market_data, num_stocks=3)['selected_stocks'])

    # Selected
    # symbols: ['AAPL', 'ABC', 'ABMD', 'ADS', 'ADSK', 'AGN', 'ALGN', 'ALXN', 'AMD', 'AMT', 'AMZN', 'APOL', 'APTV', 'AYE',
    #           'AZO', 'BIIB', 'BKNG', 'BSX', 'BXP', 'CARR', 'CBOE', 'CELG', 'CF', 'CME', 'CMG', 'COP', 'CPWR', 'DLTR',
    #           'EA', 'EIX', 'EOG', 'EP', 'ETR', 'ETSY', 'FE', 'GILD', 'GME', 'HSY', 'ICE', 'KDP', 'KO', 'KR', 'LLY',
    #           'LMT', 'LUV', 'LW', 'MCD', 'MCK', 'MHS', 'MJN', 'MO', 'MPC', 'MRNA', 'MU', 'NEE', 'NEM', 'NFLX', 'NRG',
    #           'NVDA', 'PBCT', 'PGR', 'PSA', 'RAI', 'REGN', 'ROST', 'RX', 'SBUX', 'SPG', 'TGT', 'TJX', 'TSLA', 'TXU',
    #           'ULTA', 'VLO', 'WM', 'WMB', 'WST', 'XOM']
    # Total
    # selected
    # symbols: 78