"""
Action module for FINCON agents.
Defines how agents take actions based on processed information.
"""

import json
import logging

class ActionModule:
    """
    Action module for FINCON agents.
    
    This module defines how each agent processes information
    and generates appropriate actions or outputs.
    """
    
    def __init__(self, agent_type):
        """
        Initialize the action module.
        
        Args:
            agent_type (str): Type of agent
        """
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"action_module_{agent_type}")
    
    def generate_action_template(self, action_type):
        """
        Generate template for a specific action type.
        
        Args:
            action_type (str): Type of action to generate template for
            
        Returns:
            str: Action template text
        """
        templates = {
            "trading_decision": self._get_trading_decision_template(),
            "news_analysis": self._get_news_analysis_template(),
            "filing_analysis": self._get_filing_analysis_template(),
            "ecc_analysis": self._get_ecc_analysis_template(),
            "market_data_analysis": self._get_market_data_analysis_template(),
            "portfolio_selection": self._get_portfolio_selection_template(),
            "risk_alert": self._get_risk_alert_template(),
            "belief_update": self._get_belief_update_template()
        }
        
        return templates.get(action_type, "Unknown action type.")
    
    def _get_trading_decision_template(self):
        """Get template for trading decisions (manager agent)."""
        return """
TRADING DECISION FORMAT

Date: [CURRENT_DATE]
Symbols: [SYMBOL_LIST]

For each symbol, provide the following:

Symbol: [SYMBOL]
Decision: [BUY/SELL/HOLD]
Position Size: [POSITION_SIZE]
Reasoning:
[Provide detailed reasoning for the trading decision, including key factors that influenced the decision]

Risk Assessment:
[Provide assessment of the potential risks associated with this decision]

Contributing Insights:
[List the most valuable analyst insights that contributed to this decision]

Self-Reflection:
[Reflect on previous decisions for this symbol and how they inform the current decision]
"""
    
    def _get_news_analysis_template(self):
        """Get template for news analysis (news analyst)."""
        return """
NEWS ANALYSIS FORMAT

Date: [CURRENT_DATE]
Symbols: [SYMBOL_LIST]
Period Covered: [TIME_PERIOD]

For each symbol, provide the following:

Symbol: [SYMBOL]
Key News Items:
[List the most important news items for this symbol, with dates]

News Sentiment:
- Overall Sentiment: [POSITIVE/NEGATIVE/NEUTRAL]
- Sentiment Score: [SCORE]
- Sentiment Trend: [IMPROVING/DETERIORATING/STABLE]

Key Insights:
[Extract the most important insights from the news that may impact trading decisions]

Potential Market Impact:
[Assess how the news might impact the stock price and market perception]

Recommendation:
[Based on news analysis, provide a recommendation]

Confidence Level:
[HIGH/MEDIUM/LOW - indicate confidence in this analysis]
"""
    
    def _get_filing_analysis_template(self):
        """Get template for SEC filing analysis (filing analyst)."""
        return """
SEC FILING ANALYSIS FORMAT

Date: [CURRENT_DATE]
Company: [COMPANY_NAME]
Symbol: [SYMBOL]
Filing Type: [10-K/10-Q]
Period Ending: [PERIOD_END_DATE]

Financial Performance:
[Summarize key financial metrics and performance indicators]

Management Discussion Analysis:
[Extract key points from Management's Discussion and Analysis section]

Risk Factors:
[Identify key risk factors mentioned in the filing]

Forward-Looking Statements:
[Extract important forward-looking statements and guidance]

Notable Changes:
[Highlight significant changes compared to previous filings]

Key Insights:
[Extract the most important insights that may impact trading decisions]

Recommendation:
[Based on filing analysis, provide a recommendation]

Confidence Level:
[HIGH/MEDIUM/LOW - indicate confidence in this analysis]
"""
    
    def _get_ecc_analysis_template(self):
        """Get template for earnings call analysis (ECC analyst)."""
        return """
EARNINGS CALL ANALYSIS FORMAT

Date: [CURRENT_DATE]
Company: [COMPANY_NAME]
Symbol: [SYMBOL]
Quarter: [QUARTER]
Call Date: [CALL_DATE]

Key Highlights:
[Summarize the most important points from the earnings call]

Financial Results:
[Extract key financial results discussed in the call]

Guidance and Outlook:
[Summarize forward guidance and management outlook]

Audio Analysis:
- Management Tone: [CONFIDENT/CAUTIOUS/NERVOUS/etc.]
- Confidence Level: [HIGH/MEDIUM/LOW]
- Speech Patterns: [OBSERVATIONS ABOUT SPEAKING PATTERNS]

Q&A Insights:
[Extract notable questions and answers from the Q&A section]

Key Insights:
[Extract the most important insights that may impact trading decisions]

Recommendation:
[Based on earnings call analysis, provide a recommendation]

Confidence Level:
[HIGH/MEDIUM/LOW - indicate confidence in this analysis]
"""
    
    def _get_market_data_analysis_template(self):
        """Get template for market data analysis (data analyst)."""
        return """
MARKET DATA ANALYSIS FORMAT

Date: [CURRENT_DATE]
Symbol: [SYMBOL]
Period Analyzed: [TIME_PERIOD]

Price Analysis:
- Current Price: [PRICE]
- Price Change (1D): [CHANGE_1D]
- Price Change (1W): [CHANGE_1W]
- Price Change (1M): [CHANGE_1M]
- 52-Week Range: [52W_LOW] - [52W_HIGH]

Volume Analysis:
- Current Volume: [VOLUME]
- Average Volume (10D): [AVG_VOLUME_10D]
- Volume Trend: [INCREASING/DECREASING/STABLE]

Technical Indicators:
- Moving Averages: [MA_ANALYSIS]
- RSI: [RSI_VALUE] - [RSI_INTERPRETATION]
- MACD: [MACD_ANALYSIS]
- Bollinger Bands: [BB_ANALYSIS]

Momentum Analysis:
- Short-term Momentum: [POSITIVE/NEGATIVE/NEUTRAL]
- Medium-term Momentum: [POSITIVE/NEGATIVE/NEUTRAL]
- Long-term Momentum: [POSITIVE/NEGATIVE/NEUTRAL]

Volatility Analysis:
- Historical Volatility: [VOLATILITY]
- Implied Volatility: [IMPLIED_VOLATILITY]
- Conditional Value at Risk (CVaR): [CVAR]

Market Context:
[Provide broader market context relevant to this symbol]

Key Insights:
[Extract the most important insights that may impact trading decisions]

Recommendation:
[Based on technical analysis, provide a recommendation]

Confidence Level:
[HIGH/MEDIUM/LOW - indicate confidence in this analysis]
"""
    
    def _get_portfolio_selection_template(self):
        """Get template for portfolio selection (stock selection agent)."""
        return """
PORTFOLIO SELECTION FORMAT

Date: [CURRENT_DATE]
Portfolio ID: [PORTFOLIO_ID]
Selection Criteria: [CRITERIA_DESCRIPTION]

Selected Stocks:
[List of selected stock symbols]

Selection Rationale:
[Explain the rationale behind the stock selection]

Correlation Analysis:
[Summary of correlation analysis between selected stocks]

Risk Diversification:
[Explanation of how the selected portfolio achieves risk diversification]

Sector Allocation:
[Breakdown of sector allocation in the portfolio]

Recommended Weights:
[List recommended portfolio weights for each stock]

Expected Performance:
[Projection of expected portfolio performance]

Risk Assessment:
[Assessment of portfolio risk metrics]

Confidence Level:
[HIGH/MEDIUM/LOW - indicate confidence in this portfolio selection]
"""
    
    def _get_risk_alert_template(self):
        """Get template for risk alerts (risk control agent)."""
        return """
RISK ALERT FORMAT

Date: [CURRENT_DATE]
Alert Level: [HIGH/MEDIUM/LOW]
Affected Symbols: [SYMBOL_LIST]

Risk Metrics:
- CVaR: [CVAR_VALUE]
- CVaR Change: [CVAR_CHANGE]
- Other Risk Metrics: [OTHER_METRICS]

Alert Trigger:
[Explain what triggered this risk alert]

Market Context:
[Provide broader market context relevant to this risk alert]

Potential Impact:
[Assess potential impact on trading positions]

Recommended Actions:
[List recommended actions to mitigate risk]

Alert Expiration:
[Specify when this alert should be reevaluated]
"""
    
    def _get_belief_update_template(self):
        """Get template for belief updates (risk control agent)."""
        return """
INVESTMENT BELIEF UPDATE FORMAT

Date: [CURRENT_DATE]
Episode: [EPISODE_NUMBER]
Previous Episode Performance: [BETTER/WORSE]

Updated Investment Beliefs:

Historical Momentum:
[Update beliefs about using momentum indicators]

News Insights:
[Update beliefs about interpreting news information]

Filing Information:
[Update beliefs about using SEC filing information]

Earnings Call Insights:
[Update beliefs about interpreting earnings call information]

Market Data Analysis:
[Update beliefs about technical analysis and market data]

Other Factors:
[Update beliefs about any other relevant factors]

Update Reasoning:
[Explain the reasoning behind these belief updates based on episode performance comparison]

Priority Changes:
[Highlight the most important changes in investment beliefs]

Implementation Guidance:
[Provide guidance on how to implement these updated beliefs in future decisions]
"""
    
    def format_action(self, action_data, action_type):
        """
        Format action data into standardized output.
        
        Args:
            action_data (dict): Action data
            action_type (str): Type of action
            
        Returns:
            dict: Formatted action data
        """
        try:
            formatted_action = {
                "action_type": action_type,
                "timestamp": action_data.get("timestamp", ""),
                "agent_id": action_data.get("agent_id", ""),
                "content": action_data.get("content", {}),
                "metadata": {
                    "confidence": action_data.get("confidence", 0.0),
                    "processing_time": action_data.get("processing_time", 0.0),
                    "source_data": action_data.get("source_data", [])
                }
            }
            
            self.logger.info(f"Formatted {action_type} action for {formatted_action['agent_id']}")
            return formatted_action
            
        except Exception as e:
            self.logger.error(f"Error formatting action: {str(e)}")
            return {"error": str(e)}
    
    def validate_action(self, action_data, action_type):
        """
        Validate action data against schema requirements.
        
        Args:
            action_data (dict): Action data to validate
            action_type (str): Type of action
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Implement validation logic for different action types
        # This could be expanded with a more robust validation framework
        required_fields = {
            "trading_decision": ["symbol", "decision", "position_size", "reasoning"],
            "news_analysis": ["symbol", "sentiment", "key_insights"],
            "filing_analysis": ["symbol", "key_insights", "recommendation"],
            "ecc_analysis": ["symbol", "key_highlights", "recommendation"],
            "market_data_analysis": ["symbol", "technical_indicators", "recommendation"],
            "portfolio_selection": ["selected_stocks", "rationale", "weights"],
            "risk_alert": ["alert_level", "metrics", "recommendations"],
            "belief_update": ["updated_beliefs", "reasoning"]
        }
        
        if action_type not in required_fields:
            self.logger.error(f"Unknown action type: {action_type}")
            return False
        
        content = action_data.get("content", {})
        for field in required_fields[action_type]:
            if field not in content:
                self.logger.error(f"Missing required field '{field}' in {action_type} action")
                return False
                
        return True