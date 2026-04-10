# Tool Descriptions

Tools available:

- read_market_news(date: str) -> str: Read raw market news articles for the week (verbose, for audit)
- compute_ml_signals(date: str) -> str: Get ML signals per sub_category (momentum, heat, composite, trend)
- check_last_week_pnl() -> str: Get last week's holdings and return (behavioural memory)
- retrieve_history(date: str, query: str) -> str: Find similar historical investment cases via Memos vector search
- get_industry_top_news(date: str, industry: str, top_k: int = 3) -> str: Get top-k high-confidence news for a specific sub_category (compressed format for LLM)
- get_etf_candidates(industry: str, date: str = "") -> str: Get candidate ETFs for a sub_category from the real backtest price universe; when date is provided, only return ETFs tradable for that week
- store_decision(date: str, decision: str, context: str = "") -> str: Store an investment decision in Memos for future retrieval
