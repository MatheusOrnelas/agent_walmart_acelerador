from typing import List, Dict, Any, Optional
from src.core.state import BaseState

class WalmartState(BaseState):
    """State for Walmart Agent"""
    # Input
    intent: str
    entities: Dict[str, Any]
    
    # Processing
    sql_query: str
    db_results: Any # DataFrame or list of dicts
    
    # Outputs
    final_response: str
    
    # Error handling
    error: Optional[str]

