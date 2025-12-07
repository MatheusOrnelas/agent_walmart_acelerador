from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class BaseState(TypedDict):
    """
    Base state for LangGraph agents.
    Users should extend this class to add their specific state fields.
    """
    messages: List[BaseMessage]
    execution_error: Optional[str]

