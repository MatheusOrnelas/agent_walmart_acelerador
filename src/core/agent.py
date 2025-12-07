import os
import time
import logging
from typing import List, Optional, Type
from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from .state import BaseState
from .utils import save_graph_figure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BaseLangGraphAgent(ABC):
    """
    Generic Base Class for LangGraph Agents.
    """
    
    def __init__(self, project_id: Optional[str] = None, location: str = "us-central1", deploy_type: str = "gcp"):
        """
        Initialize the LangGraph Agent.
        
        Args:
            project_id: Google Cloud Project ID. If None, tries to read from environment.
            location: Google Cloud Region.
            deploy_type: Deployment target ("gcp" or "databricks"). Defaults to "gcp".
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
        self.location = location or os.getenv("LOCATION", "us-central1")
        self.deploy_type = deploy_type or os.getenv("DEPLOY_TYPE", "gcp")
        
        if self.deploy_type == "gcp":
            self.setup_vertex_ai()
        else:
            self.logger.info(f"Deploy type is '{self.deploy_type}', skipping Vertex AI initialization.")
            
        self.app: Optional[CompiledStateGraph] = None
        self.setup_graph()
    
    def setup_vertex_ai(self) -> None:
        """Configure Vertex AI."""
        import vertexai
        if not self.project_id:
            self.logger.warning("PROJECT_ID not found. Vertex AI initialization might fail if not running in GCP environment with default credentials.")
        else:
            vertexai.init(project=self.project_id, location=self.location)
            self.logger.info(f"Vertex AI initialized with project: {self.project_id}, location: {self.location}")
    
    @abstractmethod
    def setup_graph(self) -> None:
        """
        Configure the state graph. 
        Must assign the compiled graph to self.app
        Example:
            workflow = StateGraph(MyState)
            ... add nodes and edges ...
            self.app = workflow.compile()
        """
        pass
    
    def chat(self, user_input: str, thread_id: str = "default") -> str:
        """
        Process a user message.
        
        Args:
            user_input: User message
            thread_id: Thread ID for maintaining history
            
        Returns:
            Chatbot response
        """
        if not self.app:
            raise RuntimeError("Graph not initialized. Ensure setup_graph() sets self.app")

        try:
            self.logger.info(f"Processing chat message for thread: {thread_id}")
            
            # Thread configuration
            config = {"configurable": {"thread_id": thread_id}}
            
            # Invoke the graph
            result = self.app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config
            )
            
            # Extract response
            # Assuming the standard structure where messages are in the state
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    response = last_message.content
                else:
                    response = str(last_message.content)
            else:
                 response = "No response generated."
            
            self.logger.info(f"Generated response for thread: {thread_id}")
            return response
                
        except Exception as e:
            error_msg = f"Error processing chat message: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return f"Error: {str(e)}"
    
    def get_chat_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """Get conversation history."""
        if not self.app:
            return []
            
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # Get current state
            current_state = self.app.get_state(config)
            messages = current_state.values.get("messages", [])
            self.logger.info(f"Retrieved {len(messages)} messages from thread: {thread_id}")
            return messages
        except Exception as e:
            self.logger.error(f"Error retrieving chat history for thread {thread_id}: {str(e)}")
            return []
    
    def clear_memory(self, thread_id: str = "default") -> str:
        """Clear conversation history by creating new thread."""
        new_thread_id = f"thread_{int(time.time())}"
        self.logger.info(f"Cleared memory for thread {thread_id}, new thread: {new_thread_id}")
        return new_thread_id
        
    def save_graph(self, path: str = "graph_architecture.png"):
        """Save the graph visualization."""
        if self.app:
            save_graph_figure(self.app, path)
