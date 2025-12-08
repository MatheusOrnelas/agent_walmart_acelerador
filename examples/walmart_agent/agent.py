import logging
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from src.core.agent import BaseLangGraphAgent
from .state import WalmartState
from .nodes import start_node, sql_generator_node, executor_node, response_node

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalmartAgent(BaseLangGraphAgent):
    """
    Walmart Data Agent implementation.
    Queries Databricks for Sales Analytics.
    """
    
    def setup_graph(self) -> None:
        logger.info("Setting up Walmart Agent Graph")
        
        # Initialize Graph
        workflow = StateGraph(WalmartState)
        
        # Add Nodes
        workflow.add_node("start", start_node)
        workflow.add_node("sql_gen", sql_generator_node)
        workflow.add_node("executor", executor_node)
        workflow.add_node("responder", response_node)
        
        # Add Edges
        workflow.set_entry_point("start")
        
        # Conditional routing logic
        def route_intent(state: WalmartState):
            if state.get("error"):
                return "responder" # Fail fast
            
            intent = state.get("intent")
            if intent == "sales_query":
                return "sql_gen"
            else:
                # Default to SQL gen for this example, but could route to general chat
                return "sql_gen"

        workflow.add_conditional_edges(
            "start",
            route_intent,
            {
                "sql_gen": "sql_gen",
                "responder": "responder"
            }
        )
        
        workflow.add_edge("sql_gen", "executor")
        workflow.add_edge("executor", "responder")
        workflow.add_edge("responder", END)
        
        # Compile
        self.app = workflow.compile()
        try:
            self.save_graph("walmart_agent_architecture.png")
        except Exception:
            logger.warning("Could not save graph visualization (possibly missing dependencies like graphviz)")
            
        logger.info("Walmart Agent Graph compiled successfully")

if __name__ == "__main__":
    # Test execution
    deploy_type = os.getenv("DEPLOY_TYPE", "gcp")
    project_id = os.getenv("PROJECT_ID", "mock-project")
    
    print(f"Initializing Agent with deploy_type: {deploy_type}")
    
    try:
        agent = WalmartAgent(project_id=project_id, deploy_type=deploy_type)
        print("Agent initialized. Running test query...")
        response = agent.chat("How were the sales in region North last week?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Agent failed to start or run: {e}")
