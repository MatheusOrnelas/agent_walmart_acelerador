import logging
import os
from langgraph.graph import StateGraph, END
from src.core.agent import BaseLangGraphAgent
from .state import WalmartState
from .nodes import start_node, sql_generator_node, executor_node, response_node

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
        
        # Conditional routing logic (simple for now)
        def route_intent(state: WalmartState):
            if state.get("error"):
                return "responder" # Fail fast
            if state.get("intent") == "sales_query":
                return "sql_gen"
            else:
                # For general info, skip SQL (or handle differently)
                # Here we force SQL gen for demonstration
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
        self.save_graph("walmart_agent_architecture.png")
        logger.info("Walmart Agent Graph compiled successfully")

if __name__ == "__main__":
    # Quick test
    # deploy_type can be passed here or via env var DEPLOY_TYPE
    deploy_type = os.getenv("DEPLOY_TYPE", "gcp")
    print(f"Initializing Agent with deploy_type: {deploy_type}")
    
    agent = WalmartAgent(project_id="mock-project", deploy_type=deploy_type)
    print("Agent initialized. Running test query...")
    response = agent.chat("How were the sales in region North last week?")
    print(f"Response: {response}")
