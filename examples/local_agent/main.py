import os
import sys
from typing import TypedDict, List
from dotenv import load_dotenv

# Add the project root to python path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.agent import BaseLangGraphAgent
from src.core.state import BaseState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# 1. Define Custom State (optional, can just use BaseState)
class AgentState(BaseState):
    user_name: str

# 2. Define Nodes
def start_node(state: AgentState):
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else HumanMessage(content="")
    print(f"--- Processing message: {last_message.content} ---")
    return {"user_name": "User"} # Update state

def chatbot_node(state: AgentState):
    # In a real scenario, this would call an LLM
    # llm = ChatVertexAI(model_name="gemini-pro")
    # response = llm.invoke(state["messages"])
    
    # For this example, we return a simple echo + static text
    messages = state["messages"]
    last_content = messages[-1].content
    response_content = f"Echo: {last_content} (Processed by Local Agent)"
    
    return {"messages": [AIMessage(content=response_content)]}

# 3. Define the Agent Class
class LocalExampleAgent(BaseLangGraphAgent):
    def setup_graph(self):
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("start", start_node)
        workflow.add_node("chatbot", chatbot_node)
        
        # Set entry point
        workflow.set_entry_point("start")
        
        # Add edges
        workflow.add_edge("start", "chatbot")
        workflow.add_edge("chatbot", END)
        
        # Compile
        self.app = workflow.compile()
        self.save_graph("local_agent_architecture.png")

def main():
    print("Initializing Local Agent...")
    try:
        # Initialize agent (requires Google Cloud Credentials for Vertex AI init in BaseAgent, 
        # but we can suppress errors if just testing logic depending on implementation)
        agent = LocalExampleAgent(project_id="my-project", location="us-central1")
        
        print("Agent initialized. Type 'quit' to exit.")
        
        # Chat loop
        thread_id = "test_thread_1"
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            response = agent.chat(user_input, thread_id=thread_id)
            print(f"Agent: {response}")
            
    except Exception as e:
        print(f"Failed to initialize or run agent: {e}")
        print("Note: Ensure you have Google Cloud Credentials configured if using Vertex AI features.")

if __name__ == "__main__":
    main()

