import logging
import time
import json
from typing import Dict, List, Any
from langchain_core.messages import AIMessage
from .state import WalmartState
from .chains import start_chain, sql_gen_chain, response_chain
from src.services.databricks import DatabricksService
from src.monitoring.service_monitor import ServiceMonitor, FailureType

# Initialize Logger and Monitor
logger = logging.getLogger(__name__)
monitor = ServiceMonitor(service_name="walmart-agent-nodes")

# Initialize Databricks Service (Mock credentials for example)
# In production, fetch these from env vars
# Configurar Credenciais do Databricks (Para o conector SQL)
# Pegue esses valores em Compute -> Advanced Options -> JDBC/ODBC
databricks_service = DatabricksService(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"), # Seu HTTP Path longo
    access_token=os.getenv("DATABRICKS_TOKEN")  # Seu token gerado 
)

def _log_node_start(node_name: str, state: WalmartState) -> str:
    """Helper to log node start with consistent formatting (Reused pattern)"""
    node_id = f"{node_name}_{int(time.time() * 1000)}"
    msg_count = len(state.get('messages', []))
    logger.info(f"Node start: name={node_name} id={node_id} msg_count={msg_count}")
    return node_id

def _log_node_end(node_name: str, node_id: str, success: bool = True, error_msg: str = None):
    """Helper to log node completion (Reused pattern)"""
    status = "success" if success else "error"
    if error_msg:
        logger.error(f"Node end: name={node_name} id={node_id} status={status} error='{error_msg}'")
    else:
        logger.info(f"Node end: name={node_name} id={node_id} status={status}")

def start_node(state: WalmartState) -> WalmartState:
    """Start Node: Intent Classification"""
    node_id = _log_node_start("START_NODE", state)
    try:
        messages = state["messages"]
        last_message = messages[-1].content
        
        result = start_chain.invoke({"messages": last_message})
        
        state["intent"] = result.get("intent", "general_info")
        state["entities"] = result.get("entities", {})
        
        _log_node_end("START_NODE", node_id, True)
        return state
    except Exception as e:
        _log_node_end("START_NODE", node_id, False, str(e))
        state["error"] = str(e)
        return state

def sql_generator_node(state: WalmartState) -> WalmartState:
    """Generate SQL based on natural language"""
    node_id = _log_node_start("SQL_GEN_NODE", state)
    try:
        messages = state["messages"]
        last_message = messages[-1].content
        
        sql_query = sql_gen_chain.invoke({"messages": last_message})
        
        # Basic cleaning
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        state["sql_query"] = sql_query
        logger.info(f"Generated SQL: {sql_query}")
        
        _log_node_end("SQL_GEN_NODE", node_id, True)
        return state
    except Exception as e:
        _log_node_end("SQL_GEN_NODE", node_id, False, str(e))
        state["error"] = str(e)
        return state

def executor_node(state: WalmartState) -> WalmartState:
    """Execute SQL on Databricks"""
    node_id = _log_node_start("EXECUTOR_NODE", state)
    sql_query = state.get("sql_query")
    
    # Monitor start
    req_id = monitor.log_attempt_start("EXECUTE_SQL", "databricks_gold_sales", {"query": sql_query})
    
    try:
        start_time = time.time()
        
        # Execute
        df = databricks_service.execute_query(sql_query, operation_name=f"NODE_{node_id}")
        
        # Convert to simple format for LLM
        if not df.empty:
            results = df.head(10).to_dict(orient="records")
        else:
            results = []
            
        state["db_results"] = results
        
        # Monitor success
        duration_ms = int((time.time() - start_time) * 1000)
        monitor.log_attempt_success(req_id, duration_ms, response_size=len(results))
        
        _log_node_end("EXECUTOR_NODE", node_id, True)
        return state
        
    except Exception as e:
        # Monitor failure
        monitor.log_attempt_failure(req_id, FailureType.DB_ERROR, str(e))
        _log_node_end("EXECUTOR_NODE", node_id, False, str(e))
        state["error"] = f"Database error: {str(e)}"
        state["db_results"] = []
        return state

def response_node(state: WalmartState) -> Dict[str, List[AIMessage]]:
    """Generate Final Response"""
    node_id = _log_node_start("RESPONSE_NODE", state)
    try:
        messages = state["messages"]
        last_message = messages[-1].content
        
        response = response_chain.invoke({
            "messages": last_message,
            "sql_query": state.get("sql_query", "N/A"),
            "results": state.get("db_results", "No results or Error occurred")
        })
        
        _log_node_end("RESPONSE_NODE", node_id, True)
        return {"messages": messages + [AIMessage(content=response)]}
        
    except Exception as e:
        _log_node_end("RESPONSE_NODE", node_id, False, str(e))
        return {"messages": messages + [AIMessage(content=f"Error generating response: {str(e)}")]}

