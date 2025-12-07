import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from .prompts import sql_gen_template, response_template, start_template

def setup_llm():
    """
    Configure LLM based on deployment type.
    Supports 'gcp' (Vertex AI) and 'databricks' (Databricks Serving).
    """
    deploy_type = os.getenv("DEPLOY_TYPE", "gcp")
    
    if deploy_type == "databricks":
        try:
            from langchain_community.chat_models import ChatDatabricks
            # Ensure DATABRICKS_HOST and DATABRICKS_TOKEN are set in env
            return ChatDatabricks(
                endpoint="databricks-meta-llama-3-70b-instruct", # Example endpoint
                temperature=0.2,
                max_tokens=2048
            )
        except ImportError:
            raise RuntimeError("Install 'langchain-databricks' or 'langchain-community' to use Databricks LLM.")
            
    else:
        # Default to GCP/Vertex AI
        import vertexai
        from langchain_google_vertexai import ChatVertexAI
        
        project_id = os.getenv("PROJECT_ID", "ra-data-analytics") 
        location = os.getenv("LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)
        
        return ChatVertexAI(
            model_name="gemini-1.5-pro",
            temperature=0.2,
            max_output_tokens=2048
        )

# 1. Start/Router Chain
start_prompt = PromptTemplate(
    template=start_template,
    input_variables=["messages"]
)
start_chain = start_prompt | setup_llm() | JsonOutputParser()

# 2. SQL Generation Chain
sql_gen_prompt = PromptTemplate(
    template=sql_gen_template,
    input_variables=["messages"],
    partial_variables={
        "catalog": "walmart_project",
        "schema": "gold_schema"
    }
)
sql_gen_chain = sql_gen_prompt | setup_llm() | StrOutputParser()

# 3. Response Generation Chain
response_prompt = PromptTemplate(
    template=response_template,
    input_variables=["messages", "sql_query", "results"]
)
response_chain = response_prompt | setup_llm() | StrOutputParser()
