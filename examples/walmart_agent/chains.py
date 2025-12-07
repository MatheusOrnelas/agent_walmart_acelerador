import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate

from .prompts import sql_gen_template, response_template, start_template

# def setup_llm():
#     """
#     Configure LLM based on deployment type.
#     Supports 'gcp' (Vertex AI) and 'databricks' (Databricks Serving).
#     """
#     deploy_type = os.getenv("DEPLOY_TYPE", "gcp")
    
#     if deploy_type == "databricks":
#         try:
#             from langchain_community.chat_models import ChatDatabricks
#             # Ensure DATABRICKS_HOST and DATABRICKS_TOKEN are set in env
#             return ChatDatabricks(
#                 endpoint="databricks-meta-llama-3-70b-instruct", # Example endpoint
#                 temperature=0.2,
#                 max_tokens=2048
#             )
#         except ImportError:
#             raise RuntimeError("Install 'langchain-databricks' or 'langchain-community' to use Databricks LLM.")
            
#     else:
#         # Default to GCP/Vertex AI
#         import vertexai
#         from langchain_google_vertexai import ChatVertexAI
        
#         project_id = os.getenv("PROJECT_ID", "ra-data-analytics") 
#         location = os.getenv("LOCATION", "us-central1")
#         vertexai.init(project=project_id, location=location)
        
#         return ChatVertexAI(
#             model_name="gemini-1.5-pro",
#             temperature=0.2,
#             max_output_tokens=2048
#         )

def setup_llm():
    """
    Configura o LLM com base na variável de ambiente DEPLOY_TYPE.
    Opções suportadas: 'databricks', 'gcp', 'openai'.
    """
    deploy_type = os.getenv("DEPLOY_TYPE", "databricks").lower()
    
    logger.info(f"🔄 Inicializando LLM com Provider: {deploy_type.upper()}")

    if deploy_type == "databricks":
        try:
            from langchain_databricks import ChatDatabricks
            # Se der erro de import, tente: from langchain_community.chat_models import ChatDatabricks
        except ImportError:
            try:
                from langchain_community.chat_models import ChatDatabricks
            except ImportError:
                raise ImportError("Instale 'langchain-databricks' ou 'langchain-community' para usar modelos Databricks.")

        # Nome do modelo padrão. No Databricks Foundation Models, os nomes comuns são:
        # - databricks-meta-llama-3-70b-instruct
        # - databricks-dbrx-instruct
        # - databricks-mixtral-8x7b-instruct
        model_name = os.getenv("DATABRICKS_LLM_MODEL", "databricks-meta-llama-3-70b-instruct")
        
        return ChatDatabricks(
            endpoint=model_name,
            temperature=0.1,
            max_tokens=2048
        )

    elif deploy_type == "gcp":
        try:
            import vertexai
            from langchain_google_vertexai import ChatVertexAI
        except ImportError:
            raise ImportError("Instale 'langchain-google-vertexai' para usar GCP Vertex AI.")
        
        project_id = os.getenv("PROJECT_ID")
        location = os.getenv("LOCATION", "us-central1")
        
        if not project_id:
            logger.warning("⚠️ PROJECT_ID não definido para GCP. A autenticação pode falhar.")

        # vertexai.init(project=project_id, location=location) # Opcional se as credenciais estiverem certas
        
        return ChatVertexAI(
            model_name="gemini-1.5-pro",
            temperature=0.2,
            max_output_tokens=2048
        )

    elif deploy_type == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Instale 'langchain-openai' para usar modelos OpenAI.")
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY é obrigatória para deploy_type='openai'")
            
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.1
        )
    
    else:
        raise ValueError(f"DEPLOY_TYPE '{deploy_type}' não suportado. Use: databricks, gcp, openai")



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
