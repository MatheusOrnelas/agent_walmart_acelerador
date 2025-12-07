import logging
import os
from typing import Any, List, Dict, Optional
from datetime import datetime
import pandas as pd
from .http_reader import DataReader, CircuitBreaker

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabricksService(DataReader):
    """
    Service for interacting with Databricks SQL.
    Uses databricks-sql-connector.
    """
    
    def __init__(self, server_hostname: str, http_path: str, access_token: str, 
                 catalog: str = "walmart_project", schema: str = "gold_schema",
                 enable_circuit_breaker: bool = True):
        self.server_hostname = server_hostname
        self.http_path = http_path
        self.access_token = access_token
        self.catalog = catalog
        self.schema = schema
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30) if enable_circuit_breaker else None
        
        logger.info(f"DatabricksService initialized for {server_hostname} (Catalog: {catalog})")

    def _get_connection(self):
        """
        Creates a connection to Databricks.
        Note: Requires databricks-sql-connector package.
        """
        try:
            from databricks import sql
            return sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token
            )
        except ImportError:
            logger.error("databricks-sql-connector not installed.")
            raise RuntimeError("Install databricks-sql-connector to use DatabricksService")

    def execute_query(self, query: str, operation_name: str = "SQL_EXECUTION") -> pd.DataFrame:
        """
        Executes a SQL query and returns a pandas DataFrame.
        Protected by circuit breaker.
        """
        if self.circuit_breaker:
            return self.circuit_breaker.call(self._execute_query_internal, query, operation_name)
        else:
            return self._execute_query_internal(query, operation_name)

    def _execute_query_internal(self, query: str, operation_name: str) -> pd.DataFrame:
        """Internal execution logic"""
        logger.info(f"[{operation_name}] Executing query: {query[:100]}...")
        start_time = datetime.now()
        
        try:
            # Mock execution if credentials are dummies (for testing/accelerator generation)
            if self.server_hostname == "mock_server":
                logger.warning("Executing in MOCK mode")
                return pd.DataFrame({"mock_col": ["mock_val"], "sales": [1000]})

            with self._get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    # Fetch result as pandas DataFrame
                    # databricks-sql-connector cursors usually support fetchall or direct integration
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    df = pd.DataFrame(data, columns=columns)
                    
                    duration = (datetime.now() - start_time).total_seconds()
                    logger.info(f"[{operation_name}] Success in {duration:.2f}s. Rows: {len(df)}")
                    return df
                    
        except Exception as e:
            logger.error(f"[{operation_name}] Failed: {str(e)}")
            raise e

    def read(self) -> pd.DataFrame:
        """Default read implementation - specific to the primary table"""
        query = f"SELECT * FROM {self.catalog}.{self.schema}.gold_sales_analytics LIMIT 10"
        return self.execute_query(query, "READ_DEFAULT")

    def get_sales_analytics(self, limit: int = 100) -> pd.DataFrame:
        """Specific method for the requested Walmart table"""
        query = f"SELECT * FROM {self.catalog}.{self.schema}.gold_sales_analytics LIMIT {limit}"
        return self.execute_query(query, "GET_SALES_ANALYTICS")

