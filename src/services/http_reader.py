import requests
import pandas as pd
from typing import Optional, Union, Dict, Any
import json
from abc import ABC, abstractmethod
import logging
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import os
from datetime import datetime

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Detect Cloud Run environment for optimized logging
IS_CLOUD_RUN = os.getenv('K_SERVICE') is not None

def format_json_for_logging(data: dict, compact: bool = IS_CLOUD_RUN) -> str:
    """Format JSON for logging - compact for Cloud Run, detailed for local."""
    if compact:
        return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
    else:
        return json.dumps(data, indent=2, ensure_ascii=False)

def format_headers_for_logging(headers: dict, compact: bool = IS_CLOUD_RUN) -> str:
    """Format headers for logging."""
    if compact:
        # Show only important headers in compact format
        important = {k: v for k, v in headers.items() if k.lower() in ['content-type', 'authorization', 'x-api-key']}
        return json.dumps(important, separators=(',', ':'))
    else:
        return json.dumps(dict(headers), indent=2)

class DataReader(ABC):
    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass

class CircuitBreaker:
    """Simple circuit breaker for API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        # Check if circuit is open
        if self.state == "OPEN":
            if self.last_failure_time and (
                datetime.now() - self.last_failure_time
            ).seconds < self.timeout:
                raise Exception("Circuit breaker is OPEN - API temporarily unavailable")
            else:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker moving to HALF_OPEN state")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset circuit breaker
            if self.state in ["HALF_OPEN", "OPEN"]:
                logger.info("Circuit breaker reset to CLOSED state")
                self.state = "CLOSED"
                self.failure_count = 0
                self.last_failure_time = None
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
                
            raise e

class HTTPReader(DataReader):
    """
    Makes a POST request to an HTTP API with a JSON payload and parses the HTTP response.
    Generic implementation that can be extended.
    """

    def __init__(self, url: str, timeout: int = 20, max_retries: int = 2, 
                 rate_limit_delay: float = 1.0, backoff_factor: float = 1.5,
                 enable_circuit_breaker: bool = False):
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.backoff_factor = backoff_factor
        
        # Rate limiting - track last request time per URL
        self._last_request_time = {}
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30) if enable_circuit_breaker else None
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
            backoff_factor=backoff_factor,
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"HTTPReader initialized - URL: {url}, Timeout: {timeout}s, Rate limit: {rate_limit_delay}s")

    def _apply_rate_limiting(self, operation_name: str):
        """Apply rate limiting to avoid overwhelming the API"""
        url_key = f"{self.url}:{operation_name}"
        current_time = time.time()
        
        if url_key in self._last_request_time:
            time_since_last = current_time - self._last_request_time[url_key]
            if time_since_last < self.rate_limit_delay:
                wait_time = self.rate_limit_delay - time_since_last
                if IS_CLOUD_RUN:
                    logger.info(f"Rate limit: op={operation_name} waiting={wait_time:.1f}s")
                else:
                    logger.info(f"Rate limiting: waiting {wait_time:.1f}s before {operation_name}")
                time.sleep(wait_time)
        
        self._last_request_time[url_key] = time.time()

    def make_request_with_circuit_breaker(self, payload: dict, operation_name: str, **kwargs) -> dict:
        """Make request with circuit breaker protection and rate limiting"""
        
        # Apply rate limiting first
        self._apply_rate_limiting(operation_name)
        
        if self.circuit_breaker:
            try:
                return self.circuit_breaker.call(self.make_request, payload, operation_name, **kwargs)
            except Exception as e:
                # If circuit breaker is open, return graceful fallback
                if "Circuit breaker is OPEN" in str(e):
                    return self._create_graceful_fallback(operation_name, "Circuit breaker active - API temporarily unavailable")
                raise e
        else:
            return self.make_request(payload, operation_name, **kwargs)
            
    def _create_graceful_fallback(self, operation_name: str, reason: str) -> dict:
        """Create graceful fallback response when API is unavailable"""
        fallback_data = {
            "status": "unavailable",
            "reason": reason,
            "fallback": True,
            "message": "Data unavailable at the moment due to API instability. Please try again later.",
            "operation": operation_name,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.warning(f"Using graceful fallback for {operation_name}: {reason}")
        return fallback_data

    def make_request(self, payload: dict, operation_name: str, **kwargs) -> dict:
        """
        Generic method to make HTTP requests with detailed logging and error handling.
        
        Args:
            payload (dict): Request payload
            operation_name (str): Name of the operation for logging
            **kwargs: Additional arguments for requests.post (e.g., headers)
            
        Returns:
            dict: Response data or error dict
        """
        headers = kwargs.get('headers', {}).copy()
        if 'Content-Type' not in headers:
            headers['Content-Type'] = 'application/json'
        
        # Add Cloudflare Access headers if credentials are available in environment
        cf_client_id = os.getenv('CF_ACCESS_CLIENT_ID')
        cf_client_secret = os.getenv('CF_ACCESS_CLIENT_SECRET')
        
        if cf_client_id and cf_client_secret:
            headers['CF-Access-Client-Id'] = cf_client_id.strip()
            headers['CF-Access-Client-Secret'] = cf_client_secret.strip()
        
        request_id = int(time.time() * 1000)
        
        if IS_CLOUD_RUN:
            # Compact logging
            safe_headers = {k: 'REDACTED' if 'secret' in k.lower() else v for k, v in headers.items()}
            logger.info(f"[{operation_name}] req_id={request_id} url={self.url}")
            logger.info(f"[{operation_name}] req_id={request_id} payload={format_json_for_logging(payload)}")
        else:
            # Detailed logging
            safe_headers = {k: 'REDACTED' if 'secret' in k.lower() else v for k, v in headers.items()}
            logger.info(f"[{operation_name}] Request ID: {request_id}")
            logger.info(f"[{operation_name}] Making POST request to: {self.url}")
            logger.info(f"[{operation_name}] Payload: {format_json_for_logging(payload)}")
        
        try:
            start_time = time.time()
            
            response = self.session.post(
                self.url, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            response_time = time.time() - start_time
            response_text = response.text
            
            if IS_CLOUD_RUN:
                logger.info(f"[{operation_name}] req_id={request_id} status={response.status_code} time={response_time:.2f}s")
            else:
                logger.info(f"[{operation_name}] Request ID: {request_id} - Response received in {response_time:.2f}s")
                logger.info(f"[{operation_name}] Response Status: {response.status_code}")
            
            # Check HTTP status
            response.raise_for_status()
            
            # Validate response content
            if not response_text or response_text.strip() == "":
                error_msg = "Response is empty or contains only whitespace"
                logger.error(f"[{operation_name}] req_id={request_id} error='{error_msg}'")
                return {"error": f"HTTP request failed: {error_msg}"}
            
            # Check if response looks like JSON
            if not response_text.strip().startswith(('{', '[')):
                content_preview = response_text[:100].replace('\n', ' ').replace('\r', ' ')
                error_msg = f"Response doesn't appear to be JSON. Content preview: {content_preview}"
                logger.error(f"[{operation_name}] req_id={request_id} error='{error_msg}'")
                return {"error": f"HTTP request failed: {error_msg}"}
            
            # Attempt JSON parsing
            try:
                data = response.json()
                return data
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                logger.error(f"[{operation_name}] req_id={request_id} error='{error_msg}'")
                return {"error": f"HTTP request failed: {error_msg}"}
        
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timeout after {self.timeout}s: {str(e)}"
            logger.error(f"[{operation_name}] req_id={request_id} error='{error_msg}'")
            return {"error": f"HTTP request failed: {error_msg}"}
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(f"[{operation_name}] req_id={request_id} error='{error_msg}'")
            return {"error": f"HTTP request failed: {error_msg}"}
            
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error {response.status_code}: {str(e)}"
            logger.error(f"[{operation_name}] req_id={request_id} error='{error_msg}'")
            return {"error": f"HTTP request failed: {error_msg}"}
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"[{operation_name}] req_id={request_id} error='{error_msg}'", exc_info=True)
            return {"error": f"HTTP request failed: {error_msg}"}

    def read(self) -> pd.DataFrame:
        """
        Base read method implementation.
        """
        # This is a placeholder as the base class is generic.
        # Subclasses or instances should define specific read operations.
        return pd.DataFrame()

