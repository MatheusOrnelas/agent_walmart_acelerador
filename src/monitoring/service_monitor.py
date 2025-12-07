import logging
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import threading

# Detect Cloud Run environment
IS_CLOUD_RUN = os.getenv('K_SERVICE') is not None

# Configure structured logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if not IS_CLOUD_RUN else '%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('service_monitor')

class FailureType(Enum):
    """Generic failure types for service calls"""
    TIMEOUT = "CONNECTION_TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    JSON_DECODE_ERROR = "JSON_DECODE_ERROR"
    ACCESS_DENIED = "ACCESS_DENIED"
    DNS_ERROR = "DNS_RESOLUTION_FAILED"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    DB_ERROR = "DATABASE_ERROR"  # Added for SQL/Databricks support

@dataclass
class RequestAttempt:
    """Structured record of a service request attempt"""
    timestamp: str
    request_id: str
    operation: str
    resource: str  # URL, Table Name, or Resource Identifier
    context: Dict[str, Any]  # Generic context (catalog_id, company_id, sql_query, etc.)
    timeout_seconds: Optional[int] = None
    success: bool = False
    failure_type: Optional[str] = None
    error_message: Optional[str] = None
    response_time_ms: Optional[int] = None
    retry_attempt: int = 0
    max_retries: int = 3

class ServiceMonitor:
    """Generic Monitor for service instability (API, Database, etc.)"""
    
    def __init__(self, service_name: str = "generic-service"):
        self.service_name = service_name
        self.consecutive_failures = 0
        self.total_attempts = 0
        self.total_failures = 0
        self.failure_window = []  # Last N attempts
        self.last_success_time = None
        self.downtime_start = None
        self._lock = threading.Lock()
        
        # Alert configurations
        self.consecutive_failure_threshold = 3
        self.failure_rate_threshold = 0.8  # 80%
        self.window_size = 10
        
        self.log_monitor_initialized()
    
    def log_monitor_initialized(self):
        """Log monitor initialization"""
        log_data = {
            "event": "MONITOR_INITIALIZED",
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "component": "service_monitor",
            "thresholds": {
                "consecutive_failure": self.consecutive_failure_threshold,
                "failure_rate": self.failure_rate_threshold
            }
        }
        
        if IS_CLOUD_RUN:
            logger.info(json.dumps(log_data, separators=(',', ':')))
        else:
            logger.info(f"🔍 Service Monitor initialized for {self.service_name}")
    
    def log_attempt_start(self, operation: str, resource: str, context: Dict[str, Any] = None,
                         timeout_seconds: Optional[int] = None) -> str:
        """
        Log start of a service request attempt.
        
        Args:
            operation: Name of the operation (e.g., "read_id", "execute_sql")
            resource: Target resource (URL, Table name)
            context: Additional metadata (ids, query snippets, etc.)
            
        Returns:
            str: request_id for tracking
        """
        request_id = f"req_{int(time.time() * 1000)}_{threading.get_ident()}"
        context = context or {}
        
        attempt = RequestAttempt(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            operation=operation,
            resource=resource,
            context=context,
            timeout_seconds=timeout_seconds
        )
        
        log_data = {
            "event": "REQUEST_START",
            "request_id": request_id,
            "timestamp": attempt.timestamp,
            "operation": operation,
            "resource": resource,
            "context": context,
            "timeout": timeout_seconds,
            "stats": {
                "consecutive_failures": self.consecutive_failures,
                "total_attempts": self.total_attempts
            }
        }
        
        if IS_CLOUD_RUN:
            logger.info(json.dumps(log_data, separators=(',', ':')))
        else:
            logger.info(f"🌐 REQUEST START: {request_id} | {operation} | {resource}")
        
        return request_id
    
    def log_attempt_success(self, request_id: str, response_time_ms: int, 
                           response_size: Optional[int] = None):
        """Log successful request attempt"""
        with self._lock:
            self.consecutive_failures = 0
            self.total_attempts += 1
            self.last_success_time = datetime.now()
            
            # Reset downtime if currently in failure state
            if self.downtime_start:
                downtime_duration = (datetime.now() - self.downtime_start).total_seconds()
                self.log_service_recovery(downtime_duration)
                self.downtime_start = None
            
            # Update failure window
            self.failure_window.append(True)
            if len(self.failure_window) > self.window_size:
                self.failure_window.pop(0)
        
        log_data = {
            "event": "REQUEST_SUCCESS",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "response_time_ms": response_time_ms,
                "response_size_bytes": response_size
            },
            "stats": {
                "total_attempts": self.total_attempts,
                "failure_rate": self._calculate_failure_rate()
            }
        }
        
        if IS_CLOUD_RUN:
            logger.info(json.dumps(log_data, separators=(',', ':')))
        else:
            logger.info(f"✅ SUCCESS: {request_id} | {response_time_ms}ms")
    
    def log_attempt_failure(self, request_id: str, failure_type: FailureType,
                           error_message: str, response_time_ms: Optional[int] = None,
                           retry_attempt: int = 0, max_retries: int = 3):
        """Log failed request attempt"""
        with self._lock:
            self.consecutive_failures += 1
            self.total_attempts += 1
            self.total_failures += 1
            
            # Mark downtime start on first failure
            if self.consecutive_failures == 1:
                self.downtime_start = datetime.now()
            
            # Update failure window
            self.failure_window.append(False)
            if len(self.failure_window) > self.window_size:
                self.failure_window.pop(0)
        
        # Structured log for failure
        failure_data = {
            "event": "REQUEST_FAILURE",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "error": {
                "type": failure_type.value,
                "message": error_message[:500],
                "retry_attempt": retry_attempt,
                "max_retries": max_retries
            },
            "metrics": {
                "response_time_ms": response_time_ms
            },
            "stats": {
                "consecutive_failures": self.consecutive_failures,
                "total_failures": self.total_failures,
                "failure_rate": self._calculate_failure_rate()
            }
        }
        
        if IS_CLOUD_RUN:
            logger.error(json.dumps(failure_data, separators=(',', ':')))
        else:
            logger.error(f"❌ FAILURE: {request_id} | {failure_type.value} | {error_message[:100]}")
        
        # Check for alerts
        self._check_and_generate_alerts(failure_type, error_message)
    
    def _calculate_failure_rate(self) -> float:
        """Calculate failure rate in the current window"""
        if not self.failure_window:
            return 0.0
        
        failures = self.failure_window.count(False)
        return failures / len(self.failure_window)
    
    def _check_and_generate_alerts(self, failure_type: FailureType, error_message: str):
        """Check thresholds and generate alerts"""
        
        # Alert for consecutive failures
        if self.consecutive_failures >= self.consecutive_failure_threshold:
            self.log_alert(
                alert_name="CONSECUTIVE_FAILURES",
                severity="CRITICAL",
                details={
                    "count": self.consecutive_failures,
                    "threshold": self.consecutive_failure_threshold,
                    "last_error": failure_type.value
                }
            )
        
        # Alert for high failure rate
        failure_rate = self._calculate_failure_rate()
        if failure_rate >= self.failure_rate_threshold and len(self.failure_window) >= 5:
            self.log_alert(
                alert_name="HIGH_FAILURE_RATE",
                severity="WARNING",
                details={
                    "rate": failure_rate,
                    "threshold": self.failure_rate_threshold
                }
            )
    
    def log_alert(self, alert_name: str, severity: str, details: Dict[str, Any]):
        """Generic alert logger"""
        alert_data = {
            "alert": f"{self.service_name.upper()}_{alert_name}",
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        if IS_CLOUD_RUN:
            if severity == "CRITICAL":
                logger.critical(json.dumps(alert_data, separators=(',', ':')))
            else:
                logger.warning(json.dumps(alert_data, separators=(',', ':')))
        else:
            level_icon = "🚨" if severity == "CRITICAL" else "⚠️"
            logger.warning(f"{level_icon} {alert_name}: {details}")

    def log_service_recovery(self, downtime_duration: float):
        """Log service recovery"""
        log_data = {
            "event": "SERVICE_RECOVERY",
            "timestamp": datetime.now().isoformat(),
            "downtime_duration_seconds": downtime_duration,
            "status": "Service is responding normally"
        }
        
        if IS_CLOUD_RUN:
            logger.info(json.dumps(log_data, separators=(',', ':')))
        else:
            logger.info(f"🔄 SERVICE RECOVERY: downtime {downtime_duration:.1f}s")

def classify_error(error_message: str) -> FailureType:
    """Classify error type based on message content"""
    error_lower = error_message.lower()
    
    if 'timed out' in error_lower or 'timeout' in error_lower:
        return FailureType.TIMEOUT
    elif 'connection' in error_lower and ('refused' in error_lower or 'failed' in error_lower):
        return FailureType.CONNECTION_ERROR
    elif 'access denied' in error_lower or 'forbidden' in error_lower or '403' in error_lower:
        return FailureType.ACCESS_DENIED
    elif 'dns resolution failed' in error_lower:
        return FailureType.DNS_ERROR
    elif 'json decode' in error_lower or 'json' in error_lower:
        return FailureType.JSON_DECODE_ERROR
    elif any(code in error_lower for code in ['404', '500', '502', '503', '504']):
        return FailureType.HTTP_ERROR
    elif 'sql' in error_lower or 'database' in error_lower or 'driver' in error_lower:
        return FailureType.DB_ERROR
    else:
        return FailureType.UNKNOWN_ERROR

