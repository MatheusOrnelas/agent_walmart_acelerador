import asyncio
import aiohttp
import json
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
import sys
from pathlib import Path

@dataclass
class LoadTestConfig:
    """Configuration for agent load testing"""
    name: str
    concurrent_users: int
    requests_per_user: int
    delay_between_requests: float
    ramp_up_time: float
    test_duration: Optional[int] = None
    target_url: str = "http://localhost:8080/chat"  # Default local URL

@dataclass
class RequestResult:
    """Result of a single request"""
    timestamp: str
    user_id: int
    request_id: int
    success: bool
    status_code: Optional[int]
    response_time: float
    error_type: Optional[str]
    error_message: Optional[str]
    response_size: Optional[int]
    query: str

@dataclass
class LoadTestResults:
    """Consolidated load test results"""
    config: LoadTestConfig
    start_time: str
    end_time: str
    total_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_patterns: Dict[str, int]
    individual_results: List[RequestResult]

class GenericLoadTester:
    """Generic Load Tester for Agent APIs"""
    
    def __init__(self, base_url: str, custom_queries: List[str] = None):
        self.base_url = base_url.rstrip('/')
        self.custom_queries = custom_queries or [
            "Hello, can you help me?",
            "What is the status of my request?",
            "Tell me about the available data."
        ]
        self.results: List[RequestResult] = []
        self.start_time = None
        self.end_time = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🌐 Load Tester configured for: {self.base_url}")

    def get_query(self, request_id: int) -> str:
        """Get a query from the pool"""
        return self.custom_queries[request_id % len(self.custom_queries)]

    async def make_single_request(self, session: aiohttp.ClientSession, user_id: int, request_id: int, query: str) -> RequestResult:
        """Execute a single request to the agent API"""
        timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        # Assuming standard /chat endpoint structure
        # Adjust payload structure based on specific Agent API implementation
        url = f"{self.base_url}/chat" if not self.base_url.endswith("/chat") else self.base_url
        
        payload = {
            "messages": [{"role": "user", "content": query}],
            "thread_id": f"load_test_{user_id}_{int(time.time())}"
        }
        
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                response_time = time.time() - start_time
                content = await response.text()
                
                if response.status == 200:
                    try:
                        # Try generic JSON check
                        await response.json()
                        return RequestResult(
                            timestamp=timestamp, user_id=user_id, request_id=request_id,
                            success=True, status_code=response.status, response_time=response_time,
                            error_type=None, error_message=None, response_size=len(content), query=query
                        )
                    except json.JSONDecodeError:
                        # Valid HTTP 200 but not JSON (might be text stream)
                        return RequestResult(
                            timestamp=timestamp, user_id=user_id, request_id=request_id,
                            success=True, status_code=response.status, response_time=response_time,
                            error_type="NON_JSON_RESPONSE", error_message=None, response_size=len(content), query=query
                        )
                else:
                    return RequestResult(
                        timestamp=timestamp, user_id=user_id, request_id=request_id,
                        success=False, status_code=response.status, response_time=response_time,
                        error_type=f"HTTP_{response.status}", error_message=f"Status {response.status}: {content[:100]}",
                        response_size=len(content), query=query
                    )
                    
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                timestamp=timestamp, user_id=user_id, request_id=request_id,
                success=False, status_code=None, response_time=response_time,
                error_type=type(e).__name__, error_message=str(e),
                response_size=0, query=query
            )

    async def simulate_user(self, user_id: int, config: LoadTestConfig, start_delay: float):
        """Simulate a single user behavior"""
        await asyncio.sleep(start_delay)
        self.logger.info(f"User {user_id} started")
        
        async with aiohttp.ClientSession() as session:
            for request_id in range(config.requests_per_user):
                query = self.get_query(request_id)
                result = await self.make_single_request(session, user_id, request_id, query)
                self.results.append(result)
                
                status = "✓" if result.success else "✗"
                self.logger.info(f"User {user_id} Req {request_id} {status} {result.response_time:.2f}s")
                
                if request_id < config.requests_per_user - 1:
                    await asyncio.sleep(config.delay_between_requests)

    async def run_load_test(self, config: LoadTestConfig) -> LoadTestResults:
        """Run the full load test"""
        self.logger.info(f"Starting Load Test: {config.name} ({config.concurrent_users} users)")
        self.start_time = datetime.now()
        self.results = []
        
        ramp_up_delay = config.ramp_up_time / config.concurrent_users if config.concurrent_users > 1 else 0
        tasks = []
        
        for user_id in range(config.concurrent_users):
            task = asyncio.create_task(
                self.simulate_user(user_id, config, user_id * ramp_up_delay)
            )
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        self.end_time = datetime.now()
        
        return self._analyze_results(config)

    def _analyze_results(self, config: LoadTestConfig) -> LoadTestResults:
        if not self.results:
            return None
            
        response_times = [r.response_time for r in self.results]
        success_count = sum(1 for r in self.results if r.success)
        total_count = len(self.results)
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        error_patterns = {}
        for r in self.results:
            if not r.success and r.error_type:
                error_patterns[r.error_type] = error_patterns.get(r.error_type, 0) + 1

        return LoadTestResults(
            config=config,
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat(),
            total_duration=total_duration,
            total_requests=total_count,
            successful_requests=success_count,
            failed_requests=total_count - success_count,
            success_rate=(success_count/total_count)*100 if total_count else 0,
            average_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else 0,
            requests_per_second=total_count/total_duration if total_duration else 0,
            error_patterns=error_patterns,
            individual_results=self.results
        )

    def save_results(self, results: LoadTestResults, output_dir: str = "load_test_results"):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        filename = f"{output_dir}/load_test_{results.config.name}_{int(time.time())}.json"
        
        data = asdict(results)
        # Serialize
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Results saved to {filename}")
        return filename

# CLI Entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generic Agent Load Tester")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--users", type=int, default=2)
    parser.add_argument("--reqs", type=int, default=5)
    args = parser.parse_args()
    
    config = LoadTestConfig(
        name="cli_test",
        concurrent_users=args.users,
        requests_per_user=args.reqs,
        delay_between_requests=0.5,
        ramp_up_time=1.0,
        target_url=args.url
    )
    
    tester = GenericLoadTester(base_url=args.url)
    asyncio.run(tester.run_load_test(config))

