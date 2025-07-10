#!/usr/bin/env python3
"""
Production Test Suite for VLLM Embedding Server
Comprehensive testing with real performance measurements
"""

import asyncio
import aiohttp
import time
import json
import statistics
import psutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class PerformanceBaseline:
    """Performance baseline for comparisons"""
    max_latency_ms: float = 1000.0  # 1 second max
    min_throughput_texts_per_sec: float = 5.0  # 5 texts/sec minimum
    max_error_rate: float = 1.0  # 1% max error rate
    max_memory_usage_percent: float = 90.0  # 90% max memory

class ProductionTestSuite:
    """Comprehensive production test suite"""
    
    def __init__(self, base_url: str = "http://localhost:8008", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.baseline = PerformanceBaseline()
        self.test_results: List[TestResult] = []
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _record_result(self, test_name: str, success: bool, duration: float, 
                      details: Dict[str, Any], error: Optional[str] = None):
        """Record test result"""
        result = TestResult(test_name, success, duration, details, error)
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.3f}s)")
        
        if error:
            print(f"   Error: {error}")
        
        # Print key metrics
        for key, value in details.items():
            if isinstance(value, (int, float)) and key.endswith(('_ms', '_sec', '_percent', '_mb')):
                print(f"   {key}: {value:.2f}")
            elif isinstance(value, bool):
                print(f"   {key}: {'✅' if value else '❌'}")
    
    def _generate_test_texts(self, count: int, length: int = 50) -> List[str]:
        """Generate test texts of various lengths"""
        texts = []
        test_phrases = [
            "What is machine learning and how does it work?",
            "Explain quantum computing principles",
            "How to implement neural networks",
            "Best practices for software engineering",
            "Introduction to artificial intelligence",
            "Data science fundamentals",
            "Cloud computing architecture",
            "Modern web development techniques",
            "Database optimization strategies",
            "Cybersecurity best practices"
        ]
        
        for i in range(count):
            if i < len(test_phrases):
                texts.append(test_phrases[i])
            else:
                # Generate random text
                words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) 
                        for _ in range(random.randint(5, length))]
                texts.append(' '.join(words))
        
        return texts
    
    async def test_01_connectivity(self) -> bool:
        """Test basic connectivity and health"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                duration = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    details = {
                        "status": data.get('status'),
                        "model": data.get('model'),
                        "response_time_ms": duration * 1000,
                        "has_gpu_info": 'gpu_memory' in data,
                        "circuit_breaker_state": data.get('circuit_breaker', 'unknown')
                    }
                    
                    # Check GPU memory if available
                    if 'gpu_memory' in data and data['gpu_memory']['available']:
                        gpu_info = data['gpu_memory']['gpus'][0]
                        details.update({
                            "gpu_memory_used_mb": gpu_info['used_mb'],
                            "gpu_memory_total_mb": gpu_info['total_mb'],
                            "gpu_utilization_percent": gpu_info['utilization']
                        })
                    
                    success = data.get('status') == 'healthy'
                    self._record_result("connectivity", success, duration, details)
                    return success
                else:
                    error_text = await response.text()
                    self._record_result("connectivity", False, duration, 
                                      {"status_code": response.status}, error_text)
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("connectivity", False, duration, {}, str(e))
            return False
    
    async def test_02_metrics_endpoint(self) -> bool:
        """Test metrics endpoint functionality"""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                duration = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    perf = data.get('performance', {})
                    memory = data.get('memory', {})
                    
                    details = {
                        "response_time_ms": duration * 1000,
                        "has_performance_data": 'performance' in data,
                        "has_memory_data": 'memory' in data,
                        "total_requests": perf.get('total_requests', 0),
                        "error_rate_percent": perf.get('error_rate', 0),
                        "uptime_seconds": perf.get('uptime_seconds', 0)
                    }
                    
                    if 'system' in memory:
                        details["system_memory_percent"] = memory['system']['utilization']
                    
                    success = True
                    self._record_result("metrics_endpoint", success, duration, details)
                    return success
                else:
                    error_text = await response.text()
                    self._record_result("metrics_endpoint", False, duration,
                                      {"status_code": response.status}, error_text)
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("metrics_endpoint", False, duration, {}, str(e))
            return False
    
    async def test_03_single_embedding(self) -> bool:
        """Test single embedding generation"""
        start_time = time.time()
        
        payload = {
            "input": ["How much protein should a female eat?"],
            "model": "intfloat/multilingual-e5-large"
        }
        
        try:
            async with self.session.post(f"{self.base_url}/v1/embeddings", json=payload) as response:
                duration = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    embedding = data['data'][0]['embedding']
                    
                    details = {
                        "response_time_ms": duration * 1000,
                        "embedding_dimension": len(embedding),
                        "embedding_type": type(embedding[0]).__name__,
                        "response_size_bytes": len(await response.read()),
                        "latency_acceptable": duration < (self.baseline.max_latency_ms / 1000)
                    }
                    
                    # Validate embedding quality
                    embedding_magnitude = sum(x*x for x in embedding) ** 0.5
                    details["embedding_magnitude"] = embedding_magnitude
                    details["embedding_normalized"] = 0.9 < embedding_magnitude < 1.1
                    
                    success = len(embedding) == 1024 and details["latency_acceptable"]
                    self._record_result("single_embedding", success, duration, details)
                    return success
                else:
                    error_text = await response.text()
                    self._record_result("single_embedding", False, duration,
                                      {"status_code": response.status}, error_text)
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("single_embedding", False, duration, {}, str(e))
            return False
    
    async def test_04_batch_embedding(self) -> bool:
        """Test batch embedding generation"""
        test_texts = self._generate_test_texts(10)
        start_time = time.time()
        
        try:
            async with self.session.post(f"{self.base_url}/embeddings/batch", json=test_texts) as response:
                duration = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    embeddings = [item['embedding'] for item in data['data']]
                    
                    details = {
                        "response_time_ms": duration * 1000,
                        "input_texts": len(test_texts),
                        "output_embeddings": len(embeddings),
                        "throughput_texts_per_sec": len(test_texts) / duration,
                        "avg_embedding_dimension": statistics.mean(len(emb) for emb in embeddings),
                        "throughput_acceptable": (len(test_texts) / duration) >= self.baseline.min_throughput_texts_per_sec
                    }
                    
                    success = (len(embeddings) == len(test_texts) and 
                             details["throughput_acceptable"])
                    self._record_result("batch_embedding", success, duration, details)
                    return success
                else:
                    error_text = await response.text()
                    self._record_result("batch_embedding", False, duration,
                                      {"status_code": response.status}, error_text)
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("batch_embedding", False, duration, {}, str(e))
            return False
    
    async def test_05_concurrent_load(self, num_concurrent: int = 10) -> bool:
        """Test concurrent request handling"""
        start_time = time.time()
        
        async def single_request(request_id: int) -> Dict[str, Any]:
            """Single concurrent request"""
            payload = {
                "input": [f"Concurrent test request {request_id}: machine learning basics"],
                "model": "intfloat/multilingual-e5-large"
            }
            
            req_start = time.time()
            try:
                async with self.session.post(f"{self.base_url}/v1/embeddings", json=payload) as response:
                    req_duration = time.time() - req_start
                    
                    return {
                        "request_id": request_id,
                        "success": response.status == 200,
                        "duration": req_duration,
                        "status_code": response.status
                    }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "duration": time.time() - req_start,
                    "error": str(e)
                }
        
        try:
            # Execute concurrent requests
            tasks = [single_request(i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            
            # Analyze results
            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]
            
            latencies = [r['duration'] for r in successful]
            
            details = {
                "total_requests": num_concurrent,
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "success_rate_percent": (len(successful) / num_concurrent) * 100,
                "total_duration_sec": duration,
                "throughput_requests_per_sec": num_concurrent / duration,
                "error_rate_acceptable": (len(failed) / num_concurrent * 100) <= self.baseline.max_error_rate
            }
            
            if latencies:
                details.update({
                    "avg_latency_ms": statistics.mean(latencies) * 1000,
                    "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
                    "max_latency_ms": max(latencies) * 1000,
                    "min_latency_ms": min(latencies) * 1000
                })
            
            success = (details["error_rate_acceptable"] and 
                      details["success_rate_percent"] >= 95.0)
            
            self._record_result("concurrent_load", success, duration, details)
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("concurrent_load", False, duration, {}, str(e))
            return False
    
    async def test_06_large_batch_processing(self) -> bool:
        """Test large batch processing"""
        large_texts = self._generate_test_texts(50, length=100)  # 50 texts
        start_time = time.time()
        
        try:
            async with self.session.post(f"{self.base_url}/embeddings/batch", json=large_texts) as response:
                duration = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    embeddings = [item['embedding'] for item in data['data']]
                    
                    details = {
                        "input_texts": len(large_texts),
                        "output_embeddings": len(embeddings),
                        "response_time_ms": duration * 1000,
                        "throughput_texts_per_sec": len(large_texts) / duration,
                        "avg_text_length": statistics.mean(len(text.split()) for text in large_texts),
                        "processing_acceptable": duration < 10.0  # 10 seconds max for large batch
                    }
                    
                    success = (len(embeddings) == len(large_texts) and 
                             details["processing_acceptable"])
                    
                    self._record_result("large_batch_processing", success, duration, details)
                    return success
                else:
                    error_text = await response.text()
                    self._record_result("large_batch_processing", False, duration,
                                      {"status_code": response.status}, error_text)
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("large_batch_processing", False, duration, {}, str(e))
            return False
    
    async def test_07_memory_stress(self) -> bool:
        """Test memory usage under stress"""
        start_time = time.time()
        
        # Get initial memory state
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    initial_data = await response.json()
                    initial_memory = initial_data.get('memory', {}).get('system', {}).get('utilization', 0)
                else:
                    initial_memory = 0
        except:
            initial_memory = 0
        
        # Generate memory-intensive requests
        large_texts = []
        for i in range(20):  # 20 large texts
            text = ' '.join(['memory test text'] * 50)  # Long repeating text
            large_texts.append(text)
        
        try:
            # Process large batch
            async with self.session.post(f"{self.base_url}/embeddings/batch", json=large_texts) as response:
                process_duration = time.time() - start_time
                
                # Get final memory state
                async with self.session.get(f"{self.base_url}/metrics") as mem_response:
                    if mem_response.status == 200:
                        final_data = await mem_response.json()
                        final_memory = final_data.get('memory', {}).get('system', {}).get('utilization', 0)
                        gpu_memory = final_data.get('memory', {}).get('gpu', {})
                    else:
                        final_memory = 0
                        gpu_memory = {}
                
                duration = time.time() - start_time
                
                details = {
                    "initial_memory_percent": initial_memory,
                    "final_memory_percent": final_memory,
                    "memory_increase_percent": final_memory - initial_memory,
                    "processing_time_sec": process_duration,
                    "request_successful": response.status == 200,
                    "memory_acceptable": final_memory <= self.baseline.max_memory_usage_percent
                }
                
                # Add GPU memory info if available
                if gpu_memory.get('available'):
                    for i, gpu in enumerate(gpu_memory.get('gpus', [])):
                        details[f"gpu_{i}_memory_percent"] = gpu.get('utilization', 0)
                
                success = (details["memory_acceptable"] and 
                          details["request_successful"])
                
                self._record_result("memory_stress", success, duration, details)
                return success
                
        except Exception as e:
            duration = time.time() - start_time
            self._record_result("memory_stress", False, duration, {}, str(e))
            return False
    
    async def test_08_error_handling(self) -> bool:
        """Test error handling and recovery"""
        start_time = time.time()
        
        test_cases = [
            # Invalid request formats
            {"test": "empty_input", "payload": {"input": [], "model": "test"}},
            {"test": "oversized_batch", "payload": {"input": ["test"] * 1000, "model": "test"}},
            {"test": "invalid_model", "payload": {"input": ["test"], "model": "nonexistent"}},
            {"test": "malformed_json", "payload": "invalid json"},
        ]
        
        results = {}
        
        for test_case in test_cases:
            test_name = test_case["test"]
            payload = test_case["payload"]
            
            try:
                if isinstance(payload, str):
                    # Malformed JSON test
                    async with self.session.post(f"{self.base_url}/v1/embeddings", 
                                               data=payload,
                                               headers={'Content-Type': 'application/json'}) as response:
                        results[test_name] = {
                            "status_code": response.status,
                            "handled_gracefully": 400 <= response.status < 500
                        }
                else:
                    async with self.session.post(f"{self.base_url}/v1/embeddings", json=payload) as response:
                        results[test_name] = {
                            "status_code": response.status,
                            "handled_gracefully": 400 <= response.status < 500
                        }
                        
            except Exception as e:
                results[test_name] = {
                    "status_code": 0,
                    "handled_gracefully": False,
                    "error": str(e)
                }
        
        duration = time.time() - start_time
        
        # Analyze error handling
        graceful_handling = sum(1 for r in results.values() if r.get("handled_gracefully", False))
        total_tests = len(test_cases)
        
        details = {
            "total_error_tests": total_tests,
            "gracefully_handled": graceful_handling,
            "error_handling_rate_percent": (graceful_handling / total_tests) * 100,
            "test_results": results
        }
        
        success = graceful_handling >= (total_tests * 0.75)  # 75% should be handled gracefully
        
        self._record_result("error_handling", success, duration, details)
        return success
    
    async def test_09_sustained_load(self, duration_seconds: int = 30) -> bool:
        """Test sustained load over time"""
        start_time = time.time()
        
        completed_requests = 0
        failed_requests = 0
        latencies = []
        
        async def continuous_requests():
            nonlocal completed_requests, failed_requests, latencies
            
            request_id = 0
            while time.time() - start_time < duration_seconds:
                try:
                    payload = {
                        "input": [f"Sustained load test {request_id}: artificial intelligence"],
                        "model": "intfloat/multilingual-e5-large"
                    }
                    
                    req_start = time.time()
                    async with self.session.post(f"{self.base_url}/v1/embeddings", json=payload) as response:
                        req_duration = time.time() - req_start
                        
                        if response.status == 200:
                            completed_requests += 1
                            latencies.append(req_duration)
                        else:
                            failed_requests += 1
                    
                    request_id += 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.05)  # 50ms delay
                    
                except Exception as e:
                    failed_requests += 1
                    await asyncio.sleep(0.1)
        
        # Run sustained load
        await continuous_requests()
        
        total_duration = time.time() - start_time
        total_requests = completed_requests + failed_requests
        
        details = {
            "test_duration_sec": total_duration,
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "failed_requests": failed_requests,
            "success_rate_percent": (completed_requests / max(total_requests, 1)) * 100,
            "avg_throughput_requests_per_sec": total_requests / total_duration,
            "completed_throughput_requests_per_sec": completed_requests / total_duration
        }
        
        if latencies:
            details.update({
                "avg_latency_ms": statistics.mean(latencies) * 1000,
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
                "max_latency_ms": max(latencies) * 1000
            })
        
        success = (details["success_rate_percent"] >= 95.0 and 
                  details["completed_throughput_requests_per_sec"] >= 1.0)
        
        self._record_result("sustained_load", success, total_duration, details)
        return success
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        
        # Performance summary
        latency_results = []
        throughput_results = []
        
        for result in self.test_results:
            if 'response_time_ms' in result.details:
                latency_results.append(result.details['response_time_ms'])
            if 'throughput_texts_per_sec' in result.details:
                throughput_results.append(result.details['throughput_texts_per_sec'])
            if 'throughput_requests_per_sec' in result.details:
                throughput_results.append(result.details['throughput_requests_per_sec'])
        
        performance_summary = {}
        if latency_results:
            performance_summary["latency"] = {
                "avg_ms": statistics.mean(latency_results),
                "p95_ms": sorted(latency_results)[int(len(latency_results) * 0.95)],
                "max_ms": max(latency_results),
                "min_ms": min(latency_results)
            }
        
        if throughput_results:
            performance_summary["throughput"] = {
                "avg_per_sec": statistics.mean(throughput_results),
                "max_per_sec": max(throughput_results),
                "min_per_sec": min(throughput_results)
            }
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
                "overall_status": "PASS" if passed_tests == total_tests else "FAIL"
            },
            "performance": performance_summary,
            "baseline_comparison": {
                "max_latency_requirement_ms": self.baseline.max_latency_ms,
                "min_throughput_requirement": self.baseline.min_throughput_texts_per_sec,
                "max_error_rate_requirement": self.baseline.max_error_rate,
                "max_memory_requirement": self.baseline.max_memory_usage_percent
            },
            "detailed_results": [
                {
                    "test": r.test_name,
                    "status": "PASS" if r.success else "FAIL",
                    "duration": r.duration,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.test_results
            ]
        }
    
    async def run_full_suite(self) -> Dict[str, Any]:
        """Run complete production test suite"""
        print("🏭 Production Test Suite for VLLM Embedding Server")
        print("=" * 60)
        
        # Test sequence
        test_functions = [
            ("Basic Connectivity", self.test_01_connectivity),
            ("Metrics Endpoint", self.test_02_metrics_endpoint),
            ("Single Embedding", self.test_03_single_embedding),
            ("Batch Embedding", self.test_04_batch_embedding),
            ("Concurrent Load", lambda: self.test_05_concurrent_load(10)),
            ("Large Batch Processing", self.test_06_large_batch_processing),
            ("Memory Stress Test", self.test_07_memory_stress),
            ("Error Handling", self.test_08_error_handling),
            ("Sustained Load", lambda: self.test_09_sustained_load(30))
        ]
        
        print(f"\nRunning {len(test_functions)} production tests...\n")
        
        for i, (test_name, test_func) in enumerate(test_functions, 1):
            print(f"{i:2d}. {test_name}")
            try:
                await test_func()
            except Exception as e:
                self._record_result(test_name.lower().replace(' ', '_'), False, 0, {}, str(e))
            print()
        
        # Generate and display report
        report = self.generate_report()
        
        print("=" * 60)
        print("📊 PRODUCTION TEST RESULTS")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"Overall Status: {'🟢 PASS' if summary['overall_status'] == 'PASS' else '🔴 FAIL'}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']} ({summary['success_rate']:.1f}%)")
        
        if "performance" in report and report["performance"]:
            perf = report["performance"]
            print("\n📈 Performance Summary:")
            if "latency" in perf:
                lat = perf["latency"]
                print(f"   Average Latency: {lat['avg_ms']:.1f}ms")
                print(f"   P95 Latency: {lat['p95_ms']:.1f}ms")
                print(f"   Max Latency: {lat['max_ms']:.1f}ms")
            
            if "throughput" in perf:
                thr = perf["throughput"]
                print(f"   Average Throughput: {thr['avg_per_sec']:.1f}/sec")
                print(f"   Peak Throughput: {thr['max_per_sec']:.1f}/sec")
        
        print("\n🎯 Baseline Comparison:")
        baseline = report["baseline_comparison"]
        print(f"   Max Latency Requirement: {baseline['max_latency_requirement_ms']}ms")
        print(f"   Min Throughput Requirement: {baseline['min_throughput_requirement']}/sec")
        print(f"   Max Error Rate: {baseline['max_error_rate_requirement']}%")
        
        print("\n📋 Test Details:")
        for result in report["detailed_results"]:
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            print(f"   {status_icon} {result['test']}: {result['duration']:.3f}s")
            if result["error"]:
                print(f"      Error: {result['error']}")
        
        return report

async def main():
    """Main test execution"""
    print("Starting Production Test Suite...")
    
    async with ProductionTestSuite() as test_suite:
        report = await test_suite.run_full_suite()
        
        # Save report to file
        import json
        with open('production_test_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: production_test_report.json")
        
        return report["summary"]["overall_status"] == "PASS"

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        exit(1)