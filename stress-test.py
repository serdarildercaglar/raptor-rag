#!/usr/bin/env python3
"""
Production Stress Test for RAG System
Tests batch processing, async operations, accuracy, and performance under load
Using Zulficore-specific queries for accuracy validation
"""

import asyncio
import aiohttp
import time
import json
import statistics
import psutil
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

@dataclass
class StressTestResult:
    """Stress test result data structure"""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for stress testing"""
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    throughput_qps: float
    success_rate: float
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float

class ZulficorecQueryBank:
    """
    Zulficore-specific queries for accuracy testing
    """
    
    # Core Zulficore queries with expected keywords for accuracy validation
    ZULFICORE_QUERIES = [
        {
            "query": "Zulficore protokolünün temel amacı nedir?",
            "expected_keywords": ["zulficore", "protokol", "sistem", "amaç"],
            "category": "protocol_basics"
        },
        {
            "query": "bilinç rezonansı terimi Zulficore yorumunda ne anlama geliyor?",
            "expected_keywords": ["bilinç", "rezonans", "frekans", "titreşim"],
            "category": "consciousness"
        },
        {
            "query": "peygamberlere yapılan ilahi seslenişler Zulficore tarafından nasıl yorumlanıyor?",
            "expected_keywords": ["peygamber", "ilahi", "seslenish", "musa"],
            "category": "prophets"
        },
        {
            "query": "Zulficore peygamberleri yankı taşıyıcıları olarak nasıl tanımlıyor?",
            "expected_keywords": ["yankı", "taşıyıcı", "peygamber", "frekans"],
            "category": "echo_carriers"
        },
        {
            "query": "kozmik komut frekansı Zulficore sisteminde neye karşılık geliyor?",
            "expected_keywords": ["kozmik", "komut", "frekans", "sistem"],
            "category": "cosmic_frequency"
        },
        {
            "query": "Hz Nuh'un gemisi Zulficore yorumunda neden frekans kapsülü olarak nitelendiriliyor?",
            "expected_keywords": ["nuh", "gemi", "frekans", "kapsül"],
            "category": "noah_ark"
        },
        {
            "query": "Hz İsa'nın mucizeleri Zulficore tarafından nasıl açıklanıyor?",
            "expected_keywords": ["isa", "mucize", "yankı", "frekans"],
            "category": "jesus_miracles"
        },
        {
            "query": "Oku emri Zulficore tefsirinde hangi derin anlamı taşıyor?",
            "expected_keywords": ["oku", "emir", "frekans", "titreşim"],
            "category": "read_command"
        },
        {
            "query": "Hz Adem'in Cennet'ten inişi rezonans sapmasından geri dönüş sistemi olarak neden yorumlanıyor?",
            "expected_keywords": ["adem", "cennet", "rezonans", "sapma"],
            "category": "adam_descent"
        },
        {
            "query": "Zulficore tefsirine göre zorlukla birlikte kolaylık ilkesi frekans düzeyinde nasıl karşılık buluyor?",
            "expected_keywords": ["zorluk", "kolaylık", "frekans", "ilke"],
            "category": "difficulty_ease"
        },
        {
            "query": "Hz Yunus'un balığın karnından kurtulması Zulficore tarafından nasıl yankı aktivasyonu olarak görülüyor?",
            "expected_keywords": ["yunus", "balık", "yankı", "aktivasyon"],
            "category": "jonah_rescue"
        },
        {
            "query": "yankının inşası başlıklı Zulficore Tefsiri Cilt 1'in temel odak noktası nedir?",
            "expected_keywords": ["yankı", "inşa", "tefsir", "cilt"],
            "category": "echo_construction"
        },
        {
            "query": "Zulficore'a göre rahmet kelimesi Hz Muhammed'e verilen yankı frekansıyla nasıl ilişkilendiriliyor?",
            "expected_keywords": ["rahmet", "muhammed", "yankı", "frekans"],
            "category": "mercy_frequency"
        },
        {
            "query": "Bakara Suresi 2'de geçen rehber kelimesi Zulficore yorumunda neden yönlendirici titreşim aklı olarak açıklanıyor?",
            "expected_keywords": ["bakara", "rehber", "titreşim", "akıl"],
            "category": "guidance_vibration"
        },
        {
            "query": "yankı ağacı metaforu İbrahim Suresi 24 Zulficore tarafından nasıl anlam kazanıyor?",
            "expected_keywords": ["yankı", "ağaç", "ibrahim", "metafor"],
            "category": "echo_tree"
        },
        {
            "query": "Zulficore kitabının temel mesajı ve hedef kitlesi kimlerdir?",
            "expected_keywords": ["kitap", "mesaj", "hedef", "kitle"],
            "category": "book_message"
        },
        {
            "query": "Zulficore sisteminin vaat ettiği %97 başarı oranı neyi ifade etmektedir?",
            "expected_keywords": ["sistem", "97", "başarı", "oran"],
            "category": "success_rate"
        },
        {
            "query": "Zulficore iyilik iyidir prensibini nasıl bir iş modeline dönüştürmeyi hedefliyor?",
            "expected_keywords": ["iyilik", "iyidir", "iş", "model"],
            "category": "business_model"
        },
        {
            "query": "Zulficore Kampüs projesi startup fikirleri ve mühendislik alanlarıyla ilgili hangi yenilikçi yaklaşımları içeriyor?",
            "expected_keywords": ["kampüs", "startup", "mühendislik", "yenilik"],
            "category": "campus_innovation"
        },
        {
            "query": "Zulficore sisteminin gelecekteki hedefleri arasında insanın ve zamanın dönüştürülmesi ne anlama geliyor?",
            "expected_keywords": ["hedef", "insan", "zaman", "dönüştürme"],
            "category": "future_transformation"
        }
    ]
    
    # Additional load testing queries
    LOAD_TEST_QUERIES = [
        "quantum simulation nedir ve nasıl çalışır?",
        "yapay zeka ve bilinç arasındaki ilişki nedir?",
        "frekans tabanlı öğrenme nasıl gerçekleşir?",
        "digital transformation stratejileri nelerdir?",
        "sustainable energy solutions hangileridir?",
        "blockchain technology applications nelerdir?",
        "machine learning algoritmaları hangileridir?",
        "deep learning neural networks nasıl çalışır?",
        "natural language processing techniques nelerdir?",
        "computer vision applications hangileridir?"
    ]
    
    @classmethod
    def get_random_query(cls) -> Dict[str, Any]:
        """Get random Zulficore query with metadata"""
        return random.choice(cls.ZULFICORE_QUERIES)
    
    @classmethod
    def get_load_test_queries(cls, count: int) -> List[str]:
        """Get load test queries"""
        return random.choices(cls.LOAD_TEST_QUERIES, k=count)
    
    @classmethod
    def get_all_zulficore_queries(cls) -> List[Dict[str, Any]]:
        """Get all Zulficore queries"""
        return cls.ZULFICORE_QUERIES.copy()

class ProductionStressTester:
    """
    Production-grade stress tester for RAG system
    """
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.results: List[StressTestResult] = []
        self.session_pool_size = 50  # HTTP session pool
        
    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str, payload: dict) -> tuple:
        """Make HTTP request with timing"""
        start_time = time.perf_counter()
        try:
            async with session.post(f"{self.base_url}{endpoint}", json=payload) as response:
                duration = time.perf_counter() - start_time
                if response.status == 200:
                    result = await response.json()
                    return True, duration, result, response.status
                else:
                    error_text = await response.text()
                    return False, duration, error_text, response.status
        except Exception as e:
            duration = time.perf_counter() - start_time
            return False, duration, str(e), 0
    
    def _calculate_metrics(self, durations: List[float], successes: List[bool]) -> PerformanceMetrics:
        """Calculate performance metrics"""
        durations_ms = [d * 1000 for d in durations]
        success_count = sum(successes)
        total_count = len(successes)
        
        # Get system metrics
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        return PerformanceMetrics(
            avg_latency_ms=statistics.mean(durations_ms) if durations_ms else 0,
            p95_latency_ms=np.percentile(durations_ms, 95) if durations_ms else 0,
            p99_latency_ms=np.percentile(durations_ms, 99) if durations_ms else 0,
            max_latency_ms=max(durations_ms) if durations_ms else 0,
            min_latency_ms=min(durations_ms) if durations_ms else 0,
            throughput_qps=success_count / sum(durations) if durations else 0,
            success_rate=(success_count / total_count * 100) if total_count > 0 else 0,
            error_rate=((total_count - success_count) / total_count * 100) if total_count > 0 else 0,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent
        )
    
    def _validate_accuracy(self, query_data: Dict[str, Any], response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate response accuracy against expected keywords"""
        if 'results' not in response_data:
            return {"accuracy_score": 0, "found_keywords": [], "missing_keywords": query_data['expected_keywords']}
        
        # Combine all result texts
        all_text = " ".join([result.get('text', '').lower() for result in response_data['results']])
        
        # Check keyword presence
        found_keywords = []
        missing_keywords = []
        
        for keyword in query_data['expected_keywords']:
            if keyword.lower() in all_text:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        accuracy_score = (len(found_keywords) / len(query_data['expected_keywords'])) * 100
        
        return {
            "accuracy_score": accuracy_score,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "total_results": len(response_data['results']),
            "category": query_data['category']
        }
    
    async def test_01_service_health(self) -> bool:
        """Test service health and readiness"""
        print("1️⃣ Testing Service Health...")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            
            try:
                success, duration, result, status = await self._make_request(
                    session, "/health", {}
                )
                
                if success and isinstance(result, dict):
                    details = {
                        "response_time_ms": duration * 1000,
                        "node_count": result.get('node_count', 0),
                        "service": result.get('service', 'unknown'),
                        "pattern": result.get('pattern', 'unknown')
                    }
                    
                    self._record_result("service_health", True, duration, details)
                    print(f"✅ Service healthy: {result.get('node_count', 0)} nodes")
                    return True
                else:
                    self._record_result("service_health", False, duration, 
                                      {"status": status, "error": str(result)})
                    print(f"❌ Service unhealthy: {status}")
                    return False
                    
            except Exception as e:
                duration = time.perf_counter() - start_time
                self._record_result("service_health", False, duration, {}, str(e))
                print(f"❌ Health check failed: {e}")
                return False
    
    async def test_02_single_query_performance(self) -> bool:
        """Test single query performance"""
        print("\n2️⃣ Testing Single Query Performance...")
        
        async with aiohttp.ClientSession() as session:
            durations = []
            successes = []
            
            # Test with different Zulficore queries
            test_queries = ZulficorecQueryBank.get_all_zulficore_queries()[:5]  # First 5 queries
            
            for i, query_data in enumerate(test_queries):
                payload = {
                    "query": f"query: {query_data['query']}",  # ADD QUERY PREFIX!
                    "top_k": 5,
                    "similarity_cutoff": 0.0
                }
                
                success, duration, result, status = await self._make_request(
                    session, "/retrieve", payload
                )
                
                durations.append(duration)
                successes.append(success)
                
                if success:
                    print(f"   Query {i+1}: {duration*1000:.1f}ms - {len(result.get('results', []))} results")
                else:
                    print(f"   Query {i+1}: FAILED - {status}")
            
            metrics = self._calculate_metrics(durations, successes)
            
            details = {
                "total_queries": len(test_queries),
                "avg_latency_ms": metrics.avg_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "success_rate": metrics.success_rate,
                "throughput_qps": metrics.throughput_qps
            }
            
            success = metrics.success_rate >= 95.0 and metrics.avg_latency_ms <= 200
            self._record_result("single_query_performance", success, sum(durations), details)
            
            print(f"✅ Avg latency: {metrics.avg_latency_ms:.1f}ms")
            print(f"✅ P95 latency: {metrics.p95_latency_ms:.1f}ms")
            print(f"✅ Success rate: {metrics.success_rate:.1f}%")
            
            return success
    
    async def test_03_batch_processing_stress(self) -> bool:
        """Test batch processing with varying load"""
        print("\n3️⃣ Testing Batch Processing Stress...")
        
        batch_sizes = [5, 10, 20, 50, 100]  # Increasing batch sizes
        all_results = []
        
        async with aiohttp.ClientSession() as session:
            for batch_size in batch_sizes:
                print(f"   Testing batch size: {batch_size}")
                
                # Mix of Zulficore and load test queries
                zulficore_queries = [q['query'] for q in ZulficorecQueryBank.get_all_zulficore_queries()]
                load_queries = ZulficorecQueryBank.get_load_test_queries(batch_size)
                
                # Mix queries (50% Zulficore, 50% load test)
                mixed_queries = []
                for i in range(batch_size):
                    if i < len(zulficore_queries) and i % 2 == 0:
                        mixed_queries.append(f"query: {zulficore_queries[i % len(zulficore_queries)]}")
                    else:
                        mixed_queries.append(f"query: {load_queries[i % len(load_queries)]}")
                
                payload = {
                    "queries": mixed_queries,
                    "top_k": 3,
                    "similarity_cutoff": 0.0
                }
                
                start_time = time.perf_counter()
                success, duration, result, status = await self._make_request(
                    session, "/retrieve/batch", payload
                )
                
                if success:
                    throughput = batch_size / duration
                    print(f"      ✅ {duration*1000:.1f}ms ({throughput:.1f} q/s)")
                    
                    batch_result = {
                        "batch_size": batch_size,
                        "duration_ms": duration * 1000,
                        "throughput_qps": throughput,
                        "success": True,
                        "total_results": sum(len(query_results) for query_results in result.get('results', []))
                    }
                else:
                    print(f"      ❌ FAILED - {status}")
                    batch_result = {
                        "batch_size": batch_size,
                        "duration_ms": duration * 1000,
                        "success": False,
                        "error": str(result)
                    }
                
                all_results.append(batch_result)
                
                # Brief pause between tests
                await asyncio.sleep(1)
        
        # Analyze batch performance
        successful_batches = [r for r in all_results if r['success']]
        avg_throughput = statistics.mean([r['throughput_qps'] for r in successful_batches]) if successful_batches else 0
        
        details = {
            "batch_tests": len(batch_sizes),
            "successful_batches": len(successful_batches),
            "avg_throughput_qps": avg_throughput,
            "batch_results": all_results
        }
        
        success = len(successful_batches) >= len(batch_sizes) * 0.8  # 80% success rate
        self._record_result("batch_processing_stress", success, 0, details)
        
        print(f"✅ Successful batches: {len(successful_batches)}/{len(batch_sizes)}")
        print(f"✅ Average throughput: {avg_throughput:.1f} q/s")
        
        return success
    
    async def test_04_concurrent_async_load(self) -> bool:
        """Test concurrent async request handling"""
        print("\n4️⃣ Testing Concurrent Async Load...")
        
        concurrent_levels = [10, 25, 50, 100]  # Concurrent request levels
        
        async def single_concurrent_request(session: aiohttp.ClientSession, request_id: int) -> tuple:
            """Single concurrent request"""
            query_data = ZulficorecQueryBank.get_random_query()
            payload = {
                "query": f"query: {query_data['query']}",  # ADD QUERY PREFIX!
                "top_k": 5
            }
            
            return await self._make_request(session, "/retrieve", payload)
        
        all_concurrent_results = []
        
        for concurrent_count in concurrent_levels:
            print(f"   Testing {concurrent_count} concurrent requests...")
            
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=concurrent_count + 10)
            ) as session:
                
                # Create concurrent tasks
                start_time = time.perf_counter()
                tasks = [
                    single_concurrent_request(session, i) 
                    for i in range(concurrent_count)
                ]
                
                # Execute all concurrent requests
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_duration = time.perf_counter() - start_time
                
                # Analyze results
                successes = []
                durations = []
                
                for result in results:
                    if isinstance(result, Exception):
                        successes.append(False)
                        durations.append(0)
                    else:
                        success, duration, _, _ = result
                        successes.append(success)
                        durations.append(duration)
                
                metrics = self._calculate_metrics(durations, successes)
                
                concurrent_result = {
                    "concurrent_requests": concurrent_count,
                    "total_duration_sec": total_duration,
                    "success_rate": metrics.success_rate,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "p95_latency_ms": metrics.p95_latency_ms,
                    "throughput_qps": concurrent_count / total_duration,
                    "memory_usage_mb": metrics.memory_usage_mb,
                    "cpu_usage_percent": metrics.cpu_usage_percent
                }
                
                all_concurrent_results.append(concurrent_result)
                
                print(f"      ✅ Success: {metrics.success_rate:.1f}% | "
                      f"Avg: {metrics.avg_latency_ms:.1f}ms | "
                      f"Throughput: {concurrent_count / total_duration:.1f} q/s")
                
                # Brief pause between tests
                await asyncio.sleep(2)
        
        # Overall concurrent performance analysis
        avg_success_rate = statistics.mean([r['success_rate'] for r in all_concurrent_results])
        max_throughput = max([r['throughput_qps'] for r in all_concurrent_results])
        
        details = {
            "concurrent_tests": len(concurrent_levels),
            "avg_success_rate": avg_success_rate,
            "max_throughput_qps": max_throughput,
            "concurrent_results": all_concurrent_results
        }
        
        success = avg_success_rate >= 95.0
        self._record_result("concurrent_async_load", success, 0, details)
        
        print(f"✅ Average success rate: {avg_success_rate:.1f}%")
        print(f"✅ Max throughput: {max_throughput:.1f} q/s")
        
        return success
    
    async def test_05_accuracy_validation(self) -> bool:
        """Test accuracy with Zulficore-specific queries"""
        print("\n5️⃣ Testing Accuracy Validation...")
        
        async with aiohttp.ClientSession() as session:
            accuracy_results = []
            
            # Test all Zulficore queries for accuracy
            zulficore_queries = ZulficorecQueryBank.get_all_zulficore_queries()
            
            for i, query_data in enumerate(zulficore_queries):
                payload = {
                    "query": f"query: {query_data['query']}",  # ADD QUERY PREFIX!
                    "top_k": 5,
                    "similarity_cutoff": 0.0
                }
                
                success, duration, result, status = await self._make_request(
                    session, "/retrieve", payload
                )
                
                if success:
                    accuracy_info = self._validate_accuracy(query_data, result)
                    accuracy_results.append(accuracy_info)
                    
                    print(f"   Query {i+1} ({query_data['category']}): "
                          f"{accuracy_info['accuracy_score']:.1f}% accuracy")
                    
                    if accuracy_info['accuracy_score'] < 50:
                        print(f"      ⚠️  Low accuracy! Missing: {accuracy_info['missing_keywords']}")
                else:
                    print(f"   Query {i+1}: FAILED - {status}")
                    accuracy_results.append({
                        "accuracy_score": 0,
                        "category": query_data['category'],
                        "error": str(result)
                    })
            
            # Calculate overall accuracy metrics
            valid_results = [r for r in accuracy_results if 'error' not in r]
            avg_accuracy = statistics.mean([r['accuracy_score'] for r in valid_results]) if valid_results else 0
            
            # Category-wise accuracy
            category_accuracy = {}
            for result in valid_results:
                category = result['category']
                if category not in category_accuracy:
                    category_accuracy[category] = []
                category_accuracy[category].append(result['accuracy_score'])
            
            category_avg = {
                cat: statistics.mean(scores) 
                for cat, scores in category_accuracy.items()
            }
            
            details = {
                "total_queries": len(zulficore_queries),
                "successful_queries": len(valid_results),
                "avg_accuracy": avg_accuracy,
                "category_accuracy": category_avg,
                "accuracy_distribution": {
                    "excellent (>80%)": len([r for r in valid_results if r['accuracy_score'] > 80]),
                    "good (60-80%)": len([r for r in valid_results if 60 <= r['accuracy_score'] <= 80]),
                    "fair (40-60%)": len([r for r in valid_results if 40 <= r['accuracy_score'] < 60]),
                    "poor (<40%)": len([r for r in valid_results if r['accuracy_score'] < 40])
                }
            }
            
            success = avg_accuracy >= 70.0  # 70% average accuracy threshold
            self._record_result("accuracy_validation", success, 0, details)
            
            print(f"✅ Average accuracy: {avg_accuracy:.1f}%")
            print(f"✅ Category performance:")
            for category, accuracy in category_avg.items():
                print(f"      {category}: {accuracy:.1f}%")
            
            return success
    
    async def test_06_memory_and_cpu_stress(self) -> bool:
        """Test memory and CPU usage under sustained load"""
        print("\n6️⃣ Testing Memory and CPU Stress...")
        
        duration_seconds = 60  # 1 minute sustained load
        request_rate = 5  # requests per second
        
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            memory_samples = []
            cpu_samples = []
            requests_completed = 0
            requests_failed = 0
            
            while time.perf_counter() - start_time < duration_seconds:
                # Sample system metrics
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                memory_samples.append(memory_mb)
                cpu_samples.append(cpu_percent)
                
                # Make requests at specified rate
                tasks = []
                for _ in range(request_rate):
                    query_data = ZulficorecQueryBank.get_random_query()
                    payload = {
                        "query": f"query: {query_data['query']}",  # ADD QUERY PREFIX!
                        "top_k": 3
                    }
                    
                    task = self._make_request(session, "/retrieve", payload)
                    tasks.append(task)
                
                # Execute batch of requests
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception) or not result[0]:
                        requests_failed += 1
                    else:
                        requests_completed += 1
                
                # Wait to maintain request rate
                await asyncio.sleep(1.0)  # 1 second intervals
            
            total_duration = time.perf_counter() - start_time
            total_requests = requests_completed + requests_failed
            
            details = {
                "test_duration_sec": total_duration,
                "total_requests": total_requests,
                "completed_requests": requests_completed,
                "failed_requests": requests_failed,
                "success_rate": (requests_completed / total_requests * 100) if total_requests > 0 else 0,
                "avg_memory_mb": statistics.mean(memory_samples) if memory_samples else 0,
                "max_memory_mb": max(memory_samples) if memory_samples else 0,
                "avg_cpu_percent": statistics.mean(cpu_samples) if cpu_samples else 0,
                "max_cpu_percent": max(cpu_samples) if cpu_samples else 0,
                "memory_growth_mb": (memory_samples[-1] - memory_samples[0]) if len(memory_samples) > 1 else 0,
                "throughput_qps": requests_completed / total_duration
            }
            
            # Success criteria: <90% CPU, <2GB memory, >95% success rate
            success = (details["max_cpu_percent"] < 90 and 
                      details["max_memory_mb"] < 2048 and 
                      details["success_rate"] >= 95)
            
            self._record_result("memory_cpu_stress", success, total_duration, details)
            
            print(f"✅ Success rate: {details['success_rate']:.1f}%")
            print(f"✅ Avg memory: {details['avg_memory_mb']:.1f}MB")
            print(f"✅ Max memory: {details['max_memory_mb']:.1f}MB")
            print(f"✅ Avg CPU: {details['avg_cpu_percent']:.1f}%")
            print(f"✅ Memory growth: {details['memory_growth_mb']:.1f}MB")
            
            return success
    
    async def test_07_error_recovery(self) -> bool:
        """Test error handling and recovery"""
        print("\n7️⃣ Testing Error Recovery...")
        
        async with aiohttp.ClientSession() as session:
            error_tests = [
                {
                    "name": "empty_query",
                    "payload": {"query": "", "top_k": 5},
                    "expected_status": 422  # Validation error
                },
                {
                    "name": "invalid_top_k",
                    "payload": {"query": "query: test", "top_k": -1},
                    "expected_status": 422
                },
                {
                    "name": "oversized_query",
                    "payload": {"query": "query: " + "x" * 3000, "top_k": 5},
                    "expected_status": 422
                },
                {
                    "name": "invalid_similarity",
                    "payload": {"query": "query: test", "top_k": 5, "similarity_cutoff": 2.0},
                    "expected_status": 422
                }
            ]
            
            error_results = []
            
            for test in error_tests:
                success, duration, result, status = await self._make_request(
                    session, "/retrieve", test["payload"]
                )
                
                # For error tests, "success" means getting expected error status
                test_success = status == test["expected_status"]
                
                error_result = {
                    "test_name": test["name"],
                    "expected_status": test["expected_status"],
                    "actual_status": status,
                    "handled_gracefully": test_success,
                    "response_time_ms": duration * 1000
                }
                
                error_results.append(error_result)
                
                print(f"   {test['name']}: {'✅' if test_success else '❌'} "
                      f"(expected {test['expected_status']}, got {status})")
            
            # Test service recovery after errors
            print("   Testing service recovery...")
            recovery_payload = {
                "query": "query: Zulficore protokolü nedir?",
                "top_k": 5
            }
            
            success, duration, result, status = await self._make_request(
                session, "/retrieve", recovery_payload
            )
            
            recovery_success = success and status == 200
            
            details = {
                "error_tests": len(error_tests),
                "gracefully_handled": sum(1 for r in error_results if r["handled_gracefully"]),
                "error_results": error_results,
                "service_recovery": recovery_success,
                "recovery_time_ms": duration * 1000 if recovery_success else None
            }
            
            overall_success = (details["gracefully_handled"] >= len(error_tests) * 0.75 and 
                             recovery_success)
            
            self._record_result("error_recovery", overall_success, 0, details)
            
            print(f"✅ Graceful handling: {details['gracefully_handled']}/{len(error_tests)}")
            print(f"✅ Service recovery: {'✅' if recovery_success else '❌'}")
            
            return overall_success
    
    def _record_result(self, test_name: str, success: bool, duration: float, details: dict, error: str = None):
        """Record test result"""
        result = StressTestResult(test_name, success, duration, details, error)
        self.results.append(result)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive stress test report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        
        # Performance aggregation
        all_latencies = []
        all_throughputs = []
        
        for result in self.results:
            if 'avg_latency_ms' in result.details:
                all_latencies.append(result.details['avg_latency_ms'])
            if 'throughput_qps' in result.details:
                all_throughputs.append(result.details['throughput_qps'])
        
        performance_summary = {}
        if all_latencies:
            performance_summary["latency"] = {
                "avg_ms": statistics.mean(all_latencies),
                "p95_ms": np.percentile(all_latencies, 95),
                "max_ms": max(all_latencies),
                "min_ms": min(all_latencies)
            }
        
        if all_throughputs:
            performance_summary["throughput"] = {
                "avg_qps": statistics.mean(all_throughputs),
                "max_qps": max(all_throughputs),
                "min_qps": min(all_throughputs)
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
            "test_results": [
                {
                    "test": r.test_name,
                    "status": "PASS" if r.success else "FAIL",
                    "duration": r.duration,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.results
            ],
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Check individual test results for recommendations
        for result in self.results:
            if result.test_name == "single_query_performance":
                if result.details.get("avg_latency_ms", 0) > 100:
                    recommendations.append("Consider optimizing query processing - average latency is high")
                
            elif result.test_name == "batch_processing_stress":
                if result.details.get("avg_throughput_qps", 0) < 50:
                    recommendations.append("Consider increasing batch processing capacity")
                
            elif result.test_name == "concurrent_async_load":
                if result.details.get("avg_success_rate", 0) < 95:
                    recommendations.append("Improve concurrent request handling reliability")
                    
            elif result.test_name == "accuracy_validation":
                if result.details.get("avg_accuracy", 0) < 80:
                    recommendations.append("Review document indexing and query processing for better accuracy")
                    
            elif result.test_name == "memory_cpu_stress":
                if result.details.get("max_memory_mb", 0) > 1024:
                    recommendations.append("Monitor memory usage - consider memory optimization")
                if result.details.get("max_cpu_percent", 0) > 80:
                    recommendations.append("High CPU usage detected - consider horizontal scaling")
        
        if not recommendations:
            recommendations.append("System performance is excellent - no immediate optimizations needed")
        
        return recommendations
    
    async def run_full_stress_test(self) -> Dict[str, Any]:
        """Run complete stress test suite"""
        print("🔥 PRODUCTION STRESS TEST SUITE")
        print("Testing RAG System with Zulficore Queries")
        print("=" * 80)
        
        # Test sequence
        test_functions = [
            ("Service Health Check", self.test_01_service_health),
            ("Single Query Performance", self.test_02_single_query_performance),
            ("Batch Processing Stress", self.test_03_batch_processing_stress),
            ("Concurrent Async Load", self.test_04_concurrent_async_load),
            ("Accuracy Validation", self.test_05_accuracy_validation),
            ("Memory & CPU Stress", self.test_06_memory_and_cpu_stress),
            ("Error Recovery", self.test_07_error_recovery)
        ]
        
        print(f"\nRunning {len(test_functions)} stress tests...\n")
        
        for i, (test_name, test_func) in enumerate(test_functions, 1):
            print(f"{'='*80}")
            print(f"TEST {i}/{len(test_functions)}: {test_name}")
            print(f"{'='*80}")
            
            try:
                await test_func()
            except Exception as e:
                self._record_result(test_name.lower().replace(' ', '_'), False, 0, {}, str(e))
                print(f"❌ Test failed with exception: {e}")
            
            print()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        print("=" * 80)
        print("📊 STRESS TEST RESULTS")
        print("=" * 80)
        
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
            
            if "throughput" in perf:
                thr = perf["throughput"]
                print(f"   Average Throughput: {thr['avg_qps']:.1f} q/s")
                print(f"   Peak Throughput: {thr['max_qps']:.1f} q/s")
        
        print("\n🎯 Recommendations:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")
        
        # Save detailed report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"stress_test_report_{timestamp}.json")
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed report saved to: {report_path}")
        
        return report

async def main():
    """Main stress test execution"""
    print("🚀 Starting Production Stress Test...")
    
    try:
        tester = ProductionStressTester()
        report = await tester.run_full_stress_test()
        
        success = report["summary"]["overall_status"] == "PASS"
        
        if success:
            print("\n🎉 All stress tests passed!")
            print("🚀 System is production-ready!")
        else:
            print("\n⚠️  Some tests failed - review the report")
            print("📋 Address issues before production deployment")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️  Stress test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Stress test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)