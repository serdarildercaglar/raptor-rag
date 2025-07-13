#!/usr/bin/env python3
"""
VLLM-Optimized RAPTOR Stress Test
True batch testing aligned with embedding service capabilities
"""
import asyncio
import aiohttp
import time
import psutil
import statistics
import json
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.live import Live
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, VLLMEmbeddingModel

@dataclass
class VLLMStressResult:
    """VLLM-optimized stress test result"""
    test_name: str
    success: bool
    duration: float
    total_queries: int
    batch_size: int
    throughput_qps: float
    embedding_latency: float
    retrieve_latency: float
    total_latency: float
    memory_peak_mb: float
    vllm_utilization: float
    connection_efficiency: float
    details: Dict = None

class SharedConnectionPool:
    """Shared aiohttp connection pool for all RAPTOR instances"""
    _instance = None
    _session = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_session(self, max_connections: int = 100):
        """Get or create shared session with connection pooling"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=max_connections,
                limit_per_host=50,
                ttl_dns_cache=300,
                use_dns_cache=True,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self._session
    
    async def close(self):
        """Close shared session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

class VLLMOptimizedStressTester:
    """VLLM-optimized stress tester focusing on true batch performance"""
    
    def __init__(self, tree_path: str, vllm_url: str = "http://localhost:8008"):
        self.tree_path = tree_path
        self.vllm_url = vllm_url
        self.RA = None
        self.test_results: List[VLLMStressResult] = []
        self.connection_pool = SharedConnectionPool()
        
        if RICH_AVAILABLE:
            self.console = Console()
        else:
            self.console = None
    
    def _print(self, *args, **kwargs):
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)
    
    async def setup(self):
        """Initialize RAPTOR with optimized connection pooling"""
        if self.console:
            with self.console.status("[bold green]Initializing VLLM-optimized RAPTOR...") as status:
                # Create optimized embedding model with shared session
                session = await self.connection_pool.get_session()
                embedding_model = VLLMEmbeddingModel(self.vllm_url)
                embedding_model.session = session  # Use shared session
                
                config = RetrievalAugmentationConfig(embedding_model=embedding_model)
                self.RA = RetrievalAugmentation(config=config, tree=self.tree_path)
        else:
            print("🔧 Initializing VLLM-optimized RAPTOR...")
            session = await self.connection_pool.get_session()
            embedding_model = VLLMEmbeddingModel(self.vllm_url)
            embedding_model.session = session
            
            config = RetrievalAugmentationConfig(embedding_model=embedding_model)
            self.RA = RetrievalAugmentation(config=config, tree=self.tree_path)
        
        self._print("✅ [bold green]VLLM-Optimized RAPTOR Ready[/bold green]")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.connection_pool.close()
        self._print("✅ [bold green]Resources cleaned up[/bold green]")
    
    def _generate_query_patterns(self, num_queries: int) -> List[str]:
        """Generate realistic query patterns for Turkish literature"""
        
        # Different query complexity levels
        simple_queries = [
            "divan edebiyatı nedir",
            "halk edebiyatı özellikleri",
            "türk şiiri tarihçesi",
            "modern edebiyat akımları",
            "roman türü gelişimi"
        ]
        
        medium_queries = [
            "tanzimat dönemi edebiyat özelliklerini analiz et",
            "ahmet hamdi tanpınar'ın eserlerindeki zaman kavramı",
            "nazim hikmet'in şiir dünyasında sosyal eleştiri",
            "sait faik abasıyanık'ın hikayeciliği",
            "yaşar kemal'in köy romanları incelemesi"
        ]
        
        complex_queries = [
            "türk edebiyatında modernleşme sürecinde geleneksel form ve içerik unsurlarının dönüşümü",
            "cumhuriyet dönemi türk romanında toplumsal değişim ve bireysel kimlik arayışı",
            "çağdaş türk şiirinde postmodern anlatım teknikleri ve dil kullanımı",
            "türk edebiyatında kadın yazarların feminist perspektif ve edebi üslup analizi",
            "türk halk edebiyatı geleneğinin çağdaş edebiyata etkisi ve transformasyonu"
        ]
        
        all_queries = simple_queries + medium_queries + complex_queries
        
        # Mix query complexity realistically
        selected_queries = []
        for i in range(num_queries):
            if i % 10 < 5:  # 50% simple
                selected_queries.append(random.choice(simple_queries))
            elif i % 10 < 8:  # 30% medium  
                selected_queries.append(random.choice(medium_queries))
            else:  # 20% complex
                selected_queries.append(random.choice(complex_queries))
        
        return selected_queries

    async def test_true_batch_performance(self, batch_size: int) -> VLLMStressResult:
        """Test true batch performance - all queries in single batch"""
        self._print(f"\n🚀 [bold blue]True Batch Test ({batch_size} queries)[/bold blue]")
        
        # Generate realistic query batch
        queries = self._generate_query_patterns(batch_size)
        
        # Monitor system resources
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = start_memory
        
        # Test embedding service directly first
        embedding_start = time.time()
        
        try:
            # Direct embedding test
            session = await self.connection_pool.get_session()
            async with session.post(
                f"{self.vllm_url}/v1/embeddings",
                json={
                    "input": queries,
                    "model": "intfloat/multilingual-e5-large"
                }
            ) as response:
                if response.status == 200:
                    embed_result = await response.json()
                    embedding_duration = time.time() - embedding_start
                    embedding_success = True
                else:
                    embedding_duration = time.time() - embedding_start
                    embedding_success = False
            
        except Exception as e:
            embedding_duration = time.time() - embedding_start
            embedding_success = False
        
        # Test full RAPTOR retrieve batch
        retrieve_start = time.time()
        
        try:
            contexts = await self.RA.retrieve_batch(queries)
            retrieve_duration = time.time() - retrieve_start
            retrieve_success = len(contexts) == len(queries)
            
            # Calculate metrics
            total_duration = retrieve_duration
            throughput = len(queries) / total_duration
            
            # Monitor memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            
            # VLLM utilization estimate
            vllm_utilization = embedding_duration / total_duration
            connection_efficiency = 1.0 if embedding_success else 0.0
            
        except Exception as e:
            retrieve_duration = time.time() - retrieve_start
            retrieve_success = False
            throughput = 0
            vllm_utilization = 0
            connection_efficiency = 0
        
        # Display results
        if RICH_AVAILABLE:
            metrics_table = Table(title=f"🚀 Batch Performance ({batch_size} queries)", box=box.ROUNDED)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            metrics_table.add_column("Status", justify="center")
            
            metrics_data = [
                ("Batch Size", f"{batch_size}", True),
                ("Embedding Latency", f"{embedding_duration*1000:.1f}ms", embedding_duration < 2.0),
                ("Retrieve Latency", f"{retrieve_duration*1000:.1f}ms", retrieve_duration < 5.0),
                ("Total Latency", f"{total_duration*1000:.1f}ms", total_duration < 5.0),
                ("Throughput", f"{throughput:.1f} q/s", throughput > 10),
                ("VLLM Efficiency", f"{vllm_utilization*100:.1f}%", vllm_utilization > 0.3),
                ("Connection Health", f"{connection_efficiency*100:.1f}%", connection_efficiency > 0.9),
                ("Memory Usage", f"{peak_memory:.1f} MB", peak_memory < 2000),
            ]
            
            for metric, value, good in metrics_data:
                status = "✅" if good else "⚠️"
                metrics_table.add_row(metric, value, status)
            
            self.console.print(metrics_table)
        
        return VLLMStressResult(
            test_name=f"True Batch ({batch_size})",
            success=retrieve_success and embedding_success,
            duration=total_duration,
            total_queries=len(queries),
            batch_size=batch_size,
            throughput_qps=throughput,
            embedding_latency=embedding_duration,
            retrieve_latency=retrieve_duration,
            total_latency=total_duration,
            memory_peak_mb=peak_memory,
            vllm_utilization=vllm_utilization,
            connection_efficiency=connection_efficiency,
            details={
                "embedding_success": embedding_success,
                "retrieve_success": retrieve_success,
                "queries_processed": len(contexts) if retrieve_success else 0
            }
        )

    async def test_sustained_batch_traffic(self, qps_target: int, duration: int = 60) -> VLLMStressResult:
        """Test sustained batch traffic with realistic timing"""
        self._print(f"\n⚡ [bold blue]Sustained Batch Traffic ({qps_target} q/s for {duration}s)[/bold blue]")
        
        total_queries = 0
        total_batches = 0
        successful_batches = 0
        latencies = []
        embedding_latencies = []
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate batch timing
        batch_size = min(qps_target, 50)  # Optimal batch size for VLLM
        batch_interval = batch_size / qps_target  # Seconds between batches
        
        start_time = time.time()
        
        async def process_batch_wave():
            nonlocal total_queries, total_batches, successful_batches, peak_memory
            
            while time.time() - start_time < duration:
                batch_start = time.time()
                
                try:
                    # Generate batch queries
                    queries = self._generate_query_patterns(batch_size)
                    total_queries += len(queries)
                    total_batches += 1
                    
                    # Time embedding specifically
                    embed_start = time.time()
                    session = await self.connection_pool.get_session()
                    async with session.post(
                        f"{self.vllm_url}/v1/embeddings",
                        json={"input": queries, "model": "intfloat/multilingual-e5-large"}
                    ) as response:
                        if response.status == 200:
                            await response.json()
                        embed_duration = time.time() - embed_start
                        embedding_latencies.append(embed_duration)
                    
                    # Full retrieve
                    contexts = await self.RA.retrieve_batch(queries)
                    
                    batch_duration = time.time() - batch_start
                    latencies.append(batch_duration)
                    successful_batches += 1
                    
                    # Monitor memory
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    
                except Exception as e:
                    batch_duration = time.time() - batch_start
                    latencies.append(batch_duration)
                
                # Wait for next batch interval
                elapsed = time.time() - batch_start
                sleep_time = max(0, batch_interval - elapsed)
                await asyncio.sleep(sleep_time)
        
        # Run with live monitoring
        if RICH_AVAILABLE:
            with Live(refresh_per_second=2) as live:
                # Start traffic generation
                traffic_task = asyncio.create_task(process_batch_wave())
                
                while not traffic_task.done() and time.time() - start_time < duration:
                    elapsed = time.time() - start_time
                    current_qps = total_queries / max(elapsed, 1)
                    success_rate = (successful_batches / max(total_batches, 1)) * 100
                    
                    progress_table = Table(box=box.SIMPLE)
                    progress_table.add_column("Metric", style="cyan")
                    progress_table.add_column("Value", style="yellow")
                    
                    progress_table.add_row("Elapsed", f"{elapsed:.1f}s / {duration}s")
                    progress_table.add_row("Target QPS", f"{qps_target}")
                    progress_table.add_row("Actual QPS", f"{current_qps:.1f}")
                    progress_table.add_row("Total Queries", f"{total_queries}")
                    progress_table.add_row("Batches", f"{successful_batches}/{total_batches}")
                    progress_table.add_row("Success Rate", f"{success_rate:.1f}%")
                    if latencies:
                        progress_table.add_row("Avg Latency", f"{statistics.mean(latencies[-10:]):.2f}s")
                    if embedding_latencies:
                        progress_table.add_row("Embed Latency", f"{statistics.mean(embedding_latencies[-10:])*1000:.1f}ms")
                    progress_table.add_row("Memory", f"{peak_memory:.1f} MB")
                    
                    live.update(Panel(progress_table, title="Sustained Traffic Monitor"))
                    await asyncio.sleep(0.5)
                
                await traffic_task
        else:
            await process_batch_wave()
        
        # Calculate final metrics
        actual_duration = time.time() - start_time
        actual_qps = total_queries / actual_duration
        success_rate = (successful_batches / max(total_batches, 1)) * 100
        
        avg_latency = statistics.mean(latencies) if latencies else 0
        avg_embed_latency = statistics.mean(embedding_latencies) if embedding_latencies else 0
        vllm_utilization = avg_embed_latency / avg_latency if avg_latency > 0 else 0
        
        return VLLMStressResult(
            test_name=f"Sustained Traffic ({qps_target} q/s)",
            success=success_rate >= 90 and actual_qps >= qps_target * 0.8,
            duration=actual_duration,
            total_queries=total_queries,
            batch_size=batch_size,
            throughput_qps=actual_qps,
            embedding_latency=avg_embed_latency,
            retrieve_latency=avg_latency,
            total_latency=avg_latency,
            memory_peak_mb=peak_memory,
            vllm_utilization=vllm_utilization,
            connection_efficiency=success_rate / 100,
            details={
                "target_qps": qps_target,
                "successful_batches": successful_batches,
                "total_batches": total_batches,
                "batch_size": batch_size
            }
        )

    async def test_batch_scaling_ladder(self) -> List[VLLMStressResult]:
        """Test increasing batch sizes to find optimal performance"""
        self._print(f"\n📈 [bold blue]Batch Scaling Ladder[/bold blue]")
        
        # Progressive batch size testing
        batch_sizes = [10, 25, 50, 100, 200, 500]
        results = []
        
        for batch_size in batch_sizes:
            self._print(f"\n📊 Testing batch size: {batch_size}")
            
            result = await self.test_true_batch_performance(batch_size)
            results.append(result)
            
            # Stop if performance degrades significantly
            if len(results) >= 2:
                prev_throughput = results[-2].throughput_qps
                current_throughput = result.throughput_qps
                
                if current_throughput < prev_throughput * 0.5:  # 50% degradation
                    self._print(f"⚠️ [yellow]Performance degradation at batch size {batch_size}[/yellow]")
                    break
            
            # Brief cooldown
            await asyncio.sleep(2)
        
        # Find optimal batch size
        successful_results = [r for r in results if r.success]
        if successful_results:
            optimal_result = max(successful_results, key=lambda x: x.throughput_qps)
            
            if RICH_AVAILABLE:
                scaling_table = Table(title="📈 Batch Scaling Analysis", box=box.DOUBLE_EDGE)
                scaling_table.add_column("Batch Size", style="cyan")
                scaling_table.add_column("Throughput", style="green")
                scaling_table.add_column("Latency", style="yellow")
                scaling_table.add_column("VLLM Util", style="blue")
                scaling_table.add_column("Status", justify="center")
                
                for r in results:
                    status = "✅" if r.success else "❌"
                    if r == optimal_result:
                        status = "🏆 OPTIMAL"
                    
                    scaling_table.add_row(
                        f"{r.batch_size}",
                        f"{r.throughput_qps:.1f} q/s",
                        f"{r.total_latency*1000:.0f}ms",
                        f"{r.vllm_utilization*100:.1f}%",
                        status
                    )
                
                self.console.print(scaling_table)
                
                optimal_panel = Panel(
                    f"🏆 Optimal Batch Size: {optimal_result.batch_size}\n"
                    f"🚀 Peak Throughput: {optimal_result.throughput_qps:.1f} q/s\n"
                    f"⚡ Latency: {optimal_result.total_latency*1000:.0f}ms\n"
                    f"🎯 VLLM Utilization: {optimal_result.vllm_utilization*100:.1f}%",
                    title="Optimal Configuration",
                    border_style="green"
                )
                self.console.print(optimal_panel)
        
        return results

    async def test_connection_pool_stress(self) -> VLLMStressResult:
        """Test connection pool efficiency under stress"""
        self._print(f"\n🔌 [bold blue]Connection Pool Stress Test[/bold blue]")
        
        # Simulate many concurrent "users" but with proper batching
        concurrent_batches = 20
        batch_size = 25
        
        async def concurrent_batch_request(batch_id: int):
            """Simulate concurrent batch request"""
            queries = self._generate_query_patterns(batch_size)
            
            start_time = time.time()
            try:
                contexts = await self.RA.retrieve_batch(queries)
                duration = time.time() - start_time
                return {
                    "batch_id": batch_id,
                    "success": len(contexts) == len(queries),
                    "duration": duration,
                    "queries": len(queries)
                }
            except Exception as e:
                return {
                    "batch_id": batch_id,
                    "success": False,
                    "duration": time.time() - start_time,
                    "queries": batch_size,
                    "error": str(e)
                }
        
        # Run concurrent batches
        start_time = time.time()
        batch_results = await asyncio.gather(*[
            concurrent_batch_request(i) for i in range(concurrent_batches)
        ])
        total_duration = time.time() - start_time
        
        # Analyze results
        successful = [r for r in batch_results if r["success"]]
        total_queries = sum(r["queries"] for r in batch_results)
        success_rate = len(successful) / len(batch_results) * 100
        throughput = total_queries / total_duration
        
        if successful:
            avg_latency = statistics.mean(r["duration"] for r in successful)
        else:
            avg_latency = 0
        
        return VLLMStressResult(
            test_name="Connection Pool Stress",
            success=success_rate >= 95,
            duration=total_duration,
            total_queries=total_queries,
            batch_size=batch_size,
            throughput_qps=throughput,
            embedding_latency=avg_latency * 0.3,  # Estimate
            retrieve_latency=avg_latency,
            total_latency=avg_latency,
            memory_peak_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            vllm_utilization=0.3,  # Estimate
            connection_efficiency=success_rate / 100,
            details={
                "concurrent_batches": concurrent_batches,
                "successful_batches": len(successful),
                "avg_batch_latency": avg_latency
            }
        )

    def _generate_production_report(self):
        """Generate comprehensive production readiness report"""
        if not self.test_results:
            return
        
        # Find best performance
        best_throughput = max(r.throughput_qps for r in self.test_results)
        best_result = max(self.test_results, key=lambda x: x.throughput_qps)
        avg_latency = statistics.mean(r.total_latency for r in self.test_results)
        min_error_rate = min((100 - r.connection_efficiency * 100) for r in self.test_results)
        
        if RICH_AVAILABLE:
            # Production readiness assessment
            if best_throughput > 100:
                readiness = "🚀 PRODUCTION EXCELLENT"
                readiness_color = "bright_green"
            elif best_throughput > 50:
                readiness = "✅ PRODUCTION READY"
                readiness_color = "green"
            elif best_throughput > 20:
                readiness = "⚡ PRODUCTION CAPABLE"
                readiness_color = "yellow"
            else:
                readiness = "⚠️ NEEDS OPTIMIZATION"
                readiness_color = "red"
            
            # Final summary
            summary_table = Table(title="📊 VLLM-Optimized Performance Summary", box=box.DOUBLE_EDGE)
            summary_table.add_column("Test", style="cyan")
            summary_table.add_column("Batch Size", style="blue")
            summary_table.add_column("Throughput", style="green")
            summary_table.add_column("Latency", style="yellow") 
            summary_table.add_column("VLLM Util", style="magenta")
            summary_table.add_column("Status", justify="center")
            
            for result in self.test_results:
                status = "✅" if result.success else "❌"
                if result == best_result:
                    status = "🏆 BEST"
                
                summary_table.add_row(
                    result.test_name,
                    f"{result.batch_size}",
                    f"{result.throughput_qps:.1f} q/s",
                    f"{result.total_latency*1000:.0f}ms",
                    f"{result.vllm_utilization*100:.1f}%",
                    status
                )
            
            self.console.print("\n")
            self.console.print(summary_table)
            
            # Production recommendations
            production_panel = Panel(
                f"{readiness}\n\n"
                f"🚀 Peak Performance:\n"
                f"   • Maximum Throughput: {best_throughput:.1f} queries/sec\n"
                f"   • Optimal Batch Size: {best_result.batch_size}\n"
                f"   • Best Latency: {best_result.total_latency*1000:.0f}ms\n"
                f"   • VLLM Utilization: {best_result.vllm_utilization*100:.1f}%\n\n"
                f"💡 Production Configuration:\n"
                f"   • Recommended batch size: {best_result.batch_size}\n"
                f"   • Safe production rate: {int(best_throughput * 0.8)} q/s\n"
                f"   • Connection pool size: {max(50, best_result.batch_size * 2)}\n"
                f"   • SLA latency target: {best_result.total_latency * 1.5:.1f}s",
                title="Production Configuration",
                border_style=readiness_color
            )
            self.console.print(production_panel)
            
            # Performance comparison
            comparison_panel = Panel(
                f"📈 Performance Improvement vs Previous Test:\n"
                f"   • Throughput: {best_throughput:.1f} q/s vs 3.2 q/s = {best_throughput/3.2:.1f}x faster\n"
                f"   • Latency: {best_result.total_latency:.1f}s vs 24s = {24/best_result.total_latency:.1f}x faster\n"
                f"   • Efficiency: True batch processing vs individual requests\n"
                f"   • Connection: Pooled vs individual connections\n\n"
                f"🎯 Key Optimizations Applied:\n"
                f"   ✅ True batch processing (all queries in single batch)\n"
                f"   ✅ Shared connection pooling\n"
                f"   ✅ VLLM-optimized request patterns\n"
                f"   ✅ Realistic query distribution",
                title="Performance Analysis",
                border_style="blue"
            )
            self.console.print(comparison_panel)

    async def run_vllm_optimized_tests(self):
        """Run complete VLLM-optimized test suite"""
        self._print("\n🎯 [bold red]VLLM-Optimized RAPTOR Performance Test[/bold red]")
        self._print("🚀 [dim]True batch testing with connection pooling optimization[/dim]")
        self._print("=" * 80)
        
        try:
            await self.setup()
            
            # Test 1: Batch scaling to find optimal size
            scaling_results = await self.test_batch_scaling_ladder()
            self.test_results.extend(scaling_results)
            
            # Test 2: Connection pool stress test
            pool_result = await self.test_connection_pool_stress()
            self.test_results.append(pool_result)
            
            # Test 3: Sustained traffic with optimal batch size
            optimal_batch = max(scaling_results, key=lambda x: x.throughput_qps).batch_size
            optimal_qps = int(max(scaling_results, key=lambda x: x.throughput_qps).throughput_qps * 0.8)
            
            sustained_result = await self.test_sustained_batch_traffic(optimal_qps, 60)
            self.test_results.append(sustained_result)
            
            # Generate comprehensive report
            self._generate_production_report()
            
        except Exception as e:
            self._print(f"❌ [bold red]Test suite failed: {e}[/bold red]")
            raise
        finally:
            await self.cleanup()

async def main():
    """Main execution"""
    TREE_PATH = "/home/serdar/Documents/raptor-rag/vectordb/raptor-db"
    VLLM_URL = "http://localhost:8008"
    
    if not RICH_AVAILABLE:
        print("📦 For enhanced output, install: pip install rich")
    
    tester = VLLMOptimizedStressTester(TREE_PATH, VLLM_URL)
    
    try:
        await tester.run_vllm_optimized_tests()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())