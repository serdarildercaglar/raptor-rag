#!/usr/bin/env python3
"""
FAISS RAPTOR Stress Test
Same test scenarios as before, but using FAISS-GPU integration
"""
import asyncio
import time
import psutil
import statistics
import random
from typing import List, Dict, Optional
from dataclasses import dataclass

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

from raptor.production_raptor import ProductionRAPTOR

@dataclass
class FaissStressResult:
    """FAISS stress test result"""
    test_name: str
    success: bool
    duration: float
    total_queries: int
    batch_size: int
    throughput_qps: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    error_rate: float
    concurrent_users: int
    memory_peak_mb: float
    cpu_peak_percent: float
    details: Dict = None

class FaissStressTester:
    """FAISS stress tester - same tests as VLLMOptimizedStressTester"""
    
    def __init__(self, tree_path: str, vllm_url: str = "http://localhost:8008"):
        self.tree_path = tree_path
        self.vllm_url = vllm_url
        self.raptor = None
        self.test_results: List[FaissStressResult] = []
        
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
        """Initialize FAISS RAPTOR"""
        if self.console:
            with self.console.status("[bold green]Loading FAISS RAPTOR...") as status:
                self.raptor = ProductionRAPTOR(self.tree_path, self.vllm_url)
                await self.raptor.initialize()
        else:
            print("🔧 Loading FAISS RAPTOR...")
            self.raptor = ProductionRAPTOR(self.tree_path, self.vllm_url)
            await self.raptor.initialize()
        
        self._print("✅ [bold green]FAISS RAPTOR Stress Test Ready[/bold green]")
    
    async def cleanup(self):
        if self.raptor:
            await self.raptor.close()
        self._print("✅ [bold green]Resources cleaned up[/bold green]")
    
    def _generate_realistic_queries(self, user_id: int, num_queries: int) -> List[str]:
        """Generate realistic queries for different user personas"""
        
        user_personas = [
            # Academic researcher
            [
                "edebiyat teorileri ve eleştiri yöntemleri",
                "postmodern türk edebiyatı analizi",
                "karşılaştırmalı edebiyat incelemeleri",
                "narratoloji ve metin analizi",
                "edebiyat sosyolojisi yaklaşımları"
            ],
            # Student
            [
                "divan edebiyatı özellikleri nelerdir",
                "modern türk şairleri kimlerdir",
                "roman ve hikaye farkları",
                "şiir türleri ve örnekleri",
                "edebiyat akımları özet"
            ],
            # Teacher
            [
                "öğrencilere edebiyat nasıl öğretilir",
                "edebiyat dersi etkinlikleri",
                "klasik eserlerin analiz yöntemleri",
                "edebiyat tarihinin dönemleri",
                "yazınsal metin inceleme teknikleri"
            ],
            # General reader
            [
                "popüler türk romanları önerileri",
                "güncel edebiyat eserleri",
                "okuma listesi önerileri",
                "en iyi türk şiirleri",
                "edebiyat ödüllü kitaplar"
            ],
            # Journalist/Writer
            [
                "çağdaş türk edebiyatında trendler",
                "yeni nesil yazarlar kimler",
                "edebiyat dünyasından haberler",
                "yazarlarla röportaj örnekleri",
                "kitap eleştirisi nasıl yazılır"
            ]
        ]
        
        persona = user_personas[user_id % len(user_personas)]
        queries = random.choices(persona, k=num_queries)
        
        return [f"{query} (user-{user_id})" for query in queries]

    async def test_true_batch_performance(self, batch_size: int) -> FaissStressResult:
        """Test true batch performance - all queries in single batch"""
        self._print(f"\n🚀 [bold blue]True Batch Test ({batch_size} queries)[/bold blue]")
        
        # Generate realistic query batch
        queries = []
        for i in range(batch_size):
            user_queries = self._generate_realistic_queries(i, 1)
            queries.extend(user_queries)
        
        # Monitor system resources
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = start_memory
        peak_cpu = 0
        
        # Test batch retrieval
        start_time = time.time()
        
        try:
            contexts = await self.raptor.retrieve_batch(queries)
            duration = time.time() - start_time
            
            # Calculate metrics
            throughput = len(queries) / duration
            success = len(contexts) == len(queries)
            
            # Monitor memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory = max(peak_memory, current_memory)
            current_cpu = psutil.cpu_percent()
            peak_cpu = max(peak_cpu, current_cpu)
            
        except Exception as e:
            duration = time.time() - start_time
            success = False
            throughput = 0
            contexts = []
        
        # Display results
        if RICH_AVAILABLE:
            metrics_table = Table(title=f"🚀 Batch Performance ({batch_size} queries)", box=box.ROUNDED)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            metrics_table.add_column("Status", justify="center")
            
            metrics_data = [
                ("Batch Size", f"{batch_size}", True),
                ("Total Latency", f"{duration*1000:.1f}ms", duration < 5.0),
                ("Throughput", f"{throughput:.1f} q/s", throughput > 10),
                ("Success Rate", f"{len(contexts)}/{len(queries)}", success),
                ("Memory Usage", f"{peak_memory:.1f} MB", peak_memory < 2000),
                ("CPU Peak", f"{peak_cpu:.1f}%", peak_cpu < 90),
            ]
            
            for metric, value, good in metrics_data:
                status = "✅" if good else "⚠️"
                metrics_table.add_row(metric, value, status)
            
            self.console.print(metrics_table)
        
        return FaissStressResult(
            test_name=f"True Batch ({batch_size})",
            success=success,
            duration=duration,
            total_queries=len(queries),
            batch_size=batch_size,
            throughput_qps=throughput,
            avg_latency=duration,
            p95_latency=duration,
            p99_latency=duration,
            max_latency=duration,
            error_rate=0 if success else 100,
            concurrent_users=1,
            memory_peak_mb=peak_memory,
            cpu_peak_percent=peak_cpu,
            details={
                "success": success,
                "queries_processed": len(contexts)
            }
        )

    async def test_concurrent_burst_load(self, concurrent_users: int = 50) -> FaissStressResult:
        """Test burst of concurrent users"""
        self._print(f"\n💥 [bold red]Burst Load Test ({concurrent_users} concurrent users)[/bold red]")
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = start_memory
        peak_cpu = 0
        
        async def simulate_user_burst(user_id: int):
            """Simulate a user making multiple queries in burst"""
            num_queries = random.randint(2, 4)
            queries = self._generate_realistic_queries(user_id, num_queries)
            
            start = time.time()
            try:
                nonlocal peak_memory, peak_cpu
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                current_cpu = psutil.cpu_percent()
                peak_memory = max(peak_memory, current_memory)
                peak_cpu = max(peak_cpu, current_cpu)
                
                contexts = await self.raptor.retrieve_batch(queries)
                end = time.time()
                
                return {
                    "user_id": user_id,
                    "success": True,
                    "time": end - start,
                    "queries": len(queries),
                    "total_chars": sum(len(ctx) for ctx in contexts),
                    "latency": end - start
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "success": False,
                    "error": str(e),
                    "latency": time.time() - start,
                    "queries": num_queries
                }
        
        # Run burst test with progress monitoring
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Burst test ({concurrent_users} users)...", total=1)
                
                start_time = time.time()
                results = await asyncio.gather(*[simulate_user_burst(i) for i in range(concurrent_users)])
                total_time = time.time() - start_time
                progress.update(task, advance=1)
        else:
            print(f"  💥 Running burst test with {concurrent_users} users...")
            start_time = time.time()
            results = await asyncio.gather(*[simulate_user_burst(i) for i in range(concurrent_users)])
            total_time = time.time() - start_time

        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        if successful:
            total_queries = sum(r['queries'] for r in successful)
            latencies = [r['latency'] for r in successful]
            avg_latency = statistics.mean(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
            max_latency = max(latencies)
            success_rate = len(successful) / concurrent_users * 100
            throughput = total_queries / total_time
        else:
            total_queries = avg_latency = p95_latency = p99_latency = max_latency = success_rate = throughput = 0

        error_rate = (len(failed) / concurrent_users) * 100

        # Create results table
        if RICH_AVAILABLE:
            if throughput > 50:
                perf_color = "bright_green"
                perf_icon = "🚀"
            elif throughput > 20:
                perf_color = "green" 
                perf_icon = "✅"
            elif throughput > 10:
                perf_color = "yellow"
                perf_icon = "⚡"
            else:
                perf_color = "red"
                perf_icon = "⚠️"

            results_data = [
                {"metric": "Concurrent Users", "value": f"{concurrent_users}", "good": True},
                {"metric": "Success Rate", "value": f"{success_rate:.1f}%", "good": success_rate >= 95},
                {"metric": "Total Queries", "value": f"{total_queries}", "good": True},
                {"metric": f"{perf_icon} Throughput", "value": f"{throughput:.1f} q/s", "good": throughput > 10},
                {"metric": "Avg Latency", "value": f"{avg_latency:.2f}s", "good": avg_latency < 5},
                {"metric": "P95 Latency", "value": f"{p95_latency:.2f}s", "good": p95_latency < 10},
                {"metric": "P99 Latency", "value": f"{p99_latency:.2f}s", "good": p99_latency < 15},
                {"metric": "Max Latency", "value": f"{max_latency:.2f}s", "good": max_latency < 20},
                {"metric": "Error Rate", "value": f"{error_rate:.1f}%", "good": error_rate < 5},
                {"metric": "Peak Memory", "value": f"{peak_memory:.1f} MB", "good": peak_memory < 2000},
                {"metric": "Peak CPU", "value": f"{peak_cpu:.1f}%", "good": peak_cpu < 80},
            ]
            
            table = Table(title=f"💥 Burst Load Performance ({concurrent_users} users)", box=box.ROUNDED)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            table.add_column("Status", justify="center")
            
            for result in results_data:
                status = "✅" if result.get("good", True) else "⚠️"
                table.add_row(result["metric"], result["value"], status)
            
            self.console.print(table)
            
            # Production capacity estimation
            estimated_capacity = int(throughput * 0.7)  # 70% safety margin
            capacity_panel = Panel(
                f"📊 Estimated Production Capacity:\n"
                f"   • Safe concurrent queries: ~{estimated_capacity} q/s\n"
                f"   • Max concurrent users: ~{int(estimated_capacity / 2.5)} users\n"
                f"   • Daily query capacity: ~{estimated_capacity * 86400:,} queries",
                title="Capacity Estimation",
                border_style=perf_color
            )
            self.console.print(capacity_panel)
        else:
            print(f"  Results: {len(successful)}/{concurrent_users} success ({success_rate:.1f}%)")
            print(f"  Throughput: {throughput:.1f} q/s")
            print(f"  Latency: Avg {avg_latency:.2f}s, P95 {p95_latency:.2f}s")

        return FaissStressResult(
            test_name=f"Burst Load ({concurrent_users})",
            success=success_rate >= 90 and error_rate < 10,
            duration=total_time,
            total_queries=total_queries,
            batch_size=int(total_queries / concurrent_users) if concurrent_users > 0 else 0,
            throughput_qps=throughput,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            max_latency=max_latency,
            error_rate=error_rate,
            concurrent_users=concurrent_users,
            memory_peak_mb=peak_memory,
            cpu_peak_percent=peak_cpu,
            details={"successful_users": len(successful), "failed_users": len(failed)}
        )

    async def test_batch_scaling_ladder(self) -> List[FaissStressResult]:
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
                scaling_table.add_column("Status", justify="center")
                
                for r in results:
                    status = "✅" if r.success else "❌"
                    if r == optimal_result:
                        status = "🏆 OPTIMAL"
                    
                    scaling_table.add_row(
                        f"{r.batch_size}",
                        f"{r.throughput_qps:.1f} q/s",
                        f"{r.avg_latency*1000:.0f}ms",
                        status
                    )
                
                self.console.print(scaling_table)
                
                optimal_panel = Panel(
                    f"🏆 Optimal Batch Size: {optimal_result.batch_size}\n"
                    f"🚀 Peak Throughput: {optimal_result.throughput_qps:.1f} q/s\n"
                    f"⚡ Latency: {optimal_result.avg_latency*1000:.0f}ms",
                    title="Optimal Configuration",
                    border_style="green"
                )
                self.console.print(optimal_panel)
        
        return results

    async def test_scalability_ladder(self) -> List[FaissStressResult]:
        """Test increasing concurrent users to find breaking point"""
        self._print(f"\n🪜 [bold blue]Scalability Ladder Test[/bold blue]")
        
        load_levels = [25, 50, 100, 150, 200]  # Progressive load increase
        results = []
        
        for users in load_levels:
            self._print(f"\n📈 Testing {users} concurrent users...")
            
            result = await self.test_concurrent_burst_load(users)
            results.append(result)
            
            # Stop if we hit breaking point
            if result.error_rate > 20 or result.throughput_qps < 5:
                self._print(f"⚠️ [yellow]Breaking point reached at {users} users[/yellow]")
                break
            
            # Brief cooldown between tests
            await asyncio.sleep(2)
        
        # Find optimal load point
        successful_results = [r for r in results if r.success]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.throughput_qps)
            optimal_users = best_result.concurrent_users
            
            if RICH_AVAILABLE:
                scalability_table = Table(title="🪜 Scalability Analysis", box=box.DOUBLE_EDGE)
                scalability_table.add_column("Users", style="cyan")
                scalability_table.add_column("Throughput", style="green")
                scalability_table.add_column("P95 Latency", style="yellow")
                scalability_table.add_column("Error Rate", style="red")
                scalability_table.add_column("Status", justify="center")
                
                for r in results:
                    status = "✅" if r.success else "❌"
                    if r == best_result:
                        status = "🏆 OPTIMAL"
                    
                    scalability_table.add_row(
                        f"{r.concurrent_users}",
                        f"{r.throughput_qps:.1f} q/s",
                        f"{r.p95_latency:.2f}s",
                        f"{r.error_rate:.1f}%",
                        status
                    )
                
                self.console.print(scalability_table)
                
                optimal_panel = Panel(
                    f"🏆 Optimal Configuration: {optimal_users} concurrent users\n"
                    f"🚀 Peak Throughput: {best_result.throughput_qps:.1f} queries/sec\n"
                    f"⚡ P95 Latency: {best_result.p95_latency:.2f}s\n"
                    f"💡 Recommended production limit: {int(optimal_users * 0.8)} users",
                    title="Optimal Load Point",
                    border_style="green"
                )
                self.console.print(optimal_panel)
        
        return results

    def _generate_final_report(self):
        """Generate comprehensive production readiness report"""
        if not self.test_results:
            return
        
        # Find best performance
        best_throughput = max(r.throughput_qps for r in self.test_results)
        best_result = max(self.test_results, key=lambda x: x.throughput_qps)
        avg_latency = statistics.mean(r.avg_latency for r in self.test_results)
        min_error_rate = min(r.error_rate for r in self.test_results)
        
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
            summary_table = Table(title="📊 FAISS RAPTOR Performance Summary", box=box.DOUBLE_EDGE)
            summary_table.add_column("Test", style="cyan")
            summary_table.add_column("Batch Size", style="blue")
            summary_table.add_column("Throughput", style="green")
            summary_table.add_column("Latency", style="yellow") 
            summary_table.add_column("Status", justify="center")
            
            for result in self.test_results:
                status = "✅" if result.success else "❌"
                if result == best_result:
                    status = "🏆 BEST"
                
                summary_table.add_row(
                    result.test_name,
                    f"{result.batch_size}",
                    f"{result.throughput_qps:.1f} q/s",
                    f"{result.avg_latency*1000:.0f}ms",
                    status
                )
            
            self.console.print("\n")
            self.console.print(summary_table)
            
            # Performance comparison with TreeRetriever
            comparison_panel = Panel(
                f"📈 Performance vs TreeRetriever:\n"
                f"   • Throughput: {best_throughput:.1f} q/s vs 3.4 q/s = {best_throughput/3.4:.1f}x faster\n"
                f"   • Latency: {best_result.avg_latency:.1f}s vs 59.3s = {59.3/best_result.avg_latency:.1f}x faster\n"
                f"   • Technology: FAISS-GPU vs CPU Sequential\n"
                f"   • Index: Optimized Vector Search vs Full Scan\n\n"
                f"🎯 FAISS Benefits:\n"
                f"   ✅ GPU-accelerated similarity computation\n"
                f"   ✅ Optimized vector indexing\n"
                f"   ✅ Native batch processing\n"
                f"   ✅ Production-ready performance",
                title="FAISS vs TreeRetriever Comparison",
                border_style="blue"
            )
            self.console.print(comparison_panel)
            
            # Production recommendations
            production_panel = Panel(
                f"{readiness}\n\n"
                f"🚀 Peak Performance:\n"
                f"   • Maximum Throughput: {best_throughput:.1f} queries/sec\n"
                f"   • Optimal Batch Size: {best_result.batch_size}\n"
                f"   • Best Latency: {best_result.avg_latency*1000:.0f}ms\n\n"
                f"💡 Production Configuration:\n"
                f"   • Recommended batch size: {best_result.batch_size}\n"
                f"   • Safe production rate: {int(best_throughput * 0.8)} q/s\n"
                f"   • SLA latency target: {best_result.avg_latency * 1.5:.1f}s",
                title="Production Configuration",
                border_style=readiness_color
            )
            self.console.print(production_panel)

    async def run_faiss_stress_tests(self):
        """Run complete FAISS stress test suite"""
        self._print("\n🎯 [bold red]FAISS RAPTOR Performance Test Suite[/bold red]")
        self._print("🚀 [dim]GPU-accelerated vector search with production workloads[/dim]")
        self._print("=" * 80)
        
        try:
            await self.setup()
            
            # Test 1: Batch scaling to find optimal size
            scaling_results = await self.test_batch_scaling_ladder()
            self.test_results.extend(scaling_results)
            
            # Test 2: Scalability ladder
            ladder_results = await self.test_scalability_ladder()
            self.test_results.extend(ladder_results)
            
            # Generate comprehensive report
            self._generate_final_report()
            
        except Exception as e:
            self._print(f"❌ [bold red]Test suite failed: {e}[/bold red]")
            raise
        finally:
            await self.cleanup()

async def main():
    """Main execution"""
    TREE_PATH = "/home/serdar/Documents/raptor-rag/vectordb/raptor-db"  # Update this
    VLLM_URL = "http://localhost:8008"
    
    if not RICH_AVAILABLE:
        print("📦 For enhanced output, install: pip install rich")
    
    tester = FaissStressTester(TREE_PATH, VLLM_URL)
    
    try:
        await tester.run_faiss_stress_tests()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())