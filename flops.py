"""
TFLOPS Calculator for Graph Optimization Kernel

Estimates floating-point operations per second for the edge processing kernel.
"""


def calculate_flops_per_edge():
    """
    Calculate FLOPs for processing one edge.

    Per edge computation:
    1. pos_v_x - pos_u_x  -> 1 FLOP (subtraction)
    2. pos_v_y - pos_u_y  -> 1 FLOP (subtraction)
    3. diff_x * weight    -> 1 FLOP (multiplication)
    4. diff_y * weight    -> 1 FLOP (multiplication)
    5. result_x * lr      -> 1 FLOP (multiplication)
    6. result_y * lr      -> 1 FLOP (multiplication)

    Total: 6 FLOPs per edge for the forward deltas
    We compute both +delta and -delta, but the negation is essentially free
    (or counts as 2 more FLOPs if we're strict)
    """
    flops = 6  # Core computation FLOPs
    return flops


def calculate_benchmark_stats(num_nodes, degree, num_epochs, total_time):
    """
    Calculate TFLOPS for the benchmark configuration.

    Args:
        num_nodes: Number of nodes in graph
        degree: Average degree per node
        num_epochs: Number of epochs run
        total_time: Total time in seconds

    Returns:
        dict: Statistics including TFLOPS
    """
    # Calculate number of edges
    num_edges = (num_nodes * degree) // 2

    # FLOPs per edge
    flops_per_edge = calculate_flops_per_edge()

    # Total operations in attractive phase per epoch
    attractive_flops_per_epoch = num_edges * flops_per_edge

    # Total FLOPs across all epochs (attractive phase only)
    total_attractive_flops = attractive_flops_per_epoch * num_epochs

    # Convert to TFLOPS
    total_tflops = total_attractive_flops / 1e12

    # TFLOPS/second
    tflops_per_second = total_tflops / total_time

    return {
        "num_edges": num_edges,
        "flops_per_edge": flops_per_edge,
        "attractive_flops_per_epoch": attractive_flops_per_epoch,
        "total_attractive_flops": total_attractive_flops,
        "total_tflops": total_tflops,
        "tflops_per_second": tflops_per_second,
    }


def calculate_memory_bandwidth(num_edges, total_time, index_dtype_bytes=4):
    """
    Calculate memory bandwidth utilization.

    Args:
        num_edges: Number of edges
        total_time: Total time in seconds
        index_dtype_bytes: Bytes per index (4 for int32, 8 for int64)

    Returns:
        dict: Memory statistics
    """
    # Per edge, we read:
    # - 2 indices (source, dest): 2 * index_dtype_bytes
    # - 4 floats for positions (2 per node): 4 * 4 = 16 bytes
    # - 1 weight: 4 bytes
    bytes_read_per_edge = (2 * index_dtype_bytes) + 16 + 4

    # Per edge, we write:
    # - 4 floats for deltas (2 for u, 2 for v): 4 * 4 = 16 bytes
    # - 2 indices for segment_reduce: 2 * index_dtype_bytes
    bytes_written_per_edge = 16 + (2 * index_dtype_bytes)

    # Total bytes per edge
    bytes_per_edge = bytes_read_per_edge + bytes_written_per_edge

    # Total bytes moved
    total_bytes = num_edges * bytes_per_edge

    # Bandwidth in GB/s
    bandwidth_gbs = (total_bytes / 1e9) / total_time

    return {
        "bytes_read_per_edge": bytes_read_per_edge,
        "bytes_written_per_edge": bytes_written_per_edge,
        "bytes_per_edge": bytes_per_edge,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1e9,
        "bandwidth_gbs": bandwidth_gbs,
    }


def calculate_arithmetic_intensity(flops_per_edge, bytes_per_edge):
    """
    Calculate arithmetic intensity (FLOPs per byte).

    This tells us if we're compute-bound or memory-bound.

    Args:
        flops_per_edge: FLOPs per edge
        bytes_per_edge: Bytes transferred per edge

    Returns:
        float: Arithmetic intensity in FLOPs/byte
    """
    return flops_per_edge / bytes_per_edge


def estimate_peak_performance():
    """
    Provide reference peak performance for common GPUs.

    Returns:
        dict: GPU specs
    """
    gpus = {
        "RTX 4090": {
            "fp32_tflops": 82.6,
            "memory_bandwidth_gbs": 1008,
        },
        "RTX 4080": {
            "fp32_tflops": 48.7,
            "memory_bandwidth_gbs": 716.8,
        },
        "A100 (40GB)": {
            "fp32_tflops": 19.5,
            "memory_bandwidth_gbs": 1555,
        },
        "H100": {
            "fp32_tflops": 51.0,
            "memory_bandwidth_gbs": 3350,
        },
        "V100": {
            "fp32_tflops": 15.7,
            "memory_bandwidth_gbs": 900,
        },
        "RTX 3090": {
            "fp32_tflops": 35.6,
            "memory_bandwidth_gbs": 936,
        },
    }
    return gpus


def analyze_performance(num_nodes, degree, num_epochs, total_time, gpu_name=None):
    """
    Complete performance analysis.

    Args:
        num_nodes: Number of nodes
        degree: Average degree
        num_epochs: Number of epochs
        total_time: Total time in seconds
        gpu_name: Optional GPU name for comparison
    """
    print("=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Calculate FLOPs
    flops_stats = calculate_benchmark_stats(num_nodes, degree, num_epochs, total_time)

    print(f"\n📊 Configuration:")
    print(f"  Nodes:  {num_nodes:,}")
    print(f"  Degree: {degree}")
    print(f"  Edges:  {flops_stats['num_edges']:,}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Time:   {total_time:.2f} seconds")

    print(f"\n🔢 FLOP Counts:")
    print(f"  FLOPs per edge:              {flops_stats['flops_per_edge']}")
    print(
        f"  FLOPs per epoch (attractive): {flops_stats['attractive_flops_per_epoch']:,.0f}"
    )
    print(
        f"  Total FLOPs (attractive):     {flops_stats['total_attractive_flops']:,.0f}"
    )
    print(f"  Total TFLOPS (attractive):    {flops_stats['total_tflops']:.4f}")

    print(f"\n⚡ Throughput:")
    print(f"  TFLOPS/second: {flops_stats['tflops_per_second']:.4f}")

    # Calculate memory bandwidth
    mem_stats = calculate_memory_bandwidth(flops_stats["num_edges"], total_time)

    print(f"\n💾 Memory Bandwidth:")
    print(f"  Bytes per edge (read):    {mem_stats['bytes_read_per_edge']}")
    print(f"  Bytes per edge (write):   {mem_stats['bytes_written_per_edge']}")
    print(f"  Bytes per edge (total):   {mem_stats['bytes_per_edge']}")
    print(f"  Total data transferred:   {mem_stats['total_gb']:.2f} GB")
    print(f"  Effective bandwidth:      {mem_stats['bandwidth_gbs']:.2f} GB/s")

    # Arithmetic intensity
    ai = calculate_arithmetic_intensity(
        flops_stats["flops_per_edge"], mem_stats["bytes_per_edge"]
    )
    print(f"\n🎯 Arithmetic Intensity:")
    print(f"  {ai:.4f} FLOPs/byte")

    if ai < 1:
        print("  ⚠️  MEMORY-BOUND: This kernel is limited by memory bandwidth")
    elif ai < 10:
        print("  ⚖️  BALANCED: Mix of compute and memory bottlenecks")
    else:
        print("  ⚡ COMPUTE-BOUND: This kernel is limited by compute throughput")

    # GPU comparison
    if gpu_name:
        gpus = estimate_peak_performance()
        if gpu_name in gpus:
            gpu = gpus[gpu_name]
            print(f"\n🎮 GPU Comparison ({gpu_name}):")
            print(f"  Peak FP32:           {gpu['fp32_tflops']:.2f} TFLOPS")
            print(f"  Peak Memory BW:      {gpu['memory_bandwidth_gbs']:.2f} GB/s")

            compute_util = (flops_stats["tflops_per_second"] / gpu["fp32_tflops"]) * 100
            memory_util = (
                mem_stats["bandwidth_gbs"] / gpu["memory_bandwidth_gbs"]
            ) * 100

            print(f"  Compute utilization: {compute_util:.2f}%")
            print(f"  Memory utilization:  {memory_util:.2f}%")

            print(f"\n💡 Bottleneck Analysis:")
            if memory_util > compute_util:
                print(
                    f"  Memory-bound ({memory_util:.1f}% memory vs {compute_util:.1f}% compute)"
                )
                print(f"  Optimization: Focus on reducing memory traffic")
            else:
                print(
                    f"  Compute-bound ({compute_util:.1f}% compute vs {memory_util:.1f}% memory)"
                )
                print(f"  Optimization: Focus on increasing arithmetic intensity")

    print("\n" + "=" * 70)


# Example usage with your benchmark parameters
if __name__ == "__main__":
    print("\n🔍 Example 1: Small benchmark (5 epochs)")
    analyze_performance(
        num_nodes=20_000_000,
        degree=10,
        num_epochs=5,
        total_time=10.0,  # Hypothetical time
        gpu_name="RTX 4090",
    )

    print("\n\n🔍 Example 2: Full run (2000 epochs)")
    analyze_performance(
        num_nodes=20_000_000,
        degree=10,
        num_epochs=2000,
        total_time=4000.0,  # Hypothetical time
        gpu_name="RTX 4090",
    )

    print("\n\n🔍 Example 3: A100 GPU")
    analyze_performance(
        num_nodes=20_000_000,
        degree=10,
        num_epochs=5,
        total_time=8.0,  # A100 might be faster due to bandwidth
        gpu_name="A100 (40GB)",
    )

    print("\n📋 Available GPUs for comparison:")
    gpus = estimate_peak_performance()
    for name, specs in gpus.items():
        print(f"  - {name}")
