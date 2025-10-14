import torch
import time
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="UMAP-style graph optimization benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use for computation (default: auto)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=20_000_000,
        help="Number of nodes in the graph (default: 20,000,000)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=10,
        help="Average degree per node (default: 10)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for position updates (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def get_device(device_arg):
    """Select the appropriate device based on argument and availability."""
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal (MPS) device.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using NVIDIA CUDA device.")
        else:
            device = torch.device("cpu")
            print("Using CPU device.")
    else:
        device = torch.device(device_arg)
        print(f"Using {device_arg.upper()} device.")
    return device


def set_seed(seed):
    """Set random seed for reproducibility across all random number generators.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # Set deterministic algorithms for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_device(dev):
    """Synchronize device operations."""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def format_bytes(size_bytes):
    """Format bytes with appropriate unit (B, KB, MB, GB).

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted string with appropriate unit
    """
    if size_bytes < 1024:
        return f"{size_bytes:.2f} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def get_tensor_size_bytes(tensor):
    """Calculate tensor size in bytes.

    Args:
        tensor: PyTorch tensor

    Returns:
        int: Size in bytes
    """
    return tensor.element_size() * tensor.nelement()


def print_tensor_info(name, tensor):
    """Print tensor size and memory usage.

    Args:
        name: Name of the tensor
        tensor: The tensor to print info about
    """
    size_bytes = get_tensor_size_bytes(tensor)
    size_str = format_bytes(size_bytes)
    print(f"  {name}: shape={list(tensor.shape)}, size={size_str}")


def generate_graph_data(num_nodes, avg_degree, device):
    """Generate realistic graph data for benchmarking.

    Args:
        num_nodes: Number of nodes in the graph
        avg_degree: Average degree per node
        device: PyTorch device to use

    Returns:
        tuple: (node_positions, edges, edge_weights, num_edges)
    """
    print(f"Generating realistic data for {num_nodes:,} nodes...")

    node_positions = torch.rand(num_nodes, 2, device=device, dtype=torch.float32)

    # Create the source nodes
    source_nodes = torch.arange(num_nodes, device=device).repeat_interleave(
        avg_degree // 2
    )

    # Get the number of edges dynamically from the source_nodes tensor
    num_edges = len(source_nodes)
    print(f"Dynamically set NUM_EDGES to {num_edges:,}")

    # Create the destination nodes
    dest_nodes = torch.randint(0, num_nodes, (num_edges,), device=device)

    # Shuffle and combine
    perm = torch.randperm(num_edges, device=device)
    source_nodes = source_nodes[perm]
    edges = torch.stack([source_nodes, dest_nodes], dim=1)

    edge_weights = torch.rand(num_edges, 1, device=device, dtype=torch.float32)

    print("Data generation complete.")
    print("\n--- Tensor Sizes ---")
    print_tensor_info("node_positions", node_positions)
    print_tensor_info("edges", edges)
    print_tensor_info("edge_weights", edge_weights)
    total_bytes = (
        get_tensor_size_bytes(node_positions) +
        get_tensor_size_bytes(edges) +
        get_tensor_size_bytes(edge_weights)
    )
    print(f"  Total (base tensors): {format_bytes(total_bytes)}")

    return node_positions, edges, edge_weights, num_edges


def verify_node_degrees(edges, num_nodes):
    """Verify and print statistics about node degrees.

    Args:
        edges: Edge tensor
        num_nodes: Total number of nodes
    """
    print("\n--- Verifying Node Degrees ---")
    # Flatten the edge list to get a 1D tensor of all connections
    all_connections = edges.flatten()
    # Use bincount to efficiently count the occurrences of each node ID
    node_degrees = torch.bincount(all_connections, minlength=num_nodes).float()

    # Calculate statistics
    mean_degree = node_degrees.mean().cpu().item()
    median_degree = node_degrees.median().cpu().item()
    var_degree = node_degrees.var().cpu().item()
    min_degree = node_degrees.min().cpu().item()
    max_degree = node_degrees.max().cpu().item()

    print(f"Mean degree:   {mean_degree:.2f}")
    print(f"Median degree: {median_degree:.2f}")
    print(f"Variance of degree: {var_degree:.2f}")
    print(f"Min degree: {min_degree:.2f}")
    print(f"Max degree: {max_degree:.2f}\n")


def warmup_iteration(node_positions, edges, edge_weights, learning_rate, device):
    """Run a warm-up iteration to initialize GPU kernels.

    Args:
        node_positions: Node position tensor
        edges: Edge tensor
        edge_weights: Edge weight tensor
        learning_rate: Learning rate for updates
        device: PyTorch device
    """
    with torch.no_grad():
        pos_u = node_positions[edges[:, 0]]
        pos_v = node_positions[edges[:, 1]]
        delta = (pos_v - pos_u) * edge_weights * learning_rate
        update_aggregator = torch.zeros_like(node_positions)
        update_aggregator.scatter_add_(
            0, edges[:, 0].unsqueeze(1).expand_as(delta), delta
        )
        update_aggregator.scatter_add_(
            0, edges[:, 1].unsqueeze(1).expand_as(delta), -delta
        )
        node_positions += update_aggregator
        sync_device(device)


def run_benchmark(node_positions, edges, edge_weights, learning_rate, num_iterations, device):
    """Run the main benchmark loop.

    Args:
        node_positions: Node position tensor
        edges: Edge tensor
        edge_weights: Edge weight tensor
        learning_rate: Learning rate for updates
        num_iterations: Number of iterations to run
        device: PyTorch device

    Returns:
        tuple: (total_duration, timings_dict)
    """
    timings = {
        "gather": 0.0,
        "compute": 0.0,
        "scatter_add": 0.0,
        "apply_updates": 0.0,
    }

    print(f"Starting benchmark for {num_iterations} iterations...")

    total_start_time = time.time()
    for i in range(num_iterations):
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{num_iterations}")

        with torch.no_grad():
            # 1. GATHER
            sync_device(device)
            op_start_time = time.time()
            pos_u = node_positions[edges[:, 0]]
            pos_v = node_positions[edges[:, 1]]
            sync_device(device)
            timings["gather"] += time.time() - op_start_time

            # Print tensor sizes on first iteration
            if i == 0:
                print("\n--- Intermediate Tensor Sizes (first iteration) ---")
                print_tensor_info("pos_u (gathered)", pos_u)
                print_tensor_info("pos_v (gathered)", pos_v)

            # 2. COMPUTE
            sync_device(device)
            op_start_time = time.time()
            delta = (pos_v - pos_u) * edge_weights * learning_rate
            sync_device(device)
            timings["compute"] += time.time() - op_start_time

            if i == 0:
                print_tensor_info("delta (computed)", delta)

            # 3. SCATTER-ADD
            sync_device(device)
            op_start_time = time.time()
            update_aggregator = torch.zeros_like(node_positions)
            update_aggregator.scatter_add_(
                0, edges[:, 0].unsqueeze(1).expand_as(delta), delta
            )
            update_aggregator.scatter_add_(
                0, edges[:, 1].unsqueeze(1).expand_as(delta), -delta
            )
            sync_device(device)
            timings["scatter_add"] += time.time() - op_start_time

            if i == 0:
                print_tensor_info("update_aggregator (scattered)", update_aggregator)
                # Calculate peak memory usage for intermediate tensors
                intermediate_bytes = (
                    get_tensor_size_bytes(pos_u) +
                    get_tensor_size_bytes(pos_v) +
                    get_tensor_size_bytes(delta) +
                    get_tensor_size_bytes(update_aggregator)
                )
                print(f"  Peak intermediate memory: {format_bytes(intermediate_bytes)}\n")

            # 4. APPLY UPDATES
            sync_device(device)
            op_start_time = time.time()
            node_positions += update_aggregator
            sync_device(device)
            timings["apply_updates"] += time.time() - op_start_time

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    return total_duration, timings


def compute_checksum(node_positions):
    """Compute a checksum of the final node positions for verification.

    Args:
        node_positions: Final node position tensor

    Returns:
        dict: Dictionary containing checksum metrics
    """
    checksum = {
        "sum": node_positions.sum().cpu().item(),
        "mean": node_positions.mean().cpu().item(),
        "std": node_positions.std().cpu().item(),
        "min": node_positions.min().cpu().item(),
        "max": node_positions.max().cpu().item(),
    }
    return checksum


def print_results(total_duration, timings, num_iterations, checksum):
    """Print benchmark results.

    Args:
        total_duration: Total time taken for all iterations
        timings: Dictionary of operation timings
        num_iterations: Number of iterations run
        checksum: Dictionary of checksum metrics
    """
    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_iterations} iterations: {total_duration:.4f} seconds")
    print(f"Average time per iteration: {total_duration / num_iterations:.4f} seconds\n")

    print("--- Aggregated Time Per Operation ---")
    for op, t in timings.items():
        percentage = (t / total_duration) * 100
        print(f"{op:<15}: {t:.4f} seconds ({percentage:.2f}%)")

    print("\n--- Result Checksum (for verification) ---")
    print(f"Sum:  {checksum['sum']:.10f}")
    print(f"Mean: {checksum['mean']:.10f}")
    print(f"Std:  {checksum['std']:.10f}")
    print(f"Min:  {checksum['min']:.10f}")
    print(f"Max:  {checksum['max']:.10f}")

    # Estimate time for 2000 iterations
    avg_time_per_iteration = total_duration / num_iterations
    estimated_time_2000 = avg_time_per_iteration * 2000
    estimated_hours = estimated_time_2000 / 3600
    print(f"\n--- Estimated Time for 2000 Iterations ---")
    print(f"Estimated time: {estimated_hours:.2f} hours")


def main():
    """Main entry point."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = get_device(args.device)

    node_positions, edges, edge_weights, num_edges = generate_graph_data(
        args.num_nodes, args.degree, device
    )

    verify_node_degrees(edges, args.num_nodes)

    warmup_iteration(node_positions, edges, edge_weights, args.learning_rate, device)

    total_duration, timings = run_benchmark(
        node_positions, edges, edge_weights, args.learning_rate, args.iterations, device
    )

    checksum = compute_checksum(node_positions)

    print_results(total_duration, timings, args.iterations, checksum)


if __name__ == "__main__":
    main()
