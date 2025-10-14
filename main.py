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


def sync_device(dev):
    """Synchronize device operations."""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


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

            # 2. COMPUTE
            sync_device(device)
            op_start_time = time.time()
            delta = (pos_v - pos_u) * edge_weights * learning_rate
            sync_device(device)
            timings["compute"] += time.time() - op_start_time

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

            # 4. APPLY UPDATES
            sync_device(device)
            op_start_time = time.time()
            node_positions += update_aggregator
            sync_device(device)
            timings["apply_updates"] += time.time() - op_start_time

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    return total_duration, timings


def print_results(total_duration, timings, num_iterations):
    """Print benchmark results.

    Args:
        total_duration: Total time taken for all iterations
        timings: Dictionary of operation timings
        num_iterations: Number of iterations run
    """
    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_iterations} iterations: {total_duration:.4f} seconds")
    print(f"Average time per iteration: {total_duration / num_iterations:.4f} seconds\n")

    print("--- Aggregated Time Per Operation ---")
    for op, t in timings.items():
        percentage = (t / total_duration) * 100
        print(f"{op:<15}: {t:.4f} seconds ({percentage:.2f}%)")


def main():
    """Main entry point."""
    args = parse_args()

    device = get_device(args.device)

    node_positions, edges, edge_weights, num_edges = generate_graph_data(
        args.num_nodes, args.degree, device
    )

    verify_node_degrees(edges, args.num_nodes)

    warmup_iteration(node_positions, edges, edge_weights, args.learning_rate, device)

    total_duration, timings = run_benchmark(
        node_positions, edges, edge_weights, args.learning_rate, args.iterations, device
    )

    print_results(total_duration, timings, args.iterations)


if __name__ == "__main__":
    main()
