import torch
import time
import argparse
from tqdm import tqdm


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
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for attractive forces (default: 0.001)",
    )
    parser.add_argument(
        "--negative-learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate for repulsive forces (default: 0.0001)",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=5,
        help="Number of negative samples per epoch (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use-int64",
        action="store_true",
        help="Use int64 instead of int32 for edge indices (default: int32)",
    )
    parser.add_argument(
        "--use-segment-reduce",
        action="store_true",
        help="Use segment_reduce instead of scatter_add (requires sorting)",
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
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


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


def generate_graph_data(num_nodes, avg_degree, device, use_int64=False):
    """Generate realistic graph data for benchmarking.

    Args:
        num_nodes: Number of nodes in the graph
        avg_degree: Average degree per node
        device: PyTorch device to use
        use_int64: If True, use int64 for edge indices instead of int32

    Returns:
        tuple: (node_positions, edges, edge_weights, num_edges)
    """
    print(f"Generating realistic data for {num_nodes:,} nodes...")

    # Determine index dtype
    index_dtype = torch.int64 if use_int64 else torch.int32
    dtype_name = "int64" if use_int64 else "int32"
    print(f"Using {dtype_name} for edge indices")

    # Check if int32 can represent num_nodes
    if not use_int64 and num_nodes > 2147483647:
        raise ValueError(
            f"Cannot use int32 for {num_nodes:,} nodes (max: 2,147,483,647). "
            "Use int64 (add --use-int64 flag)"
        )

    node_positions = torch.rand(num_nodes, 2, device=device, dtype=torch.float32)

    # Create the source nodes
    source_nodes = torch.arange(
        num_nodes, device=device, dtype=index_dtype
    ).repeat_interleave(avg_degree // 2)

    # Get the number of edges dynamically from the source_nodes tensor
    num_edges = len(source_nodes)
    print(f"Dynamically set NUM_EDGES to {num_edges:,}")

    # Create the destination nodes
    dest_nodes = torch.randint(
        0, num_nodes, (num_edges,), device=device, dtype=index_dtype
    )

    # Shuffle and combine
    perm = torch.randperm(num_edges, device=device, dtype=index_dtype)
    source_nodes = source_nodes[perm]
    edges = torch.stack([source_nodes, dest_nodes], dim=1)

    edge_weights = torch.rand(num_edges, 1, device=device, dtype=torch.float32)

    print("Data generation complete.")
    print("\n--- Tensor Sizes ---")
    print_tensor_info("node_positions", node_positions)
    print_tensor_info("edges", edges)
    print_tensor_info("edge_weights", edge_weights)
    total_bytes = (
        get_tensor_size_bytes(node_positions)
        + get_tensor_size_bytes(edges)
        + get_tensor_size_bytes(edge_weights)
    )
    print(f"  Total (base tensors): {format_bytes(total_bytes)}")

    return node_positions, edges, edge_weights, num_edges


def apply_updates_segment_reduce(
    node_positions, indices_src, indices_dst, deltas, num_nodes, device
):
    """Apply updates using segment_reduce (requires sorting).

    Args:
        node_positions: Node position tensor to update
        indices_src: Source node indices
        indices_dst: Destination node indices
        deltas: Delta values to apply
        num_nodes: Total number of nodes
        device: PyTorch device

    Returns:
        dict: Timing breakdown
    """
    timings = {
        "concat": 0.0,
        "sort": 0.0,
        "reduce": 0.0,
    }

    with torch.no_grad():
        # 1. CONCATENATE source and dest
        sync_device(device)
        op_start_time = time.time()
        all_indices = torch.cat([indices_src, indices_dst])
        all_deltas = torch.cat([deltas, -deltas], dim=0)
        sync_device(device)
        timings["concat"] += time.time() - op_start_time

        # 2. SORT by node index
        sync_device(device)
        op_start_time = time.time()
        sorted_indices, perm = torch.sort(all_indices)
        sorted_deltas = all_deltas[perm]
        sync_device(device)
        timings["sort"] += time.time() - op_start_time

        # 3. REDUCE using segment_reduce
        sync_device(device)
        op_start_time = time.time()
        # Compute segment lengths (how many updates per node)
        segment_lengths = torch.bincount(sorted_indices, minlength=num_nodes)
        # segment_reduce sums all deltas for each node
        node_updates = torch.segment_reduce(
            data=sorted_deltas, reduce="sum", lengths=segment_lengths
        )
        # Apply updates
        node_positions += node_updates
        sync_device(device)
        timings["reduce"] += time.time() - op_start_time

    return timings


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


def apply_repulsive_forces(
    node_positions,
    num_nodes,
    num_pairs,
    negative_learning_rate,
    device,
    index_dtype,
    use_segment_reduce=False,
):
    """Apply repulsive forces to random pairs of nodes.

    Args:
        node_positions: Node position tensor
        num_nodes: Total number of nodes
        num_pairs: Number of random pairs to sample
        negative_learning_rate: Learning rate for repulsive forces
        device: PyTorch device
        index_dtype: Data type for indices (int32 or int64)
        use_segment_reduce: If True, use segment_reduce instead of scatter_add

    Returns:
        dict: Timing breakdown for the operation
    """
    if use_segment_reduce:
        timings = {
            "sample": 0.0,
            "gather": 0.0,
            "compute": 0.0,
            "concat": 0.0,
            "sort": 0.0,
            "reduce": 0.0,
        }
    else:
        timings = {
            "sample": 0.0,
            "gather": 0.0,
            "compute": 0.0,
            "scatter_add": 0.0,
        }

    with torch.no_grad():
        # 1. SAMPLE random pairs
        sync_device(device)
        op_start_time = time.time()
        random_pairs = torch.randint(
            0, num_nodes, (num_pairs, 2), device=device, dtype=index_dtype
        )
        sync_device(device)
        timings["sample"] += time.time() - op_start_time

        # 2. GATHER
        sync_device(device)
        op_start_time = time.time()
        pos_u = node_positions[random_pairs[:, 0]]
        pos_v = node_positions[random_pairs[:, 1]]
        sync_device(device)
        timings["gather"] += time.time() - op_start_time

        # 3. COMPUTE (repulsive delta - push apart)
        sync_device(device)
        op_start_time = time.time()
        delta = (pos_v - pos_u) * negative_learning_rate
        sync_device(device)
        timings["compute"] += time.time() - op_start_time

        # 4. APPLY UPDATES
        if use_segment_reduce:
            # Use segment_reduce approach (note: delta sign flip for repulsion)
            update_timings = apply_updates_segment_reduce(
                node_positions,
                random_pairs[:, 0],
                random_pairs[:, 1],
                -delta,  # Negative delta for source nodes
                num_nodes,
                device,
            )
            timings["concat"] += update_timings["concat"]
            timings["sort"] += update_timings["sort"]
            timings["reduce"] += update_timings["reduce"]
        else:
            # SCATTER-ADD (note: signs are flipped to push apart)
            # Update node_positions directly without intermediate tensor
            sync_device(device)
            op_start_time = time.time()
            node_positions.scatter_add_(
                0, random_pairs[:, 0].unsqueeze(1).expand_as(delta), -delta
            )
            node_positions.scatter_add_(
                0, random_pairs[:, 1].unsqueeze(1).expand_as(delta), delta
            )
            sync_device(device)
            timings["scatter_add"] += time.time() - op_start_time

    return timings


def warmup_iteration(
    node_positions,
    edges,
    edge_weights,
    learning_rate,
    device,
    num_nodes=None,
    use_segment_reduce=False,
):
    """Run a warm-up iteration to initialize GPU kernels.

    Args:
        node_positions: Node position tensor
        edges: Edge tensor
        edge_weights: Edge weight tensor
        learning_rate: Learning rate for updates
        device: PyTorch device
        num_nodes: Total number of nodes (required if use_segment_reduce=True)
        use_segment_reduce: If True, use segment_reduce instead of scatter_add
    """
    with torch.no_grad():
        pos_u = node_positions[edges[:, 0]]
        pos_v = node_positions[edges[:, 1]]
        delta = (pos_v - pos_u) * edge_weights * learning_rate

        if use_segment_reduce:
            # Use segment_reduce approach
            apply_updates_segment_reduce(
                node_positions, edges[:, 0], edges[:, 1], delta, num_nodes, device
            )
        else:
            # Update directly without intermediate tensor
            node_positions.scatter_add_(
                0, edges[:, 0].unsqueeze(1).expand_as(delta), delta
            )
            node_positions.scatter_add_(
                0, edges[:, 1].unsqueeze(1).expand_as(delta), -delta
            )
        sync_device(device)


def run_benchmark(
    node_positions,
    edges,
    edge_weights,
    learning_rate,
    negative_learning_rate,
    num_negatives,
    num_epochs,
    device,
    num_nodes,
    num_edges,
    index_dtype,
    use_segment_reduce=False,
):
    """Run the main benchmark loop.

    Args:
        node_positions: Node position tensor
        edges: Edge tensor
        edge_weights: Edge weight tensor
        learning_rate: Learning rate for attractive forces
        negative_learning_rate: Learning rate for repulsive forces
        num_negatives: Number of negative samples per epoch
        num_epochs: Number of epochs to run
        device: PyTorch device
        num_nodes: Total number of nodes
        num_edges: Number of edges
        index_dtype: Data type for indices
        use_segment_reduce: If True, use segment_reduce instead of scatter_add

    Returns:
        tuple: (total_duration, timings_dict)
    """
    if use_segment_reduce:
        timings = {
            "attractive": {
                "gather": 0.0,
                "compute": 0.0,
                "concat": 0.0,
                "sort": 0.0,
                "reduce": 0.0,
            },
            "repulsive": {
                "sample": 0.0,
                "gather": 0.0,
                "compute": 0.0,
                "concat": 0.0,
                "sort": 0.0,
                "reduce": 0.0,
            },
        }
    else:
        timings = {
            "attractive": {
                "gather": 0.0,
                "compute": 0.0,
                "scatter_add": 0.0,
            },
            "repulsive": {
                "sample": 0.0,
                "gather": 0.0,
                "compute": 0.0,
                "scatter_add": 0.0,
            },
        }

    print(f"Starting benchmark for {num_epochs} epochs...")
    print(f"Each epoch: 1 attractive pass + {num_negatives} repulsive passes")

    total_start_time = time.time()
    for i in tqdm(range(num_epochs), desc="Benchmark", unit="epoch"):
        # ATTRACTIVE PHASE: Process all edges (pull connected nodes together)
        with torch.no_grad():
            # 1. GATHER
            sync_device(device)
            op_start_time = time.time()
            pos_u = node_positions[edges[:, 0]]
            pos_v = node_positions[edges[:, 1]]
            sync_device(device)
            timings["attractive"]["gather"] += time.time() - op_start_time

            # Print tensor sizes on first epoch
            if i == 0:
                print("\n--- Attractive Phase Tensor Sizes (first epoch) ---")
                print_tensor_info("pos_u (gathered)", pos_u)
                print_tensor_info("pos_v (gathered)", pos_v)

            # 2. COMPUTE
            sync_device(device)
            op_start_time = time.time()
            delta = (pos_v - pos_u) * edge_weights * learning_rate
            sync_device(device)
            timings["attractive"]["compute"] += time.time() - op_start_time

            if i == 0:
                print_tensor_info("delta (computed)", delta)

            # 3. APPLY UPDATES
            if use_segment_reduce:
                # Use segment_reduce approach
                update_timings = apply_updates_segment_reduce(
                    node_positions, edges[:, 0], edges[:, 1], delta, num_nodes, device
                )
                timings["attractive"]["concat"] += update_timings["concat"]
                timings["attractive"]["sort"] += update_timings["sort"]
                timings["attractive"]["reduce"] += update_timings["reduce"]
            else:
                # SCATTER-ADD
                # Update node_positions directly without intermediate tensor
                sync_device(device)
                op_start_time = time.time()
                node_positions.scatter_add_(
                    0, edges[:, 0].unsqueeze(1).expand_as(delta), delta
                )
                node_positions.scatter_add_(
                    0, edges[:, 1].unsqueeze(1).expand_as(delta), -delta
                )
                sync_device(device)
                timings["attractive"]["scatter_add"] += time.time() - op_start_time

            if i == 0:
                # Calculate peak memory usage for intermediate tensors
                intermediate_bytes = (
                    get_tensor_size_bytes(pos_u)
                    + get_tensor_size_bytes(pos_v)
                    + get_tensor_size_bytes(delta)
                )
                print(
                    f"  Peak intermediate memory (attractive): {format_bytes(intermediate_bytes)}\n"
                )

        # REPULSIVE PHASE: Process random pairs (push random nodes apart)
        for neg_iter in range(num_negatives):
            neg_timings = apply_repulsive_forces(
                node_positions,
                num_nodes,
                num_edges,
                negative_learning_rate,
                device,
                index_dtype,
                use_segment_reduce,
            )
            # Accumulate timings
            for key in neg_timings:
                timings["repulsive"][key] += neg_timings[key]

            # Print info on first repulsive pass of first epoch
            if i == 0 and neg_iter == 0:
                print("--- Repulsive Phase ---")
                print(f"Sampling {num_edges:,} random pairs per repulsive pass")
                print(
                    f"Total repulsive samples per epoch: {num_edges * num_negatives:,}\n"
                )

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


def print_results(total_duration, timings, num_epochs, checksum):
    """Print benchmark results.

    Args:
        total_duration: Total time taken for all epochs
        timings: Dictionary of operation timings (nested dict with attractive/repulsive)
        num_epochs: Number of epochs run
        checksum: Dictionary of checksum metrics
    """
    print("\n--- Benchmark Results ---")
    print(f"Total time for {num_epochs} epochs: {total_duration:.4f} seconds")
    print(f"Average time per epoch: {total_duration / num_epochs:.4f} seconds\n")

    # Calculate totals for attractive and repulsive phases
    attractive_total = sum(timings["attractive"].values())
    repulsive_total = sum(timings["repulsive"].values())

    print("--- Phase Breakdown ---")
    attractive_pct = (attractive_total / total_duration) * 100
    repulsive_pct = (repulsive_total / total_duration) * 100
    print(f"Attractive phase: {attractive_total:.4f} seconds ({attractive_pct:.2f}%)")
    print(f"Repulsive phase:  {repulsive_total:.4f} seconds ({repulsive_pct:.2f}%)\n")

    print("--- Attractive Phase Operations ---")
    for op, t in timings["attractive"].items():
        percentage = (t / attractive_total) * 100 if attractive_total > 0 else 0
        print(f"{op:<15}: {t:.4f} seconds ({percentage:.2f}%)")

    print("\n--- Repulsive Phase Operations ---")
    for op, t in timings["repulsive"].items():
        percentage = (t / repulsive_total) * 100 if repulsive_total > 0 else 0
        print(f"{op:<15}: {t:.4f} seconds ({percentage:.2f}%)")

    print("\n--- Result Checksum (for verification) ---")
    print(f"Sum:  {checksum['sum']:.10f}")
    print(f"Mean: {checksum['mean']:.10f}")
    print(f"Std:  {checksum['std']:.10f}")
    print(f"Min:  {checksum['min']:.10f}")
    print(f"Max:  {checksum['max']:.10f}")

    # Estimate time for 2000 epochs
    avg_time_per_epoch = total_duration / num_epochs
    estimated_time_2000 = avg_time_per_epoch * 2000
    estimated_hours = estimated_time_2000 / 3600
    print(f"\n--- Estimated Time for 2000 Epochs ---")
    print(f"Estimated time: {estimated_hours:.2f} hours")


def main():
    """Main entry point."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    device = get_device(args.device)

    # Determine index dtype
    index_dtype = torch.int64 if args.use_int64 else torch.int32

    node_positions, edges, edge_weights, num_edges = generate_graph_data(
        args.num_nodes, args.degree, device, args.use_int64
    )

    verify_node_degrees(edges, args.num_nodes)

    warmup_iteration(
        node_positions,
        edges,
        edge_weights,
        args.learning_rate,
        device,
        args.num_nodes,
        args.use_segment_reduce,
    )

    total_duration, timings = run_benchmark(
        node_positions,
        edges,
        edge_weights,
        args.learning_rate,
        args.negative_learning_rate,
        args.num_negatives,
        args.epochs,
        device,
        args.num_nodes,
        num_edges,
        index_dtype,
        args.use_segment_reduce,
    )

    checksum = compute_checksum(node_positions)

    print_results(total_duration, timings, args.epochs, checksum)


if __name__ == "__main__":
    main()
