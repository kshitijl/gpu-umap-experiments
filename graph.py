import torch
import time

# --- Configuration ---
NUM_NODES = 20_000_000
AVG_DEGREE = 10
NUM_ITERATIONS = 5
LEARNING_RATE = 0.001

# --- Device Selection ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS) device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")


# --- Helper function for synchronization ---
def sync_device(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# --- Corrected Realistic Data Generation ---
print(f"Generating realistic data for {NUM_NODES:,} nodes...")

node_positions = torch.rand(NUM_NODES, 2, device=device, dtype=torch.float32)


# 1. Create the source nodes.
source_nodes = torch.arange(NUM_NODES, device=device).repeat_interleave(AVG_DEGREE // 2)

# --- THIS IS THE KEY CHANGE ---
# Get the number of edges dynamically from the source_nodes tensor
# instead of using a hardcoded variable.
NUM_EDGES = len(source_nodes)
print(f"Dynamically set NUM_EDGES to {NUM_EDGES:,}")

# 2. Create the destination nodes.
dest_nodes = torch.randint(0, NUM_NODES, (NUM_EDGES,), device=device)

# 3. Shuffle and combine
perm = torch.randperm(NUM_EDGES, device=device)
source_nodes = source_nodes[perm]
edges = torch.stack([source_nodes, dest_nodes], dim=1)

edge_weights = torch.rand(NUM_EDGES, 1, device=device, dtype=torch.float32)

print("Data generation complete.")


# --- NEW: Degree Calculation Check ---
print("\n--- Verifying Node Degrees ---")
# Flatten the edge list to get a 1D tensor of all connections
all_connections = edges.flatten()
# Use bincount to efficiently count the occurrences of each node ID
# This gives us the total degree (in-degree + out-degree) for each node.
node_degrees = torch.bincount(all_connections, minlength=NUM_NODES).float()

# Calculate statistics
mean_degree = node_degrees.mean().cpu().item()
median_degree = node_degrees.median().cpu().item()
var_degree = node_degrees.var().cpu().item()
min_degree = node_degrees.min().cpu().item()
max_degree = node_degrees.max().cpu().item()

print(f"Mean degree:   {mean_degree:.2f}")
print(f"Median degree: {median_degree:.2f}")
print(f"Variance of degree: {var_degree:.2f}\n")
print(f"Min degree: {min_degree:.2f}\n")
print(f"Max degree: {max_degree:.2f}\n")

# --- Profiling Setup ---
timings = {
    "gather": 0.0,
    "compute": 0.0,
    "scatter_add": 0.0,
    "apply_updates": 0.0,
}

print(f"Starting benchmark for {NUM_ITERATIONS} iterations...")
# --- Warm-up Iteration ---
with torch.no_grad():
    pos_u = node_positions[edges[:, 0]]
    pos_v = node_positions[edges[:, 1]]
    delta = (pos_v - pos_u) * edge_weights * LEARNING_RATE
    update_aggregator = torch.zeros_like(node_positions)
    update_aggregator.scatter_add_(0, edges[:, 0].unsqueeze(1).expand_as(delta), delta)
    update_aggregator.scatter_add_(0, edges[:, 1].unsqueeze(1).expand_as(delta), -delta)
    node_positions += update_aggregator
    sync_device(device)

# --- Main Benchmark Loop ---
total_start_time = time.time()
for i in range(NUM_ITERATIONS):
    if (i + 1) % 10 == 0:
        print(f"Iteration {i + 1}/{NUM_ITERATIONS}")

    with torch.no_grad():
        # --- 1. GATHER ---
        sync_device(device)
        op_start_time = time.time()
        pos_u = node_positions[edges[:, 0]]
        pos_v = node_positions[edges[:, 1]]
        sync_device(device)
        timings["gather"] += time.time() - op_start_time

        # --- 2. COMPUTE ---
        sync_device(device)
        op_start_time = time.time()
        delta = (pos_v - pos_u) * edge_weights * LEARNING_RATE
        sync_device(device)
        timings["compute"] += time.time() - op_start_time

        # --- 3. SCATTER-ADD ---
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

        # --- 4. APPLY UPDATES ---
        sync_device(device)
        op_start_time = time.time()
        node_positions += update_aggregator
        sync_device(device)
        timings["apply_updates"] += time.time() - op_start_time

total_end_time = time.time()
total_duration = total_end_time - total_start_time

# --- Results ---
print("\n--- Benchmark Results ---")
print(f"Total time for {NUM_ITERATIONS} iterations: {total_duration:.4f} seconds")
print(f"Average time per iteration: {total_duration / NUM_ITERATIONS:.4f} seconds\n")

print("--- Aggregated Time Per Operation ---")
for op, t in timings.items():
    percentage = (t / total_duration) * 100
    print(f"{op:<15}: {t:.4f} seconds ({percentage:.2f}%)")
