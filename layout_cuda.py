#!/usr/bin/env python3
"""
CuPy Example with Custom CUDA Kernel
Demonstrates writing raw CUDA C++ code as a string
"""

import cupy as cp
import numpy as np
import scipy.sparse
from tqdm import tqdm
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from utils import read_coo_array

with open('layout_cuda_kernel.cpp', 'r') as file:
    layout_cuda_kernel = file.read()

# initialize cuRAND states
init_curand = cp.RawKernel(layout_cuda_kernel, 'init_curand_states')

# Helper to get curandState size
get_sizeof_curand_state = cp.RawKernel(layout_cuda_kernel, 'get_sizeof_curand_state')

positive_sample_pass = cp.RawKernel(layout_cuda_kernel, 'positive_sample_pass')

def allocate_curand_states(num_states):
    """Allocate memory for curandState array"""
    size_result = cp.zeros(1, dtype=cp.uint64)
    get_sizeof_curand_state((1,), (1,), (size_result,))
    cp.cuda.Stream.null.synchronize()
    state_size = int(size_result[0])
    return cp.empty(num_states * state_size, dtype=cp.uint8), state_size

def reorder_graph(G: scipy.sparse.coo_array) -> tuple[scipy.sparse.coo_array, NDArray[np.int32]]:
    G_csr = G.tocsr()
    perm = reverse_cuthill_mckee(G_csr, symmetric_mode=True)
    reordered_graph = G_csr[perm, :][:, perm]
    # reordered_positions = positions[perm]

    G = reordered_graph.tocoo()
    mask = G.row <= G.col  # Upper triangle only
    G.row, G.col, G.data = (
        G.row[mask],
        G.col[mask],
        G.data[mask],
    )

    # Sort
    sort_idx = np.lexsort((G.col, G.row))
    G.row = G.row[sort_idx]
    G.col = G.col[sort_idx]
    G.data = G.data[sort_idx]

    return G, perm

def initialize_layout(
    X: NDArray[np.float32],
    n_components: int = 2,
    random_state: np.random.RandomState | None = None,
) -> NDArray[np.float32]:
    if random_state is None:
        random_state = np.random.RandomState(42)

    pca = PCA(n_components=n_components, random_state=random_state)
    positions = pca.fit_transform(X).astype(np.float32)
    positions = noisy_scale_coords(
        positions, random_state, max_coord=10, noise=0.0001
    )

    return positions

def layout(sources, targets, weights, initial_pos, n_epochs=500):
    assert sources.shape == targets.shape
    n_edges = len(sources)
    n_nodes, dim = initial_pos.shape
    assert dim == 2

    pos_x = cp.asarray(initial_pos[:, 0], dtype=cp.float32)
    pos_y = cp.asarray(initial_pos[:, 1], dtype=cp.float32)
    
    print(f"Sampling {n_edges:,} edges using custom kernel...")

    # Configure grid and block dimensions
    threads_per_block = 32
    blocks_per_grid = (n_edges + threads_per_block - 1) // threads_per_block
    # threads_per_block = 4
    # blocks_per_grid = 1
    

    total_threads = blocks_per_grid * threads_per_block
    
    print(f"Grid: {blocks_per_grid} blocks × {threads_per_block} threads")

    grad_x = cp.zeros((n_nodes,), dtype=cp.float32)
    grad_y = cp.zeros((n_nodes,), dtype=cp.float32)

    a = cp.float32(1.5769434605754993)
    b = cp.float32(0.8950608781680347)
    gamma = cp.float32(1.0)
    alpha = cp.float32(1.0)

    curand_states, state_size = allocate_curand_states(total_threads)
    print(curand_states, state_size)

    init_curand((blocks_per_grid,), (threads_per_block,), (curand_states, 0))

    print("sources", sources)
    print("targets", targets)
    
    initial_alpha = alpha

    for n in tqdm(range(n_epochs)):
        cp.cuda.runtime.deviceSynchronize()
    
        # Launch the kernel
        positive_sample_pass(
            (blocks_per_grid,),      # grid dimensions
            (threads_per_block,),     # block dimensions
            (
                pos_x, pos_y, grad_x, grad_y, n_nodes,
                sources, targets, weights, n_edges,
                a, b, gamma, alpha,
                curand_states
            ),
            shared_mem=64,
        )

        pos_x += grad_x
        pos_y += grad_y
        grad_x.fill(0)
        grad_y.fill(0)

        p = float(n) / float(n_epochs)
        alpha = initial_alpha * (1.0 - p)

    return np.column_stack((cp.asnumpy(pos_x), cp.asnumpy(pos_y)))


def main():
    print("CuPy Custom Kernel Example")
    print(f"GPU ID: {cp.cuda.Device().id}")

    fss = read_coo_array("fss-1e5.arrow")
    assert fss.shape[0] == fss.shape[1]

    sources = cp.asarray(fss.row, dtype=cp.int32)
    targets = cp.asarray(fss.col, dtype=cp.int32)
    weights = cp.asarray(fss.data, dtype=cp.float32)
    assert sources.shape == targets.shape

    # sources = cp.asarray(np.load("sources-1e3.npy"), dtype=cp.int32)
    # targets = cp.asarray(np.load("targets-1e3.npy"), dtype=cp.int32)
    # sources = cp.asarray(np.load("sources-1e5.npy"), dtype=cp.int32)
    # targets = cp.asarray(np.load("targets-1e5.npy"), dtype=cp.int32)

    n_nodes = fss.shape[0]
    n_edges = len(weights)

    print(f"n_nodes: {n_nodes}, n_edges: {n_edges}")
    print('sources', sources)
    print('targets', targets)

    assert np.all(sources < n_nodes)
    assert np.all(targets < n_nodes)

    rand = np.random.RandomState(42)
    initial_pos = np.column_stack((rand.random(n_nodes), rand.random(n_nodes)))
    initial_pos *= 10
    final_pos = layout(sources, targets, weights, initial_pos)

    print("final pos", final_pos[:10])


if __name__ == "__main__":
    main()