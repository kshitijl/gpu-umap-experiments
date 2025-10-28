import os

import numpy as np
import pyarrow as pa
import scipy.sparse
from numpy.typing import NDArray

def save(directory: str, filename: str, array: NDArray):
    path = os.path.join(directory, filename)
    print(f"saving {path} {array.shape} [{array.dtype}]")
    np.save(path, array)


def load(directory: str, filename: str) -> NDArray:
    path = os.path.join(directory, filename)
    array = np.load(path, mmap_mode="r")
    print(f"loaded {path} {array.shape} [{array.dtype}]")
    return array

class ArrowReader:
    def __init__(self, file_path, schema: pa.Schema | None = None):
        self.file_path = file_path
        self.mmap_file = None
        self.reader = None
        self.schema = schema

    def __enter__(self):
        self.mmap_file = pa.memory_map(self.file_path, "r")
        self.reader = pa.ipc.RecordBatchFileReader(self.mmap_file)
        if self.schema is None:
            self.schema = self.reader.schema
        else:
            assert self.schema.equals(self.reader.schema)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mmap_file:
            self.mmap_file.close()

    def get_columns_as_numpy(self, batch_index=0):
        """Get all columns as numpy arrays (zero-copy)."""
        assert self.reader is not None
        batch = self.reader.get_record_batch(batch_index)
        return [col.to_numpy(zero_copy_only=True) for col in batch.columns]

node_schema = pa.schema(
    [
        pa.field("id", pa.uint32()),
        pa.field("incoming_degree", pa.uint32()),
        pa.field("outgoing_degree", pa.uint32()),
    ]
)

class NodeReader(ArrowReader):
    def __init__(self, file_path: str):
        super().__init__(file_path, node_schema)

    def get_nodes(
        self, batch_index=0
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]:
        """Get node data as typed numpy arrays (zero-copy).

        Returns:
            tuple: (ids, incoming_degrees, outgoing_degrees )
        """
        columns = self.get_columns_as_numpy(batch_index)
        if len(columns) != 3:
            raise ValueError(f"Expected 3 columns for node data, got {len(columns)}")

        ids: NDArray[np.uint32] = columns[0]
        incoming_degrees: NDArray[np.uint32] = columns[1]
        outgoing_degrees: NDArray[np.uint32] = columns[2]
        return (ids, incoming_degrees, outgoing_degrees)

edge_schema = pa.schema(
    [
        pa.field("weight", pa.float32()),
        pa.field("source", pa.int32()),
        pa.field("target", pa.int32()),
    ]
)

class EdgeReader(ArrowReader):
    def __init__(self, file_path: str):
        super().__init__(file_path, edge_schema)

    def get_edges(
        self, batch_index=0
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
        """Get edge data as typed numpy arrays (zero-copy).

        Returns:
            tuple: (weights, sources, targets)
        """
        columns = self.get_columns_as_numpy(batch_index)
        if len(columns) != 3:
            raise ValueError(f"Expected 3 columns for edge data, got {len(columns)}")

        weights: NDArray[np.float32] = columns[0]
        sources: NDArray[np.int32] = columns[1]
        targets: NDArray[np.int32] = columns[2]
        return (weights, sources, targets)

def read_coo_array(file_path: str, n_nodes: int | None = None, copy=False):
    with EdgeReader(file_path) as reader:
        (weights, sources, targets) = reader.get_edges()

    if copy:
        weights, sources, targets = weights.copy(), sources.copy(), targets.copy()

    if n_nodes is None:
        return scipy.sparse.coo_array((weights, (sources, targets)))
    else:
        return scipy.sparse.coo_array(
            (weights, (sources, targets)), shape=(n_nodes, n_nodes)
        )
