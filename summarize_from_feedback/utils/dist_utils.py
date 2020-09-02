import logging
import os
import socket
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, Tuple, Iterable, TypeVar, Generic

import numpy as np
import torch
import torch.distributed as dist

from summarize_from_feedback.model_layout import ModelLayout


class Comm:
    """
    Thin wrapper around a dist.Group that stores the ranks and can print when in
    verbose mode
    """

    def __init__(self, ranks, my_rank):
        ranks = list(sorted(ranks))
        self._group = dist.new_group(ranks)
        self._mpi_group = create_mpi_group(ranks)
        self.ranks = ranks
        self.size = len(ranks)
        self.my_rank = my_rank

    @property
    def my_index(self):
        return self.ranks.index(self.my_rank)

    #################################################
    ################ MPI COMMS ######################
    #################################################

    def barrier(self, name):
        self._mpi_group.barrier()

    def mpi_all_gather(self, tensor, name, validate_data_safety=True):
        """
        all_gather using MPI. Slower, but accepts a broader variety of input types
        """
        if validate_data_safety:
            validate_data_is_mpi_safe(tensor)
        return self._mpi_group.allgather(tensor)

    ####################################################################
    ########################## STANDARD COMMS ##########################
    ####################################################################

    def broadcast(self, tensor, src, name):
        self._broadcast(tensor, src, name, async_op=False)
        return tensor

    def _broadcast(self, tensor, src, name, async_op=False):
        if dist.get_backend() == "nccl":
            assert (
                tensor.is_cuda
            ), f"Bad tensor - NCCL backend only supports cuda tensors: {name}; {tensor}"

        if len(self.ranks) == 1:
            # Conform to the comm.broadcast and comm.all_reduce API, but do no work
            if async_op:
                return NoopPromise()
            else:
                return tensor

        return dist.broadcast(tensor, src, group=self._group, async_op=async_op)

    def all_reduce(self, tensor, name):
        if dist.get_backend() == "nccl":
            assert tensor.is_cuda, f"Bad tensor - NCCL backend only supports cuda tensors: {name}"

        if len(self.ranks) == 1:
            return tensor

        dist.all_reduce(tensor, group=self._group, async_op=False)
        return tensor

    def all_gather_no_backward(self, tensor, name):
        if dist.get_backend() == "nccl":
            assert tensor.is_cuda, f"Bad tensor - NCCL backend only supports cuda tensors: {name}"

        tensor_list = [
            torch.zeros(tensor.size(), dtype=tensor.dtype, device=tensor.device)
            for _ in range(self.size)
        ]
        dist.all_gather(tensor_list, tensor, group=self._group)
        return tensor_list


def setup_cuda_device_and_dist(
    backend="nccl", master_addr=None, port=29500, world_size=None, device="cuda"
) -> torch.device:
    """
    Set up the cuda device and then initialize nccl. We do these together because
    it's important that we initialize dist *after* we set the cuda device, otherwise GPU 0 will
    be responsible for all NCCL comms and will hang / OOM

    :param master_addr: The address of the master rank. Set to "127.0.0.1" to run locally.
    :param backend: One of ['nccl', 'gloo']. NCCL is ~10x faster, but often fails silently on
        inappropriate inputs, whereas gloo will often give a useful error message. We therefore
        recommend using gloo for debugging.
    :param port: Port that will be used when the master receives connection during the TCP
        initialization dance.

    :return: cuda device for this rank
    """

    # This must be imported in order to get errors from all ranks to show up
    from mpi4py import MPI

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = world_size or MPI.COMM_WORLD.Get_size()

    if device == "cuda":
        # Pin this rank to a specific GPU on the node
        local_rank = mpi_rank % int(os.environ.get("NUM_GPU", "8"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device(device)

    if dist.is_initialized():
        return device

    if master_addr is None:
        # Get the ip-address for rank 0 and broadcast it to all the ranks
        master_addr = MPI.COMM_WORLD.bcast(socket.gethostbyname(socket.gethostname()))

    os.environ["RANK"] = str(mpi_rank)
    os.environ["WORLD_SIZE"] = str(mpi_size)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(port)

    assert dist.is_available()

    if mpi_rank == 0:
        logging.info(f"All nodes will connecting to master_addr: {master_addr}")

    # It's important that we initialize dist *after* we set the cuda device, otherwise
    # GPU 0 will be responsible for all NCCL comms and will hang / OOM
    dist.init_process_group(backend=backend, init_method=f"env://")

    return device


_comm_cache: Dict[Tuple[int], "Comm"] = {}


def create_mpi_group(ranks):
    from mpi4py import MPI

    group = MPI.COMM_WORLD.group.Incl(ranks)
    return MPI.COMM_WORLD.Create_group(group)


def validate_data_is_mpi_safe(data, name="<unknown>"):
    known_safe_types = (int, float, str, bool, type(None), np.ndarray, np.generic)

    if isinstance(data, known_safe_types):
        pass
    elif isinstance(data, torch.Tensor):
        if data.is_cuda:
            raise ValueError(
                f"Data name={name} was a cuda tensor. MPI cannot handle CUDA tensors"
                f" as they result in unexpected CUDA OOMs."
            )
    elif isinstance(data, dict):
        for k, v in data.items():
            validate_data_is_mpi_safe(k)
            validate_data_is_mpi_safe(v, name=k)
    elif isinstance(data, Iterable):
        for item in data:
            validate_data_is_mpi_safe(item)
    else:
        raise ValueError(f"Data name={name} had unsupported type: {type(data)}")


T = TypeVar("T")


class Promise(ABC, Generic[T]):
    @abstractmethod
    def wait(self) -> T:
        pass


class NoopPromise(Promise[None]):
    def wait(self):
        return


@lru_cache()  # Memoize when using the same layout
def create_data_parallel_comm(layout: ModelLayout) -> Comm:
    """When using NCCL, all ranks must participate in construction of communicators. We use this
            object to instantiate the NCCL communicators correctly and provide a simplified API"""

    _my_dp_comm = None
    for other_rank in layout.ranks_in_my_replica:
        other_layout = ModelLayout(layout=layout.layout, my_rank=other_rank)
        dp_group = Comm(other_layout.dp_sibling_ranks, my_rank=other_rank)

        if other_rank == layout.my_rank:
            _my_dp_comm = dp_group

    return _my_dp_comm


@lru_cache()
def create_within_replica_comm(layout):
    """
    Create a comm for all the shards and depths within a single replica

    Note that when using NCCL, all ranks must participate in construction of communicators. We use
    this object to instantiate the NCCL communicators correctly and provide a simplified API"""

    _my_comm = None
    for sibling_rank in layout.dp_sibling_ranks:
        layout_for_sibling = ModelLayout(layout=layout.layout, my_rank=sibling_rank)
        ranks_in_replica = layout_for_sibling.ranks_in_my_replica
        within_replica_comm = Comm(ranks_in_replica, my_rank=sibling_rank)

        if sibling_rank == layout.my_rank:
            _my_comm = within_replica_comm
    return _my_comm


def create_model_parallel_comm(layout: ModelLayout):
    """When using NCCL, all ranks must participate in construction of communicators. We use this
            object to instantiate the NCCL communicators correctly and provide a simplified API"""
    _my_mp_comm = None

    # Set up model-parallel communication
    for replica_idx, ranks in enumerate(layout.layout):
        mp_group = Comm(ranks, my_rank=ranks[layout.shard_idx])
        if replica_idx == layout.replica_idx:
            _my_mp_comm = mp_group

    assert _my_mp_comm is not None
    return _my_mp_comm
