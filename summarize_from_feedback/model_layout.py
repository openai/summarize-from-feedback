import math

import numpy as np


class ModelLayout:
    """Holds the structure of the model and the current rank's position within it"""

    @classmethod
    def standard(cls, *, total_gpus, my_rank, n_shards=1):
        assert my_rank < total_gpus, f"Bad rank {my_rank} for total_gpus = {total_gpus}"

        ranks = np.arange(0, total_gpus)

        gpus_per_replica = n_shards
        assert (
            total_gpus % gpus_per_replica == 0
        ), f"Total GPUs ({total_gpus}) is not divisible by {gpus_per_replica}"

        replicas = total_gpus // gpus_per_replica
        layout_np = np.reshape(ranks, [replicas, n_shards])
        return cls(layout_np, my_rank)

    def __eq__(self, other):
        if not isinstance(other, ModelLayout):
            return False
        if self.my_rank != other.my_rank:
            return False
        return np.array_equal(self.layout, other.layout)

    def __hash__(self):
        # Best way to hash a numpy array according to stack overflow
        # https://stackoverflow.com/a/16592241/610785
        return hash((self.layout.tostring(), self.my_rank))

    def __init__(self, layout, my_rank):
        """Layout is a numpy array with replica, shard"""
        self.layout = layout
        self.my_rank = my_rank

        self.total_gpus = layout.size
        self.all_ranks = list(range(self.total_gpus))

        self.n_replicas, self.n_shards = layout.shape

        if self.n_shards == 4:
            print(
                "WARNING: Using n_shards == 4 is currently slow because we have not"
                "implemented an efficient ring following the [0,1,3,2] pattern"
            )

        ([replica_idx], [shard_idx]) = np.where(layout == my_rank)

        self.replica_idx = int(replica_idx)
        self.shard_idx = int(shard_idx)

        # Create convenient accessors
        self.dp_sibling_ranks = [replica[shard_idx] for replica in layout]
        self.mp_sibling_ranks = list(layout[replica_idx])

        self.ranks_in_my_replica = layout[replica_idx].flatten().tolist()

        self.is_in_first_replica = self.replica_idx == 0

        self.replica_root = self.ranks_in_my_replica[0]
        self.is_replica_root = self.replica_root == self.my_rank

        self.is_logging_rank = self.is_replica_root and self.replica_idx == 0

        self.ranks_on_my_node = list(
            range(math.floor(self.my_rank / 8) * 8, 8 + math.floor(self.my_rank / 8) * 8)
        )
