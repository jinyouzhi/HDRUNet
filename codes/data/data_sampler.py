import paddle
"""
Modified from torch.utils.data.distributed.DistributedSampler
Support enlarging the dataset for *iter-oriented* training, for saving time when restart the
dataloader after each epoch
"""
import math


class DistIterSampler(paddle.io.Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, ratio=100):
        if num_replicas is None:
>>>>>>            if not torch.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
>>>>>>            num_replicas = torch.distributed.get_world_size()
        if rank is None:
>>>>>>            if not torch.distributed.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = paddle.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * ratio / self.
            num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = paddle.framework.core.default_cpu_generator()
        g.manual_seed(self.epoch)
        indices = paddle.randperm(n=self.total_size).tolist()
        dsize = len(self.dataset)
        indices = [(v % dsize) for v in indices]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
