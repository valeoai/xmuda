from torch.utils.data.sampler import Sampler


class IterationBasedBatchSampler(Sampler):
    """
    Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled

    References:
        https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration < self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                yield batch
                iteration += 1
                if iteration >= self.num_iterations:
                    break

    def __len__(self):
        return self.num_iterations - self.start_iter


def test_IterationBasedBatchSampler():
    from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler
    sampler = RandomSampler([i for i in range(9)])
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, 6, start_iter=0)

    # check __len__
    # assert len(batch_sampler) == 5
    for i, index in enumerate(batch_sampler):
        print(i, index)
        # assert [i * 2, i * 2 + 1] == index

    # # check start iter
    # batch_sampler.start_iter = 2
    # assert len(batch_sampler) == 3


if __name__ == '__main__':
    test_IterationBasedBatchSampler()
