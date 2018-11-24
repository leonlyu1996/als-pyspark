from pyspark.rdd import RDD


class Partitioner:

    def __init__(self, num_partitions):
        self.num_partitions = num_partitions

    def get_partition(self, key):
        pass


class HashPartitioner(Partitioner):

    def __init__(self, num_partitions):
        assert num_partitions >= 0, "Number of partitions ($partitions) cannot be negative."
        Partitioner.__init__(self, num_partitions)

    def get_partition(self, key):
        if key is None:
            return 0
        else:
            return self.non_negative_mod(hash(key), self.num_partitions)

    def non_negative_mod(self, x, mod):
        raw_mod = x % mod
        raw_mod += mod if raw_mod < 0 else 0
        return raw_mod

