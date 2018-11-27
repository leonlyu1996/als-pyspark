from collections import namedtuple
import numpy as np
from pyspark.accumulators import AccumulatorParam
from Rating import RatingBlockBuilder
from util.encoder import LocalIndexEncoder



class InBlock(object):

    def __init__(self,
                 src_ids,
                 dst_ptrs,
                 dst_encoded_incides,
                 ratings):
        """

        :param src_ids:                 (list)
        :param dst_ptrs:                (list)
        :param dst_encoded_incides:     (list)
        :param ratings:                 (list)
        """
        self.src_ids = src_ids
        self.dst_ptrs = dst_ptrs
        self.dst_encoded_incides = dst_encoded_incides
        self.ratings = ratings
        self.size = len(ratings)
        # print "dst_encoded_incides: {}, self.size: {}, dst_ptrs: {}".format(len(dst_encoded_incides), self.size, len(dst_ptrs))
        assert len(dst_encoded_incides) == self.size
        assert len(dst_ptrs) == len(src_ids) + 1

    # def __reduce__(self):
    #     size = len(self.ratings)
    #     print "dst_encoded_incides: {}, self.size: {}, dst_ptrs: {}"\
    #         .format(len(self.dst_encoded_incides), size, len(self.dst_ptrs))
    #
    #     assert len(self.dst_encoded_incides) == size
    #     assert len(self.dst_ptrs) == len(self.src_ids) + 1
    #     return InBlock, (self.src_ids, self.dst_ptrs, self.dst_encoded_incides, self.ratings)

        # (namedtuple("InBlock", ["src_ids", "dst_ptrs", "dst_encoded_incides", "ratings"]))


class UncompressedInBlock:

    def __init__(self, src_ids, dst_encoded_incides, ratings):
        uncomp_in_blocks_tuples = \
            zip(src_ids, dst_encoded_incides, ratings)

        sorted_uncomp_in_blocks_tuples = \
            sorted(uncomp_in_blocks_tuples, key=lambda x: x[0])

        # print "sorted_uncomp_in_blocks_tuples: {}".format(sorted_uncomp_in_blocks_tuples)
        self._src_ids, self._dst_encoded_incides, self._ratings = \
            zip(*sorted_uncomp_in_blocks_tuples)

        self.size = len(src_ids)

    def compress(self):
        size = len(self._src_ids)
        assert size > 0, "Empty in-link block should not exist."

        # TimSort, sort UncompressedInBlock according to the srcIds
        # implemented in __init__

        unique_src_ids = list()
        dst_counts = list()
        pre_src_id = self._src_ids[0]
        unique_src_ids.append(pre_src_id)

        # print "__src_ids: {}".format(self._src_ids)
        curr_count = 1
        for i in range(1, self.size):
            src_id = self._src_ids[i]
            # print "src_id: {}".format(src_id)
            # print "pre_src_id: {}".format(pre_src_id)
            if src_id != pre_src_id:
                # print "curr_count: {}".format(curr_count)
                unique_src_ids.append(src_id)
                dst_counts.append(curr_count)
                pre_src_id = src_id
                curr_count = 0

            curr_count += 1

        dst_counts.append(curr_count)

        num_unique = len(unique_src_ids)
        dst_ptrs = np.zeros(num_unique + 1, dtype=int)
        sum_count = 0
        i = 0
        #print "unique_src_ids {}".format(unique_src_ids)
        while i < num_unique:
            sum_count += dst_counts[i]
            i += 1
            dst_ptrs[i] = int(sum_count)

        #print "unique_src_ids: {}, dst_counts: {}, dst_ptrs: {}".format(unique_src_ids, dst_counts, dst_ptrs)
        # pdb.set_trace()
        return InBlock(unique_src_ids,
                       dst_ptrs,
                       self._dst_encoded_incides,
                       self._ratings)


class UncompressedInBlockBuilder:

    def __init__(self, encoder):
        self.__src_ids = list()
        self.__dst_encoded_incides = list()
        self.__ratings = list()
        # assert isinstance(encoder, LocalIndexEncoder)
        self.encoder = encoder


    def add(self,
            dst_block_id,
            src_ids,
            dst_local_incides,
            ratings):

        size = len(src_ids)
        assert len(dst_local_incides) == size
        assert len(ratings) == size

        self.__src_ids += src_ids
        self.__ratings += ratings

        for i in range(size):
            encoded_incides = \
                self.encoder.encode(dst_block_id, dst_local_incides[i])

            self.__dst_encoded_incides.append(encoded_incides)

    def build(self):
        return UncompressedInBlock(self.__src_ids,
                                   self.__dst_encoded_incides,
                                   self.__ratings)

class LocalIndexEncoder(AccumulatorParam):

    def zero(self, value):
        return [RatingBlockBuilder() for _ in range(value)]

    def addInPlace(self, value1, value2):
        """

        :param value1:  (list) list of RatingBlockBuilder
        :param value2:  (idx, RatingBlockBuilder) RatingBlockBuilder to be added in the list
        :return:
        """
        value1[value2[0]] = value2[1]
        return value1
