from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from pyspark.rdd import RDD
import numpy as np
from collections import namedtuple
from pyspark.mllib.common import callMLlibFunc
from pyspark.storagelevel import StorageLevel
from pyspark import SparkContext
from pyspark.accumulators import AccumulatorParam
from util.partitioner import HashPartitioner, Partitioner
from util.encoder import LocalIndexEncoder
from scipy.linalg import blas, cholesky
from scipy.optimize import nnls
from numpy.linalg.linalg import LinAlgError


sc = SparkContext()


# class Rating:
#
#     def __init__(self, user, item, rating):
#         self.user = user
#         self.item = item
#         self.rating = rating

class Rating(namedtuple("Rating", ["user", "item", "rating"])):
    """
    Represents a (user, product, rating) tuple.

    >>> r = Rating(1, 2, 5.0)
    >>> (r.user, r.item, r.rating)
    (1, 2, 5.0)
    >>> (r[0], r[1], r[2])
    (1, 2, 5.0)

    .. versionadded:: 1.2.0
    """

    def __reduce__(self):
        return Rating, (int(self.user), int(self.item), float(self.rating))


class RatingBlock(namedtuple("RatingBlock", ["src_ids", "dst_ids", "ratings"])):

    # def __init__(self, src_ids, dst_ids, ratings):
    #     assert isinstance(src_ids, list), "src_ids should be a list!"
    #     assert len(dst_ids) == len(src_ids)
    #     assert len(ratings) == len(src_ids)
    #
    #     self.src_ids = src_ids
    #     self.dst_ids = dst_ids
    #     self.ratings = ratings

    def __reduce__(self):
        return RatingBlock, (self.src_ids, self.dst_ids, self.ratings)

    def size(self):
        return len(self.src_ids)


class RatingBlockBuilder:
    def __init__(self):
        self.__src_ids = list()
        self.__dst_ids = list()
        self.__ratings = list()
        self.size = 0

    def add(self, r):
        assert isinstance(r, Rating), \
            "input of add should be an instance of Rating!"

        self.size += 1
        self.__src_ids += [r.user]
        self.__dst_ids += [r.item]
        self.__ratings += [r.rating]

    def merge(self, other):
        # assert isinstance(other, RatingBlock), \
        #     "input of merge should be an instance of RatingBlock!"

        self.size += len(other.src_ids)
        self.__src_ids += other.src_ids
        self.__dst_ids += other.dst_ids
        self.__ratings += other.ratings

    def build(self):
        return RatingBlock(self.__src_ids,
                           self.__dst_ids,
                           self.__ratings)


class InBlock:

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
        self.size = len(src_ids)
        assert len(dst_ptrs) == self.size
        assert len(dst_ptrs) == self.size + 1


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

        curr_count = 1
        for i in range(1, self.size):
            src_id = self._src_ids[i]
            if src_id != pre_src_id:
                unique_src_ids.append(src_id)
                dst_counts.append(curr_count)
                pre_src_id = src_id
                curr_count = 0

            curr_count += 1

        dst_counts.append(curr_count)

        print "unique_src_ids: {}, dst_counts: {}".format(unique_src_ids, dst_counts)
        num_unique = len(unique_src_ids)
        dst_ptrs = np.zeros(num_unique + 1)
        sum_count = 0
        i = 0
        while i < num_unique:
            sum_count += unique_src_ids[i]
            i += 1
            dst_ptrs[i] = sum_count

        return InBlock(unique_src_ids,
                       dst_ptrs,
                       self._dst_encoded_incides,
                       self._ratings)


class UncompressedInBlockBuilder:

    def __init__(self, encoder):
        self.__src_ids = list()
        self.__dst_encoded_incides = list()
        self.__ratings = list()
        assert isinstance(encoder, LocalIndexEncoder)
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


class AccumulateRatingBlockBuilder(AccumulatorParam):

    def zero(self, value):
        return [RatingBlockBuilder()] * value

    def addInPlace(self, value1, value2):
        """

        :param value1:  (list) list of RatingBlockBuilder
        :param value2:  (idx, RatingBlockBuilder) RatingBlockBuilder to be added in the list
        :return:
        """
        value1[value2[0]] = value2[1]
        return value1


class NormalEquation:

    def __init__(self, k):
        self.tri_k = k * (k + 1) / 2
        self.ata = np.zeros(self.tri_k)
        self.atb = np.zeros(k)
        self.da = np.zeros(k)
        self.k = k

    def copy2double(self, a):
        for i in range(self.k):
            self.da[i] = a[i]

    def add(self, a, b, c=1.0):
        assert c > 0
        assert a.shape[0] == self.k
        self.copy2double(a)

        # use ata as return?
        self.ata = blas.dspr(self.k, c, self.da, self.ata, lower=1)
        if b != 0:
            self.atb = blas.daxpy(self.da, self.atb, n=self.k, a=b)

        return self

    def merge(self, other):

        assert other.k == self.k
        self.ata = \
            blas.daxpy(other.ata, self.ata, n=self.ata.shape[0], a=1.0)

        self.atb =\
            blas.daxpy(other.atb, self.atb, n=self.atb.shape[0], a=1.0)

        return self

    def reset(self):
        self.ata.fill(0.0)
        self.atb.fill(0.0)


class LeastSquaresNESolver:

    def __init__(self):
        pass

    def solve(self, ne, lamd):
        pass


class NNLSSolver(LeastSquaresNESolver):

    def __init__(self):
        LeastSquaresNESolver.__init__(self)
        self.__rank = -1
        self.__ata = None
        self.__initialized = False

    def initialize(self, rank):
        if not self.__initialized:
            self.__rank = rank
            self.__ata = np.zeros(rank * rank)
            self.__initialized = True

        else:
            assert self.__rank == rank

    def solve(self, ne, lamd):
        rank = ne.k
        self.initialize(rank)
        self.fill_ata(ne.ata, lamd)
        x = nnls(self.__ata, ne.atb)
        ne.reset()
        return x

    def fill_ata(self, tri_ata, lamd):

        pos = 0
        for i in range(self.__rank):

            for j in range(i + 1):

                a = tri_ata[pos]
                self.__ata[i * self.__rank + j] = a
                self.__ata[j * self.__rank + i] = a
                pos += 1

            self.__ata[i * self.__rank + i] += lamd


class CholeskySolver(LeastSquaresNESolver):

    def __init__(self):
        LeastSquaresNESolver.__init__(self)
        self.__ata = None

    def solve(self, ne, lamd):
        k = ne.k
        j = 2
        for i in range(ne.tri_k):
            ne.ata[i] += lamd
            i += j
            j += 1

        self.fill_ata(ne.ata, lamd, k)
        try:
            inverse_ata = cholesky(self.__ata)
        except LinAlgError:
            print "2-th leading minor of the array may be not positive definite"
            exit(1)

        x = np.dot(inverse_ata, ne.atb.T)
        ne.reset()
        return x

    def fill_ata(self, tri_ata, lamd, rank):
        pos = 0
        for i in range(rank):

            for j in range(i + 1):
                a = tri_ata[pos]
                self.__ata[i * rank + j] = a
                self.__ata[j * rank + i] = a
                pos += 1

            self.__ata[i * rank + i] += lamd


#########
# Class NewALS
#########


class NewALS:

    def __init__(self):
        pass

    def train(self,
              ratings,
              rank=10,
              num_user_blocks=10,
              num_item_blocks=10,
              max_iter=10,
              reg_param=0.1,
              implicit_prefs=False,
              alpha=1.0,
              nonnegative=False,
              intermediate_rdd_storage_level=StorageLevel.MEMORY_AND_DISK,
              final_rdd_storage_level=StorageLevel.MEMORY_AND_DISK,
              check_point_interval=10,
              seed=0):

        assert isinstance(ratings, RDD)
        assert not ratings.isEmpty(), "ratings RDD is empty!"
        assert intermediate_rdd_storage_level is not None, \
            "ALS is not designed to run without persisting intermediate RDDs."

        def zip_id_and_factor(items):
            """

            :param items: (_, (ids, factors))
            :return:
            """

            ids = items[1][0]
            factors = items[1][1]
            return zip(ids, factors)

        user_part = HashPartitioner(num_user_blocks)
        item_part = HashPartitioner(num_item_blocks)

        block_ratings = self.partition_ratings(ratings, user_part, item_part)\
                            .persist(intermediate_rdd_storage_level)

        (user_in_blocks, user_out_blocks) = \
            self.make_blocks(block_ratings,
                             user_part,
                             item_part,
                             intermediate_rdd_storage_level)

        print user_out_blocks.collect()
        user_out_blocks.count()

        swapped_block_ratings = \
            block_ratings.map(lambda ((user_block_id, item_block_id), block):
                                     ((item_block_id, user_block_id),
                                      RatingBlock(block.dst_ids,
                                                  block.src_ids,
                                                  block.ratings)))

        (item_in_blocks, item_out_blocks) = \
            self.make_blocks(swapped_block_ratings,
                             item_part,
                             user_part,
                             intermediate_rdd_storage_level)

        item_out_blocks.count()

        user_local_index_encoder = LocalIndexEncoder(user_part.num_partitions)
        item_local_index_encoder = LocalIndexEncoder(item_part.num_partitions)

        user_factors = self.initialize(user_in_blocks, rank)
        item_factors = self.initialize(item_in_blocks, rank)

        # check point file?

        solver = NNLSSolver if nonnegative else CholeskySolver

        if implicit_prefs:
            for iter in range(1, max_iter + 1):

                user_factors.persist(intermediate_rdd_storage_level)
                previous_item_factors = item_factors
                item_factors = self.compute_factors(src_factor_blocks=user_factors,
                                                    src_out_blocks=user_out_blocks,
                                                    dst_in_blocks=item_in_blocks,
                                                    rank=rank,
                                                    reg_param=reg_param,
                                                    src_encoder=user_local_index_encoder,
                                                    implicit_prefs=implicit_prefs,
                                                    alpha=alpha,
                                                    solver=solver)
                previous_item_factors.unpersist()
                item_factors.persist(intermediate_rdd_storage_level)

                previous_user_factors = user_factors
                user_factors = self.compute_factors(src_factor_blocks=item_factors,
                                                    src_out_blocks=item_out_blocks,
                                                    dst_in_blocks=user_in_blocks,
                                                    rank=rank,
                                                    reg_param=reg_param,
                                                    src_encoder=item_local_index_encoder,
                                                    implicit_prefs=implicit_prefs,
                                                    alpha=alpha,
                                                    solver=solver)

                previous_user_factors.unpersist()

        else:

            for iter in range(0, max_iter):

                item_factors = self.compute_factors(src_factor_blocks=user_factors,
                                                    src_out_blocks=user_out_blocks,
                                                    dst_in_blocks=item_in_blocks,
                                                    rank=rank,
                                                    reg_param=reg_param,
                                                    src_encoder=user_local_index_encoder,
                                                    solver=solver)

                user_factors = self.compute_factors(src_factor_blocks=item_factors,
                                                    src_out_blocks=item_out_blocks,
                                                    dst_in_blocks=user_in_blocks,
                                                    rank=rank,
                                                    reg_param=reg_param,
                                                    src_encoder=item_local_index_encoder,
                                                    solver=solver)

        user_id_and_factors = \
            user_in_blocks.mapValues(lambda in_block: in_block.src_ids)\
                          .join(user_factors)\
                          .mapPartitions(lambda items: items.flatMap(zip_id_and_factor),
                                         preservesPartitioning=True)\
                          .persist(final_rdd_storage_level)

        item_id_and_factors = \
            item_in_blocks.mapValues(lambda in_blocks: in_blocks.src_ids)\
                          .join(item_factors)\
                          .mapPartitions(lambda items: items.flatMap(zip_id_and_factor),
                                         preservesPartitioning=True)\
                          .persist(final_rdd_storage_level)

        if final_rdd_storage_level != StorageLevel.None:
            user_id_and_factors.count()
            item_factors.unpersist()
            item_id_and_factors.count()
            user_in_blocks.unpersist()
            user_out_blocks.unpersist()
            item_in_blocks.unpersist()
            item_out_blocks.unpersist()
            block_ratings.unpersist()

        return user_id_and_factors, item_id_and_factors

    def partition_ratings(self,
                          rating,
                          src_part,
                          dst_part):
        """

        :param rating: (rdd of Rating obj)
        :param src_part:
        :param dst_part:
        :return:
        """
        assert isinstance(rating, RDD), "rating should be an RDD"
        assert isinstance(src_part, Partitioner)
        assert isinstance(dst_part, Partitioner)

        def map_rating(iterator, rating_bolck_builders):

            for r in iterator:
                src_block_id = src_part.get_partition(r.user)
                dst_block_id = dst_part.get_partition(r.item)
                idx = src_block_id + src_part.num_partitions * dst_block_id
                builder = rating_bolck_builders[idx]
                # different user and product id can be hashed to same builder
                builder.add(r)

                # if builder size >= 2048, build the full builder and insert a new builder
                if builder.size >= 2048:
                    rating_bolck_builders[idx] = RatingBlockBuilder()
                    yield ((src_block_id, dst_block_id), builder.build())

            for idx, builder in enumerate(rating_bolck_builders):
                if builder.size > 0:
                    src_block_id = idx % src_part.num_partitions
                    dst_block_id = idx / src_part.num_partitions

                    yield ((src_block_id, dst_block_id), builder.build())

        def aggregate(block):

            builder = RatingBlockBuilder()
            for blk in block:
                print type(blk)
                # exit(1)
                builder.merge(blk)
            # block.foreach(builder.merge)

            # yield or return? yield -> get a generator
            return builder.build()

        num_partitions = \
            src_part.num_partitions * dst_part.num_partitions

        rating_bolck_builders = \
            [RatingBlockBuilder()] * num_partitions

        # rating_bolck_builders = sc.accumulator(num_partitions, AccumulateRatingBlockBuilder())

        rating_bolcks = \
            rating.mapPartitions(lambda iter: map_rating(iter, rating_bolck_builders))\
                  .groupByKey()\
                  .mapValues(aggregate)

        return rating_bolcks

    def make_blocks(self,
                    rating_blocks,
                    src_part,
                    dst_part,
                    storage_level):

        assert isinstance(rating_blocks, RDD)

        def map_rating_blocks(rating_block):
            """

            :param rating_block: ((src_block_id, dst_block_id), RatingBlock(src_ids, dst_ids, ratings))
            :return: (in_block, out_block)
            """
            # assert isinstance(rating_block, ((list, list), RatingBlock))

            block = rating_block[1]

            dst_id_set = set()
            dst_id_to_local_index = dict()

            for dst_id in block.dst_ids:
                dst_id_set.add(dst_id)

            print "dst_id_set: {}".format(dst_id_set)

            sorted_dst_ids = sorted(list(dst_id_set))

            # i is the dst_id position in this block
            for i, dst_id in enumerate(sorted_dst_ids):
                dst_id_to_local_index[dst_id] = i

            dst_local_indices = \
                map(lambda id: dst_id_to_local_index[id], block.dst_ids)

            yield (rating_block[0][0], (rating_block[0][1],
                                        block.src_ids,
                                        list(dst_local_indices),
                                        block.ratings))

        def build_in_block(iterator, dst_part):
            """

            :param iterator:    iterator of rating blocks which have been grouped by abd hashed
                                (dst_block_id, src_ids, dst_local_indices, ratings)
            :return:
            """
            encoder = \
                LocalIndexEncoder(dst_part.num_partitions)

            builder = \
                UncompressedInBlockBuilder(encoder)

            # rating block in each in_block
            for rating_block in iterator:

                builder.add(rating_block[0],
                            rating_block[1],
                            rating_block[2],
                            rating_block[3])

            return builder.build().compress()

        def build_out_block(in_block, dst_part):
            """

            :param in_block: InBlock(srcIds, dstPtrs, dstEncodedIndices, _)
            :return:
            """

            encoder = LocalIndexEncoder(dst_part.num_partitions)
            # list of arrays
            active_ids = [np.zeros(0)] * dst_part.num_partitions
            seen = np.empty(dst_part.num_partitions, dtype=bool)

            for i in range(in_block.size):

                seen.fill(False)

                for j in range(in_block.dst_ptrs[i], in_block.dst_ptrs[i + 1]):
                    block_id = \
                        encoder.get_block_id(in_block.dst_encoded_incides[j])

                    if not seen[block_id]:
                        active_ids[block_id] = \
                            np.append(active_ids[block_id], [i])

                        seen[block_id] = True

            return active_ids

        # HashPartitioner(src_part.num_partitions).get_partition

        # For groupByKey, in ALS.scala it new a partitioner use src_part.num_partitions
        # in fact we can use src_part directly
        in_blocks = \
            rating_blocks.map(map_rating_blocks)\
                         .groupByKey(numPartitions=src_part.num_partitions,
                                     partitionFunc=src_part.get_partition)\
                         .mapValues(lambda value: build_in_block(value, dst_part))\
                         .persist(storage_level)

        print "in_blocks: {}".format(in_blocks)
        exit(1)

        # in ALS.scala the array is used
        out_blocks = \
            in_blocks.mapValues(lambda value: build_out_block(value, dst_part))\
                     .persist(storage_level)

        return in_blocks, out_blocks

    def initialize(self, in_blocks, rank):

        assert isinstance(in_blocks, RDD)

        # In ALS.scala, it use XOR random to initialize factors
        # Here use uniform random
        def ini_random_factor(src_in_block, rank):
            """

            :param src_in_block:  (src_block_id, in_block)
            :param rank:
            :return:
            """
            # list of arrays
            factors = [np.random.uniform(0, 1, rank)] * src_in_block[1].size
            return src_in_block[0], factors

        return in_blocks.map(lambda x: ini_random_factor(x, rank))


    def compute_factors(self,
                        src_factor_blocks,
                        src_out_blocks,
                        dst_in_blocks,
                        rank,
                        reg_param,
                        src_encoder,
                        implicit_prefs=False,
                        alpha=1.0,
                        solver=LeastSquaresNESolver):

        assert isinstance(src_factor_blocks, RDD)
        assert isinstance(src_out_blocks, RDD)
        assert isinstance(dst_in_blocks, RDD)

        def trans_src_out_blocks(joined_src_out):
            """

            :param joined_src_out:  (srcBlockId, (srcOutBlock, srcFactors)
            :return:
            """
            src_out_block = joined_src_out[1][0]
            src_block_id = joined_src_out[0]
            src_factors = joined_src_out[1][1]
            assert isinstance(src_out_block, RDD)

            yield src_out_block.zipWithIndex()\
                               .map(lambda (active_incides, dst_block_id):
                                    (dst_block_id, (src_block_id, active_incides.map(lambda idx: src_factors[idx]))))

        def compute_dst_factors(in_block_with_factors,
                                num_src_blocks,
                                src_encoder,
                                solver):
            """

            :param in_block_with_factors: (InBlock(dstIds, srcPtrs, srcEncodedIndices, ratings), srcFactors)
            :return:
            """
            in_block = in_block_with_factors[0]
            src_factors = in_block_with_factors[1]

            sorted_src_factors = np.zeros(num_src_blocks)

            src_factors.foreach(lambda (src_block_id, factors):
                                np.put(sorted_src_factors, [src_block_id], [factors]))

            dst_factors = [np.zeros(0)] * in_block.size
            ls = NormalEquation(rank)
            for j in range(in_block.size):
                ls.reset()

                if implicit_prefs:
                    ls.merge(y_t_y)

                num_explicits = 0
                for i in range(in_block.dst_ptrs[j], in_block.dst_ptrs[j + 1]):

                    encoded = in_block.dst_encoded_incides[i]

                    block_id = src_encoder.blockId(encoded)
                    local_index = src_encoder.get_local_index(encoded)
                    src_factor = sorted_src_factors[block_id][local_index]

                    rating = in_block.ratings[i]

                    if implicit_prefs:
                        c1 = alpha * np.abs(rating)

                        num_explicits += 1 if rating > 0.0 else 0
                        ls.add(src_factor, 1.0 + c1 if rating > 0.0 else 0.0, c1)

                    else:
                        ls.add(src_factor, rating)
                        num_explicits += 1

                dst_factors[j] = solver.solve(ls, num_explicits * reg_param)

            return dst_factors

        num_src_blocks = \
            src_factor_blocks.getNumPartitions()

        y_t_y = \
            self.compute_y_t_y(src_factor_blocks, rank) if implicit_prefs else None

        src_out = src_out_blocks.join(src_factor_blocks)\
                                .flatMap(trans_src_out_blocks)

        assert isinstance(src_out, RDD)

        partitioner = HashPartitioner(dst_in_blocks.getNumPartitions())

        merged = src_out.groupByKey(partitioner)

        dst_factors =  \
            dst_in_blocks.join(merged)\
                         .mapValues(lambda in_block_with_factors:
                                    compute_dst_factors(in_block_with_factors, num_src_blocks, src_encoder, solver))

        return dst_factors

    def compute_y_t_y(self, factor_blocks, rank):

        def intra_add(ne, factors):

            factors.foreach(ne.add(_, 0.0))

            yield ne

        def inter_add(ne1, ne2):

            return ne1.merge(ne2)

        # return type is NormEquation
        return factor_blocks.values()\
                            .aggregate((NormalEquation(rank)),
                                       (lambda (ne, factors): intra_add(ne, factors)),
                                       (lambda (ne1, ne2): inter_add(ne1, ne2)))



if __name__ == "__main__":
    # sc = SparkContext()

    data = sc.textFile("data/test.data")
    ratings = data.map(lambda l: l.split(',')) \
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    # print ratings.isEmpty()
    print ratings.collect()

    # Build the recommendation model using Alternating Least Squares
    rank = 10
    numIterations = 10

    als = NewALS()

    num_user_blocks = max(sc.defaultParallelism, ratings.getNumPartitions() / 2)
    num_item_blocks = max(sc.defaultParallelism, ratings.getNumPartitions() / 2)

    (user_id_and_factors, item_id_and_factors) = als.train(ratings=ratings,
                                                           rank=rank,
                                                           num_user_blocks=num_user_blocks,
                                                           num_item_blocks=num_item_blocks,
                                                           max_iter=numIterations,
                                                           reg_param=0.01,
                                                           nonnegative=False)

    print user_id_and_factors
    print item_id_and_factors






