from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from Rating import Rating, RatingBlock, RatingBlockBuilder
from InBlock import InBlock, UncompressedInBlock, UncompressedInBlockBuilder
from Solver import NormalEquation, NNLSSolver, LeastSquaresNESolver, CholeskySolver
from pyspark.rdd import RDD
import numpy as np
from collections import namedtuple
from pyspark.mllib.common import callMLlibFunc
from pyspark.storagelevel import StorageLevel
from pyspark import SparkContext, SparkConf
from util.partitioner import HashPartitioner, Partitioner
from util.encoder import LocalIndexEncoder
from py4j.java_gateway import java_import
import time
import pdb


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
            items = list(items)
            if items:
                # print "items: {}".format(items)
                items = items[0]
                ids = items[1][0]
                factors = items[1][1]
                for pair in zip(ids, factors):
                    yield pair
                # yield zip(ids, factors)

        user_part = HashPartitioner(num_user_blocks)
        item_part = HashPartitioner(num_item_blocks)

        block_ratings = self.partition_ratings(ratings, user_part, item_part)\
                            .persist(intermediate_rdd_storage_level)

        # print "block_rating: {}".format(block_ratings.glom().collect())
        (user_in_blocks, user_out_blocks) = \
            self.make_blocks(block_ratings,
                             user_part,
                             item_part,
                             intermediate_rdd_storage_level)

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

        seed = round(time.time())
        user_factors = self.initialize(user_in_blocks, rank, seed)
        item_factors = self.initialize(item_in_blocks, rank, seed + 1)

        print "init_user_factors {}".format(user_factors.glom().collect())
        print "init_item_factors {}".format(item_factors.glom().collect())
        # exit(0)
        # check point file?

        solver = NNLSSolver() if nonnegative else CholeskySolver()

        if implicit_prefs:
            for iter in range(1, max_iter + 1):

                user_factors.persist(intermediate_rdd_storage_level)
                print "user_factors {}".format(user_factors)
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
                print "user_factors {}".format(user_factors.collect())

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
                          .mapPartitions(zip_id_and_factor,
                                         preservesPartitioning=True)\
                          .persist(final_rdd_storage_level)

        print "user_id_and_factors {}".format(user_id_and_factors.glom().collect())
        # user_id_and_factors = \
        #     user_in_blocks.mapValues(lambda in_block: in_block.src_ids)\
        #                   .join(user_factors)\
        #                   .mapPartitions(lambda items: items.flatMap(zip_id_and_factor),
        #                                  preservesPartitioning=True)\
        #                   .persist(final_rdd_storage_level)

        # print "user_id_and_factors: {}".format(user_id_and_factors.collect())
        item_id_and_factors = \
            item_in_blocks.mapValues(lambda in_blocks: in_blocks.src_ids)\
                          .join(item_factors)\
                          .mapPartitions(zip_id_and_factor,
                                         preservesPartitioning=True)\
                          .persist(final_rdd_storage_level)

        if final_rdd_storage_level != None:
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

        # print "rating_in_partition: {}".format(rating.collect())
        # print "len_rating: {}".format(len(rating.collect()))

        def map_rating(iterator, num_partitions):

            rating_bolck_builders = \
                [RatingBlockBuilder() for _ in range(num_partitions)]

            for r in iterator:
                src_block_id = src_part.get_partition(r.user)
                dst_block_id = dst_part.get_partition(r.item)
                idx = src_block_id + src_part.num_partitions * dst_block_id
                builder = rating_bolck_builders[idx]
                # different user and product id can be hashed to same builder
                # print "(r.user {}, r.item {}, r.rating {})".format(r.user, r.item, r.rating)
                # print "src_block_id {}, dst_block_id {}".format(src_block_id, dst_block_id)
                builder.add(r)

                # if builder size >= 2048, build the full builder and insert a new builder
                if builder.size >= 2048:
                    rating_bolck_builders[idx] = RatingBlockBuilder()
                    yield ((src_block_id, dst_block_id), builder.build())

            # print "rating_block_builder++++: {}".format(len(rating_bolck_builders))

            for idx, builder in enumerate(rating_bolck_builders):
                if builder.size > 0:
                    src_block_id = idx % src_part.num_partitions
                    dst_block_id = idx / src_part.num_partitions

                    yield ((src_block_id, dst_block_id), builder.build())


        def aggregate(block):

            builder = RatingBlockBuilder()

            for blk in block:
                # print "blk src_ids: {}, dst_ids: {}, ratings: {}".format(blk.src_ids, blk.dst_ids, blk.ratings)
                builder.merge(blk)
            # block.foreach(builder.merge)

            # yield or return? yield -> get a generator
            return builder.build()

        num_partitions = \
            src_part.num_partitions * dst_part.num_partitions

        # print "num_partitions_should_be_4 : {}".format(num_partitions)

        # rating_bolck_builders = sc.accumulator(num_partitions, AccumulateRatingBlockBuilder())

        rating_bolcks = \
            rating.mapPartitions(lambda iter: map_rating(iter, num_partitions))\
                  .groupByKey()\
                  .mapValues(aggregate)

        # print "rating blocks : {}".format(rating_bolcks.glom().collect())
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

            sorted_dst_ids = sorted(list(dst_id_set))

            # i is the dst_id position in this block
            for i, dst_id in enumerate(sorted_dst_ids):
                dst_id_to_local_index[dst_id] = i

            dst_local_indices = \
                map(lambda id: dst_id_to_local_index[id], block.dst_ids)

            return (rating_block[0][0], (rating_block[0][1],
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

                #print rating_block

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
            active_ids = [np.zeros(0, dtype=int) for _ in range(dst_part.num_partitions)]
            seen = np.empty(dst_part.num_partitions, dtype=bool)

            # print "in_block.src_ids {}".format(in_block.src_ids)

            for i in range(len(in_block.src_ids)):

                seen.fill(False)

                for j in range(in_block.dst_ptrs[i], in_block.dst_ptrs[i + 1]):
                    block_id = \
                        encoder.get_block_id(in_block.dst_encoded_incides[j])

                    # print "block_id {}".format(block_id)
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

        # print "in_blocks: {}".format(in_blocks.collect())

        # in ALS.scala the array is used
        out_blocks = \
            in_blocks.mapValues(lambda value: build_out_block(value, dst_part))\
                     .persist(storage_level)

        # print "out_blocks: {}".format(out_blocks.collect())

        return in_blocks, out_blocks

    def initialize(self, in_blocks, rank, seed):

        assert isinstance(in_blocks, RDD)

        # In ALS.scala, it use XOR random to initialize factors
        # Here use uniform random
        def ini_random_factor(src_in_block, rank, seed):
            """

            :param src_in_block:  (src_block_id, in_block)
            :param rank:
            :return:
            """
            # list of arrays
            np.random.seed(src_in_block[0] + int(seed))
            factors = \
                [np.random.uniform(0, 1, rank) for _ in range(len(src_in_block[1].src_ids))]

            return src_in_block[0], factors

        return in_blocks.map(lambda x: ini_random_factor(x, rank, seed))
        # return in_blocks.mapPartitionsWithIndex(lambda index, x: ini_random_factor(index, x, rank))

    # [[(0, [array([0.98587914, 0.64050906, 0.73872144, 0.90498466, 0.83292195,
    #               0.5986962, 0.39871989, 0.55432934, 0.0321833, 0.04190795]),
    #        array([0.36081376, 0.35504538, 0.45142644, 0.78485457, 0.29190276,
    #               0.0273109, 0.7925073, 0.14460332, 0.88612463, 0.14855])])],
    #  [(1, [array([0.98587914, 0.64050906, 0.73872144, 0.90498466, 0.83292195,
    #               0.5986962, 0.39871989, 0.55432934, 0.0321833, 0.04190795]),
    #        array([0.36081376, 0.35504538, 0.45142644, 0.78485457, 0.29190276,
    #               0.0273109, 0.7925073, 0.14460332, 0.88612463, 0.14855])])]]
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

            :param joined_src_out:  (srcBlockId, (srcOutBlock, srcFactors))
            :return:
            """
            src_out_block = joined_src_out[1][0]
            src_block_id = joined_src_out[0]
            src_factors = joined_src_out[1][1]

            # print "src_out_block {}, src_block_id {}, src_factors {}"\
            #     .format(src_out_block, src_block_id, src_factors)

            # assert isinstance(src_out_block, RDD)
            dst_block_ids = [i for i in range(len(src_out_block))]
            print "src_out_block {}".format(src_out_block)
            src_out_block = zip(src_out_block, dst_block_ids)
            print "zipped_block_and_id {}".format(src_out_block)
            results = map(lambda zip_block_and_id:
                          (zip_block_and_id[1], (src_block_id, list(map(lambda idx: src_factors[idx], zip_block_and_id[0])))), src_out_block)

            print "result: {}".format(list(results))

            for result in list(results):
                yield result
            # yield joined_src_out.map(lambda x: x[1][0])\
            #                     .zipWithIndex()\
            #                     .map(lambda (active_incides, dst_block_id):
            #                          (dst_block_id, (src_block_id, active_incides.map(lambda idx: src_factors[idx]))))

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

            sorted_src_factors = [list() for _ in range(num_src_blocks)]
            # print "src_factors {}".format([i for i in src_factors])
            for factor in src_factors:
                sorted_src_factors[factor[0]] = factor[1]

            # print "sorted_src_factors {}".format(sorted_src_factors)

            dst_factors = [np.zeros(0) for _ in range(in_block.size)]
            # ["tri_k", "ata", "atb", "da", "k"]
            #     self.tri_k = k * (k + 1) / 2
            #     self.ata = np.zeros(self.tri_k)
            #     self.atb = np.zeros(k)
            #     self.da = np.zeros(k)
            #     self.k = k
            ls = NormalEquation(rank)
            for j in range(len(in_block.src_ids)):
                ls.reset()

                if implicit_prefs:
                    ls.merge(y_t_y)

                num_explicits = 0
                for i in range(in_block.dst_ptrs[j], in_block.dst_ptrs[j + 1]):

                    encoded = in_block.dst_encoded_incides[i]

                    block_id = src_encoder.get_block_id(encoded)
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

        print "src_out_blocks {}".format(src_out_blocks.collect())

        src_out = src_out_blocks.join(src_factor_blocks)\
                                .flatMap(trans_src_out_blocks)

        print "src_out {}".format(src_out.collect())
        assert isinstance(src_out, RDD)

        partitioner = HashPartitioner(dst_in_blocks.getNumPartitions())

        merged = src_out.groupByKey(numPartitions=partitioner.num_partitions,
                                    partitionFunc=partitioner.get_partition)

        print "groupbykey {}".format(merged.collect())
        dst_factors =  \
            dst_in_blocks.join(merged)
        print "intermmediate dst_in_blocks {}".format(dst_factors.glom().collect())
        dst_factors = dst_factors.mapValues(lambda in_block_with_factors:
                                    compute_dst_factors(in_block_with_factors, num_src_blocks, src_encoder, solver))

        # print "dst_factors {}".format(dst_factors.collect())
        # dst_factors =  \
        #     dst_in_blocks.join(merged)\
        #                  .mapValues(lambda in_block_with_factors:
        #                             compute_dst_factors(in_block_with_factors, num_src_blocks, src_encoder, solver))

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

    import os
    os.environ["PYSPARK_PYTHON"] = "/Users/leon/anaconda3/envs/py27/bin/python"
    conf = SparkConf().set("spark.jars",
                           "/Users/az/Downloads/HKUST/Big_Data_Computing_5003/project/als/ALS/netlib-java-1.1.jar")

    sc = SparkContext(conf=conf)

    data = sc.textFile("../data/test.data", 2)
    ratings = data.map(lambda l: l.split(',')) \
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    # print ratings.isEmpty()
    print ratings.glom().collect()

    # Build the recommendation model using Alternating Least Squares
    rank = 10
    numIterations = 10

    als = NewALS()

    num_user_blocks = max(ratings.getNumPartitions(), ratings.getNumPartitions() / 2)
    print num_user_blocks

    num_item_blocks = max(ratings.getNumPartitions(), ratings.getNumPartitions() / 2)

    (user_id_and_factors, item_id_and_factors) = als.train(ratings=ratings,
                                                           rank=rank,
                                                           num_user_blocks=num_user_blocks,
                                                           num_item_blocks=num_item_blocks,
                                                           max_iter=numIterations,
                                                           reg_param=0.01,
                                                           nonnegative=False)

    print "user_id_and_factor: {}".format(user_id_and_factors.collect())
    print "item_id_and_factor {}".format(item_id_and_factors.collect())


# test.map(x => (x.user, (x.product, x.rating))).join(userFactors).map { case (userId, ((prodId, rating), userFactor)) =>
#       (prodId, (rating, userFactor))
#     }.join(prodFactors).values.map { case ((rating, userFactor), prodFactor) =>
#       (blas.sdot(k, userFactor, 1, prodFactor, 1), rating)
#     }
# mse = predictionAndRatings.map { case (pred, rating) =>
#       val err = pred - rating
#       err * err
#     }.mean()
    predictionAndRatings = ratings.map(lambda user_prod_pair: (user_prod_pair[0], (user_prod_pair[1], user_prod_pair[2])))\
                                  .join(user_id_and_factors)\
                                  .map(lambda joined: (joined[1][0][0], (joined[1][0][1], joined[1][1]))).join(item_id_and_factors)\
                                  .mapValues(lambda value: (np.dot(value[0][1], value[1]), value[0][0]))
    # predictionAndRatings = data.map(lambda user_prod_pair: (user_prod_pair[0], (user_prod_pair[1], user_prod_pair[2])))\
    #                            .join(user_id_and_factors)\
    #                            .map(lambda joined: (joined[1][0], (joined[1][1], joined[2]))).join(item_id_and_factors)\
    #                            .mapValues(lambda value: (np.dot(value[0][1], value[1]), value[0][0]))
    print "predict_and_rating {}".format(predictionAndRatings.collect())
    mse = predictionAndRatings.map(lambda pred_and_rating: np.power(pred_and_rating[1][0] - pred_and_rating[1][1], 2)).mean()
    print "mse {}".format(mse)






