from collections import namedtuple


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
        # print "builder.build src_ids {}, dst_ids {}, ratings {}"\
        #     .format(self.__src_ids, self.__dst_ids, self.__ratings)
        return RatingBlock(self.__src_ids,
                           self.__dst_ids,
                           self.__ratings)