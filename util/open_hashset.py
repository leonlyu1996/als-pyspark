

class OpenHashSet:

    MAX_CAPACITY = 2 ** 30

    def __init__(self, init_capacity, load_factor):

        assert init_capacity <= OpenHashSet.MAX_CAPACITY, \
            "Can't make capacity bigger than {} elements".format(OpenHashSet.MAX_CAPACITY)
        assert init_capacity >= 0, "Invalid initial capacity"
        assert init_capacity < 1.0, "Load factor must be less than 1.0"
        assert init_capacity > 0.0, "Load factor must be greater than 0.0"


    def _hasher(self):
        pass

