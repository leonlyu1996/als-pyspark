class LocalIndexEncoder:

    def __init__(self, num_blocks):
        assert isinstance(num_blocks, int)
        self.num_blocks = num_blocks
        # use 32 bits to store the blockId and local index id
        self.num_local_index_bits = min((num_blocks - 1).bit_length(), 31)
        self.local_index_mask = (1 << self.num_local_index_bits) - 1

    def encode(self, block_id, local_index):
        assert block_id < self.num_blocks,\
            "block id should be smaller than number of blocks"

        assert (local_index & ~self.local_index_mask) == 0

        return block_id << self.num_local_index_bits | local_index

    def get_block_id(self, encoded):

        return encoded >> self.num_local_index_bits

    def get_local_index(self, encoded):

        return encoded & self.local_index_mask






