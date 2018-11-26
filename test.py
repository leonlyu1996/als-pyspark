a = [1, 2, 3, 4, 5, 6]

rank = 3
pos = 0
for i in range(rank):

    for j in range(i, rank):
        print "i: {}, j: {}, v {}".format(i, j, a[pos])
        pos += 1