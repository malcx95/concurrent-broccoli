# 1

(BLOCK_SIZE + KERNEL_SIZE)^2

# 2

Most of the time it copies 1x3 bytes, sometimes 2x3 bytes.

# 3

Some pixels are simply read by more than 1 thread.

# 4

Block sizes must be powers of 2.

# 5

2400 us -> 130 us for a 5x5 filter.

# 6

No, and we would have to change the order at which elements are accessed to make it coalesced.

# 7

700 -> 300 us for a 21 x 21 filter

# 8

Yes.

# 9

Slightly slower, it's now 180 us.

# 10

Using constant memory

# 11

Sorting by insertion sort.

# 12

3x3, more than this and the hairs of the dandelion disappear.


