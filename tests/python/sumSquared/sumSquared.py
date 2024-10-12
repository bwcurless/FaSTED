#!/usr/bin/env python

val = 0
# Check first 32 rows (points)
for row in range(32):
    sum = 0
    # Sum up the first 64 squared terms
    for x in range(64):
        sum += val ** 2
        val += 1
    print(f"Row {row} = {sum}")

