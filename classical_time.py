import time
import statistics
import random
from math import pi, sqrt, floor

n = 4
m = n
A = [
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
]

def toy_hash_bits(x: int) -> str:
    """Simple parity-based hash function."""
    bits = []
    for j in range(m):
        s = 0
        for i in range(n):
            if A[j][i]:
                s ^= ((x >> i) & 1)
        bits.append(str(s))
    return ''.join(bits)


def target_function():
    hidden = random.randint(0, 2**n - 1)
    target_bits = toy_hash_bits(hidden)

    S = [x for x in range(2**n) if toy_hash_bits(x) == target_bits]
    m_solutions = len(S)


def benchmark(func, runs=10):
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    return times

if __name__ == "__main__":
    execution_times = benchmark(target_function)
    print(f"Ran {len(execution_times)} times")
    print(f"Fastest: {min(execution_times):.6f} s")
    print(f"Slowest: {max(execution_times):.6f} s")
    print(f"Average: {statistics.mean(execution_times):.6f} s")
    print(execution_times)




