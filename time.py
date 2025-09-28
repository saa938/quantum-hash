import time
import statistics
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import random
from math import pi, sqrt, floor

n = 8
m = 8
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
    bits = []
    for j in range(8):
        s = 0
        for i in range(8):
            if A[j][i]:
                s ^= ((x >> i) & 1)
        bits.append(str(s))
    return ''.join(bits)

oracle_anc_count = max(1, m - 2)
diff_anc_count = max(1, n - 2)
in_idx = lambda i: i
out_idx = lambda j: n + j
marker_idx = n + m
oracle_anc_start = n + m + 1
oracle_ancillas = [oracle_anc_start + a for a in range(oracle_anc_count)]
diff_anc_start = oracle_anc_start + oracle_anc_count
diff_ancillas = [diff_anc_start + a for a in range(diff_anc_count)]
total_qubits = diff_anc_start + diff_anc_count

def apply_oracle(qc: QuantumCircuit, target):
    for j in range(m):
        for i in range(n):
            if A[j][i]:
                qc.cx(in_idx(i), out_idx(j))
    for j, bit in enumerate(target):
        if bit == '0':
            qc.x(out_idx(j))
    controls = [out_idx(j) for j in range(m)]
    qc.mcx(controls, marker_idx, oracle_ancillas)
    qc.z(marker_idx)
    qc.mcx(controls, marker_idx, oracle_ancillas)
    for j, bit in enumerate(target):
        if bit == '0':
            qc.x(out_idx(j))
    for j in range(m):
        for i in range(n):
            if A[j][i]:
                qc.cx(in_idx(i), out_idx(j))

# --- diffusion operator acting on the n input qubits (uses diff_ancillas only) ---
# Implements: H^{⊗n} X^{⊗n} (MCZ on all n qubits) X^{⊗n} H^{⊗n}
def apply_diffusion(qc: QuantumCircuit):
    # H and X on all input qubits
    for i in range(n):
        qc.h(in_idx(i))
        qc.x(in_idx(i))

    # Implement MCZ by doing H on the final input qubit then MCX with others as controls
    target = in_idx(n - 1)
    controls = [in_idx(i) for i in range(n - 1)]

    qc.h(target)
    if len(controls) == 0:
        # n == 1; mcx() degenerates to X on target
        qc.x(target)
    else:
        qc.mcx(controls, target, diff_ancillas)
    qc.h(target)

    # undo X and H on inputs
    for i in range(n):
        qc.x(in_idx(i))
        qc.h(in_idx(i))

# --- Backend ---
backend = AerSimulator(seed_simulator=42)

# --- Main quantum search function ---
def target_function():
    hidden = random.randint(0, 2**n - 1)
    target = toy_hash_bits(hidden)

    qc = QuantumCircuit(total_qubits, n)

    # initial uniform superposition on input
    for i in range(n):
        qc.h(in_idx(i))

    # pick number of Grover iterations
    # N = 2**n
    # if m_solutions == 0:
    #     r = 0
    # else:
    #     r = max(1, int(floor((pi / 4) * sqrt(N / m_solutions))))

    # print("Grover iterations r =", r)

    # perform r iterations of (oracle + diffusion)
    for _ in range(3):
        apply_oracle(qc, target)
        apply_diffusion(qc)

    # measure input register (measured into classical bits 0..n-1)
    for i in range(n):
        qc.measure(in_idx(i), i)

    # simulate
    backend = AerSimulator(seed_simulator=42)
    tqc = transpile(qc, backend)
    job = backend.run(tqc, shots=4096)
    res = job.result()
    counts = res.get_counts()

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    """
    print("\nCounts (top 8):", sorted_counts[:8])
    """

    top_bs, top_count = sorted_counts[0]
    top_int = int(top_bs, 2)
    """
    print("Top candidate (int):", top_int, format(top_int, f'0{n}b'))
    print("Quantum hash:", toy_hash_bits(top_int))
    print("Real hash:", toy_hash_bits(hidden))
    """
    print("Is it a true solution?:", toy_hash_bits(top_int) == toy_hash_bits(hidden))

# --- Benchmark ---
import time
import statistics

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