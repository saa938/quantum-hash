from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import random
from math import pi, sqrt, floor
import time

start = time.time()

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
    for j in range(m):
        s = 0
        for i in range(n):
            if A[j][i]:
                s ^= ((x >> i) & 1)
        bits.append(str(s))
    return ''.join(bits)


hidden = random.randint(0, 2**n - 1)
target = toy_hash_bits(hidden)


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

def apply_oracle(qc: QuantumCircuit):
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

def apply_diffusion(qc: QuantumCircuit):
    for i in range(n):
        qc.h(in_idx(i))
        qc.x(in_idx(i))

    target = in_idx(n - 1)
    controls = [in_idx(i) for i in range(n - 1)]

    qc.h(target)
    if len(controls) == 0:
        qc.x(target)
    else:
        qc.mcx(controls, target, diff_ancillas)
    qc.h(target)

    for i in range(n):
        qc.x(in_idx(i))
        qc.h(in_idx(i))

qc = QuantumCircuit(total_qubits, n)

for i in range(n):
    qc.h(in_idx(i))

for _ in range(3):
    apply_oracle(qc)
    apply_diffusion(qc)

for i in range(n):
    qc.measure(in_idx(i), i)

backend = AerSimulator(seed_simulator=42)
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=4096)
res = job.result()
counts = res.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

top_bs, top_count = sorted_counts[0]
top_int = int(top_bs, 2)

print("Is it a true solution?:", toy_hash_bits(top_int) == toy_hash_bits(hidden))

end = time.time()
print(f"Execution time: {end - start:.4f} seconds")
