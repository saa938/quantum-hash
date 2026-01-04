import random
import time
import math
import multiprocessing as mp
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

n = 9
m = n
ROUNDS = 100
TARGET_OPS = int(1e6)
TIE_REL_TOL = 0.05
GROVER_ITERS = 3
SHOTS = 1024
PREBUILD_ALL = True

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

expected_candidates = max(1, 2 ** (n - 1))
iter_per_candidate = max(100, int(TARGET_OPS / expected_candidates))

def heavy_hash(x: int, iters=iter_per_candidate) -> int:
    h = x & 0xFFFFFFFF
    for _ in range(iters):
        h = (h * 1664525 + 1013904223) & 0xFFFFFFFF
        h ^= (h >> 13)
        h = (h + 0x9e3779b9) & 0xFFFFFFFF
    return h

def classical_worker(target_val: int, out_q: mp.Queue):
    start = time.perf_counter()
    for x in range(2 ** n):
        if heavy_hash(x) == target_val:
            elapsed = time.perf_counter() - start
            out_q.put(("classical", elapsed))
            return
    out_q.put(("classical", float("inf")))

def build_grover_circuit_for_target(target_bits: str, iterations=GROVER_ITERS):
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
        for j, bit in enumerate(target_bits):
            if bit == "0":
                qc.x(out_idx(j))
        controls = [out_idx(j) for j in range(m)]
        qc.mcx(controls, marker_idx, oracle_ancillas)
        qc.z(marker_idx)
        qc.mcx(controls, marker_idx, oracle_ancillas)
        for j, bit in enumerate(target_bits):
            if bit == "0":
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
        if controls:
            qc.mcx(controls, target, diff_ancillas)
        else:
            qc.x(target)
        qc.h(target)
        for i in range(n):
            qc.x(in_idx(i))
            qc.h(in_idx(i))

    qc = QuantumCircuit(total_qubits, n)
    for i in range(n):
        qc.h(in_idx(i))
    for _ in range(iterations):
        apply_oracle(qc)
        apply_diffusion(qc)
    for i in range(n):
        qc.measure(in_idx(i), i)
    return qc

def prebuild_transpiled_circuits(backend):
    mapping = {}
    all_targets = [format(t, f"0{m}b") for t in range(2 ** m)]
    print(f"Prebuilding/transpiling {len(all_targets)} Grover circuits (this may take a while)...")
    for tbits in all_targets:
        qc = build_grover_circuit_for_target(tbits)
        tqc = transpile(qc, backend, optimization_level=1)
        mapping[tbits] = tqc
    print("Prebuild complete.")
    return mapping

def run_time_race(rounds=ROUNDS, target_ops=TARGET_OPS):
    backend = AerSimulator()
    tqc_map = prebuild_transpiled_circuits(backend) if PREBUILD_ALL else {}

    classical_wins = quantum_wins = ties = 0

    for r in range(1, rounds + 1):
        hidden = random.randint(0, 2 ** n - 1)
        target_bits = toy_hash_bits(hidden)
        heavy_target = heavy_hash(hidden)

        q_for_child = mp.Queue()
        p_class = mp.Process(target=classical_worker, args=(heavy_target, q_for_child))
        p_class.start()

        start_q = time.perf_counter()
        if PREBUILD_ALL:
            tqc = tqc_map[target_bits]
            backend.run(tqc, shots=SHOTS).result()
        else:
            qc = build_grover_circuit_for_target(target_bits)
            tqc = transpile(qc, backend, optimization_level=1)
            backend.run(tqc, shots=SHOTS).result()
        t_quantum = time.perf_counter() - start_q

        if not q_for_child.empty():
            _, t_classical = q_for_child.get_nowait()
        else:
            _, t_classical = q_for_child.get()

        if p_class.is_alive():
            p_class.terminate()
        p_class.join(timeout=0.1)

        if abs(t_classical - t_quantum) / max(t_classical, t_quantum, 1e-12) < TIE_REL_TOL:
            ties += 1
            result_str = "tie"
        elif t_classical < t_quantum:
            classical_wins += 1
            result_str = "classical"
        else:
            quantum_wins += 1
            result_str = "quantum"

        if r % max(1, rounds // 10) == 0 or r == rounds:
            print(f"Round {r}/{rounds} | winner: {result_str} | t_class={t_classical:.4f}s t_quantum={t_quantum:.4f}s")

    print("\n=== FINAL RESULTS ===")
    print(f"Rounds: {rounds}")
    print(f"TARGET_OPS per round: {target_ops:,}")
    print(f"Classical wins: {classical_wins}")
    print(f"Quantum wins:   {quantum_wins}")
    print(f"Ties:           {ties}")

if __name__ == "__main__":
    print("Starting time-based race")
    print(f"n={n}, expected_candidates={expected_candidates}, iter_per_candidate={iter_per_candidate}")
    run_time_race()
