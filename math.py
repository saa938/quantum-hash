"""
Backtracking solver for the "circles + fixed row/column sums" puzzle.

Key clarifications (implements the problem statement exactly):
 - Grid is given by row_patterns: '.' = white cell (may place 1..9), '#' = shaded (must remain empty).
 - No digit may repeat in any row or column.
 - Every ROW must sum to the same target R (set R_fixed or leave None to search).
 - Every COLUMN must sum to the same target C (set C_fixed or leave None to search).
 - Circles are specified by (digits_multiset, intersection_row, intersection_col).
   The circle sits on a grid intersection (between cells). The four touching cells are the
   cells that share that intersection: (ir-1,ic-1),(ir-1,ic),(ir,ic-1),(ir,ic) if those
   coordinates are valid cell coordinates. **Only white cells among those are considered
   adjacent white cells.** The rule of the puzzle states that *the multiset of digits in
   the adjacent cells equals the digits printed in the circle*. Therefore the number of
   digits printed in a circle must equal the number of adjacent white cells. (If they
   mismatch the input is invalid for the puzzle.)
 - The solver first enumerates all valid placements of circle digits onto their adjacent
   white cells (treating each circle independently), prunes by row/column uniqueness and
   sum bounds, and then fills the remaining white cells row-by-row by backtracking.

How to use:
 - Edit R_fixed and C_fixed as needed (or set to None to search).
 - Edit row_patterns to match the grid (the user provided the 5x12 pattern already).
 - Provide circles as a list of (digits_list, inter_r, inter_c) where inter_r in 0..ROWS
   and inter_c in 0..COLS (these are intersection coordinates; e.g. (1,2) sits between
   cells). Coordinates are 0-indexed.
 - Run. The program prints the first found solution or reports failure.

This implementation corrects the "circle white cell logic" by requiring that the circle's
printed digits match exactly the digits placed into the adjacent white cells computed
from the intersection coordinates.
"""

from itertools import permutations, combinations
from collections import defaultdict
import sys

# ------------------ CONFIGURATION (edit as needed) ------------------
R_fixed = 24     # set to None to let solver search reasonable R values
C_fixed = 10     # set to None to let solver search reasonable C values

ROWS = 5
COLS = 12

row_patterns = [
    "............",
    "##..#.##..##",
    "....#.......",
    ".##.#..####.",
    "............"
]

# Circles specification:
# Each circle = (digits_list, intersection_row, intersection_col)
# intersection_row ranges 0..ROWS, intersection_col ranges 0..COLS
# Example format (fill with actual data for your puzzle):
# circles = [
#     ([1,3,9], 2, 4),
#     ([4,5], 3, 1),
#     ...
# ]
# NOTE: The number of printed digits MUST equal the number of ADJACENT WHITE CELLS
# that touch that intersection in the current row_patterns.
circles = [
    ([1,3,9], [(3,4),(3,5),(4,4),(4,5)]),
    ([3,6],    [(6,14),(6,15)]),
    ([4,5,6],  [(4,1),(4,2),(5,1)]),
    ([8,9],    [(6,10),(6,11)]),
    ([2,4],    [(3,12),(3,13),(4,12),(4,13)]),
    ([1,3,7],  [(3,14),(3,15),(4,14)]),
    ([1],      [(6,3)]),
    ([3,7,8],  [(1,7),(1,8),(2,7),(2,8)])
]

# --------------------------------------------------------------------

# Basic validation
if len(row_patterns) != ROWS:
    raise ValueError("row_patterns length != ROWS")
for rline in row_patterns:
    if len(rline) != COLS:
        raise ValueError("each row pattern string must have length COLS")

# Build white cell set and helper maps
white_cells = set((r,c) for r in range(ROWS) for c in range(COLS) if row_patterns[r][c]=='.')
cells_by_row = {r: [c for (rr,c) in white_cells if rr==r] for r in range(ROWS)}
cells_by_col = {c: [r for (r,cc) in white_cells if cc==c] for c in range(COLS)}

# Function to get adjacent white cells for a circle placed at intersection (ir,ic)
def adjacent_white_cells_from_intersection(ir, ic):
    """Return list of white-cell coordinates (r,c) that touch intersection (ir,ic)."""
    adj = []
    # neighbor offsets relative to intersection: (-1,-1),(-1,0),(0,-1),(0,0)
    for dr in (-1,0):
        for dc in (-1,0):
            rr = ir + dr
            cc = ic + dc
            if 0 <= rr < ROWS and 0 <= cc < COLS and row_patterns[rr][cc] == '.':
                adj.append((rr, cc))
    return adj

# Validate circles: compute adjacent white cells and ensure counts match printed digits
circle_cells = []  # will contain tuples (digits_list, [list of adjacent white cells])
for idx, entry in enumerate(circles):
    digits, ir, ic = entry
    adj = adjacent_white_cells_from_intersection(ir, ic)
    # According to puzzle text, the circle gives all the digits in the cells that touch the circle.
    # That implies the number of digits printed == number of adjacent white cells (each must be filled).
    if len(digits) != len(adj):
        raise ValueError(
            f"Circle {idx} at intersection ({ir},{ic}) has {len(digits)} printed digits "
            f"but {len(adj)} adjacent white cells {adj}. They must match per the rules."
        )
    circle_cells.append((list(digits), adj))

# Helper: generate assignments for a circle (assign the circle's multiset to the adjacent white cells)
def generate_circle_assignments(digits, adj_cells):
    """
    digits: multiset list, len=m
    adj_cells: list of exactly m white cell coords
    returns list of assignments: each assignment is list of ((r,c), digit)
    """
    m = len(digits)
    # assign digits to the cells in all distinct permutations (dedupe using set for perms with duplicates)
    assignments = []
    for perm in set(permutations(digits, m)):
        assignments.append(list(zip(adj_cells, perm)))
    return assignments

# Precompute circle option lists
circle_options = [generate_circle_assignments(d, cells) for (d,cells) in circle_cells]

# Quick infeasibility check
for i, opts in enumerate(circle_options):
    if not opts:
        print(f"No assignments possible for circle {i}; aborting.")
        sys.exit(1)

# Utility: compute maximum additional sum possible in a column given currently used digits in that column
def max_additional_for_column(col_idx, cused, grid):
    rem_rows = [r for r in range(ROWS) if (r,col_idx) in white_cells and grid[r][col_idx] is None]
    if not rem_rows:
        return 0
    avail = [d for d in range(9,0,-1) if d not in cused[col_idx]]
    return sum(avail[:len(rem_rows)])

# Main solver: assign circle options first (DFS), then fill remaining cells row-by-row
def solve(R_target=None, C_target=None):
    # If R/C not given, generate reasonable candidate ranges
    max_row_white = max(len(cells_by_row[r]) for r in range(ROWS))
    max_col_white = max(len(cells_by_col[c]) for c in range(COLS))
    R_candidates = [R_target] if R_target is not None else list(range(0, sum(range(9, 9-max_row_white, -1)) + 1))
    C_candidates = [C_target] if C_target is not None else list(range(0, sum(range(9, 9-max_col_white, -1)) + 1))

    # initial grid state
    base_grid = [[None]*COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if row_patterns[r][c] == '#':
                base_grid[r][c] = None  # shaded; stays None and will not be filled

    # Try each candidate pair R,C
    for R_try in R_candidates:
        for C_try in C_candidates:
            result = try_with_R_C(R_try, C_try, base_grid)
            if result is not None:
                return (R_try, C_try, result)
    return None

def try_with_R_C(R_try, C_try, base_grid):
    # state
    grid = [row[:] for row in base_grid]
    rused = [set() for _ in range(ROWS)]
    cused = [set() for _ in range(COLS)]
    rsum = [0]*ROWS
    csum = [0]*COLS

    # DFS assign circles
    def dfs_assign_circle(ci):
        if ci == len(circle_options):
            # proceed to fill the rest
            ok, final_grid = fill_remaining(grid, rused, cused, rsum, csum, R_try, C_try)
            return final_grid if ok else None

        for opt in circle_options[ci]:
            conflict = False
            changed = []
            for (r,c), d in opt:
                if grid[r][c] is not None and grid[r][c] != d:
                    conflict = True; break
                if d in rused[r] or d in cused[c]:
                    conflict = True; break
                if rsum[r] + d > R_try or csum[c] + d > C_try:
                    conflict = True; break
            if conflict:
                continue
            # apply
            for (r,c), d in opt:
                if grid[r][c] is None:
                    grid[r][c] = d
                    rused[r].add(d); cused[c].add(d)
                    rsum[r] += d; csum[c] += d
                    changed.append((r,c,d))
            # column feasibility prune
            col_bad = False
            for col in range(COLS):
                if csum[col] > C_try:
                    col_bad = True; break
                max_rem = max_additional_for_column(col, cused, grid)
                if csum[col] + max_rem < C_try:
                    col_bad = True; break
            if not col_bad:
                res = dfs_assign_circle(ci+1)
                if res is not None:
                    return res
            # undo
            for (rr,cc,d) in changed:
                grid[rr][cc] = None
                rused[rr].remove(d); cused[cc].remove(d)
                rsum[rr] -= d; csum[cc] -= d
        return None

    return dfs_assign_circle(0)

def fill_remaining(grid, rused, cused, rsum, csum, R_target, C_target):
    # Prepare free positions per row
    row_free = {r: [c for c in range(COLS) if (r,c) in white_cells and grid[r][c] is None] for r in range(ROWS)}
    # order rows: fewest free cells first, and larger remaining sum
    row_order = sorted(range(ROWS), key=lambda r: (len(row_free[r]), -(R_target - rsum[r])))

    digits = list(range(1,10))

    def backtrack_row(idx):
        if idx == len(row_order):
            # verify column sums
            if all(rsum[r] == R_target for r in range(ROWS)) and all(csum[c] == C_target for c in range(COLS)):
                return True
            return False
        r = row_order[idx]
        positions = row_free[r]
        need = R_target - rsum[r]
        if need < 0:
            return False
        if not positions:
            return need == 0 and backtrack_row(idx+1)

        possible_row_digits = [d for d in digits if d not in rused[r]]
        max_k = min(len(positions), len(possible_row_digits))

        # k=0 allowed only if need==0
        for k in range(0, max_k+1):
            if k == 0:
                if need != 0: continue
                if backtrack_row(idx+1):
                    return True
                continue
            # quick bounds
            if len(possible_row_digits) < k:
                continue
            min_sum = sum(sorted(possible_row_digits)[:k])
            max_sum = sum(sorted(possible_row_digits, reverse=True)[:k])
            if not (min_sum <= need <= max_sum):
                continue
            for comb in combinations(possible_row_digits, k):
                if sum(comb) != need:
                    continue
                for perm in set(permutations(comb)):
                    # try to place perm into positions (we'll place digits into first k positions of 'positions' list;
                    # remaining positions in that row stay empty)
                    conflict = False
                    changed = []
                    for col, d in zip(positions, perm):
                        if d in cused[col]:
                            conflict = True; break
                        if csum[col] + d > C_target:
                            conflict = True; break
                    if conflict:
                        continue
                    # apply
                    for col, d in zip(positions, perm):
                        grid[r][col] = d
                        rused[r].add(d); cused[col].add(d)
                        rsum[r] += d; csum[col] += d
                        changed.append((r,col,d))
                    # column-level feasibility check
                    col_ok = True
                    for col_idx in range(COLS):
                        if csum[col_idx] > C_target:
                            col_ok = False; break
                        max_rem = max_additional_for_column(col_idx, cused, grid)
                        if csum[col_idx] + max_rem < C_target:
                            col_ok = False; break
                    if col_ok:
                        if backtrack_row(idx+1):
                            return True
                    # undo
                    for (rr,cc,d) in changed:
                        grid[rr][cc] = None
                        rused[rr].remove(d); cused[cc].remove(d)
                        rsum[rr] -= d; csum[cc] -= d
        return False

    ok = backtrack_row(0)
    return ok, [row[:] for row in grid] if ok else (False, None)

# Run solver
if __name__ == "__main__":
    if not circles:
        print("No circles provided in 'circles' list. Fill 'circles' with (digits_list, inter_r, inter_c) entries and run.")
        sys.exit(0)

    sol = solve(R_fixed, C_fixed)
    if sol is None:
        print("No solution found for provided R/C and circle mapping.")
    else:
        R_sol, C_sol, grid_sol = sol
        print(f"Solution found for R={R_sol}, C={C_sol}:")
        for r in range(ROWS):
            out = []
            for c in range(COLS):
                if row_patterns[r][c] == '#':
                    out.append('#')
                else:
                    v = grid_sol[r][c]
                    out.append(str(v) if v is not None else '.')
            print(' '.join(out))
