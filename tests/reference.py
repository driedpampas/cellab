"""Independent, deliberately simple implementations of the example automata.

These are written the slow, obvious way: plain Python loops over every cell,
with no NumPy tricks, no rule language and no shared code with the engine. The
tests run the engine and these side by side on random grids; if they ever
disagree, one of them is wrong.
"""

VON_NEUMANN = [(-1, 0), (0, -1), (0, 1), (1, 0)]
MOORE = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def neighbours(grid, r, c, offsets, wrap, edge=None):
    rows, cols = len(grid), len(grid[0])
    values = []
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if wrap:
            values.append(grid[rr % rows][cc % cols])
        elif 0 <= rr < rows and 0 <= cc < cols:
            values.append(grid[rr][cc])
        else:
            values.append(edge)
    return values


def each_cell(grid, new_value):
    return [[new_value(r, c) for c in range(len(grid[0]))] for r in range(len(grid))]


def elementary(row, rule, wrap=True, edge=0):
    n, out = len(row), []
    for i in range(n):
        left = row[i - 1] if i > 0 else (row[-1] if wrap else edge)
        right = row[i + 1] if i < n - 1 else (row[0] if wrap else edge)
        out.append((rule >> (4 * left + 2 * row[i] + right)) & 1)
    return out


def traffic(row):                      # 0 = EMPTY, 1 = CAR, circular road, cars move right
    n = len(row)
    return [1 if (row[i] == 1 and row[(i + 1) % n] == 1) or (row[i] == 0 and row[i - 1] == 1) else 0
            for i in range(n)]


def forest_fire(grid):                 # 0 EMPTY, 1 TREE, 2 FIRE; edges are EMPTY
    def new(r, c):
        me = grid[r][c]
        if me == 2:
            return 0
        if me == 1 and 2 in neighbours(grid, r, c, VON_NEUMANN, wrap=False, edge=0):
            return 2
        return me
    return each_cell(grid, new)


def brians_brain(grid):                # 0 OFF, 1 FIRING, 2 DYING; wrap-around
    def new(r, c):
        me = grid[r][c]
        if me == 1:
            return 2
        if me == 2:
            return 0
        return 1 if neighbours(grid, r, c, MOORE, wrap=True).count(1) == 2 else 0
    return each_cell(grid, new)


def game_of_life(grid):                # 0 DEAD, 1 ALIVE; wrap-around
    def new(r, c):
        alive = neighbours(grid, r, c, MOORE, wrap=True).count(1)
        return 1 if alive == 3 or (grid[r][c] == 1 and alive == 2) else 0
    return each_cell(grid, new)


def sandpile(grid):                    # grains; edges hold 0 and swallow grains
    def new(r, c):
        me = grid[r][c]
        gained = sum(1 for v in neighbours(grid, r, c, VON_NEUMANN, wrap=False, edge=0) if v >= 4)
        return me - (4 if me >= 4 else 0) + gained
    return each_cell(grid, new)


def cyclic(grid, levels=4, threshold=2):
    def new(r, c):
        nxt = (grid[r][c] + 1) % levels
        return nxt if neighbours(grid, r, c, MOORE, wrap=True).count(nxt) >= threshold else grid[r][c]
    return each_cell(grid, new)


def voter(grid):                       # 0 A, 1 B; wrap-around; ties keep the old opinion
    def new(r, c):
        around = neighbours(grid, r, c, MOORE, wrap=True)
        a, b = around.count(0), around.count(1)
        return 0 if a > b else 1 if b > a else grid[r][c]
    return each_cell(grid, new)


def crystal_growth(grid, needed=1):    # 0 EMPTY, 1 CRYSTAL; edges EMPTY
    def new(r, c):
        if grid[r][c] == 0 and neighbours(grid, r, c, VON_NEUMANN, wrap=False, edge=0).count(1) == needed:
            return 1
        return grid[r][c]
    return each_cell(grid, new)


def epidemic(grid):                    # 0 S, 1-3 infected for 1/2/3 turns, 4 R; edges S
    def new(r, c):
        me = grid[r][c]
        if me == 0:
            infected = sum(1 for v in neighbours(grid, r, c, MOORE, wrap=False, edge=0) if v in (1, 2, 3))
            return 1 if infected >= 2 else 0
        return me + 1 if me in (1, 2, 3) else me
    return each_cell(grid, new)


def diffusion(grid):                   # average of self and 4 neighbours, wrap, 2 decimals
    def new(r, c):
        return round((grid[r][c] + sum(neighbours(grid, r, c, VON_NEUMANN, wrap=True))) / 5, 2)
    return each_cell(grid, new)
