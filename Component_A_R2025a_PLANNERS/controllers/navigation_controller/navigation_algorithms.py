"""Pure-Python occupancy mapping, grid planning and PID utilities.

This module has no Webots dependency, so both planners and control math can be
tested on a regular Python installation before opening the world.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

GRID = 50
CELL_M = 0.24
MAP_ORIGIN = -6.0
# Circumscribed chassis/wheel/camera footprint plus a small tracking margin.
ROBOT_RADIUS_M = 0.20
START = (3, 3)
# This cell centre (x=4.68, z=1.56) is within 4 cm of the design's goal marker.
GOAL = (31, 44)
OBSTACLES = [
    (5, 21, 8, 10),
    (9, 11, 14, 28),
    (16, 31, 20, 22),
    (19, 21, 29, 43),
    (27, 40, 7, 9),
    (31, 33, 14, 27),
    (35, 47, 37, 39),
    (39, 41, 11, 20),
    (5, 7, 36, 44),
]
NEIGHBOURS = (
    (-1, -1, math.sqrt(2)),
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2)),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, math.sqrt(2)),
    (1, 0, 1.0),
    (1, 1, math.sqrt(2)),
)


@dataclass(frozen=True)
class PlanResult:
    algorithm: str
    path: tuple[tuple[int, int], ...]
    cost_cells: float
    explored: int

    @property
    def distance_m(self) -> float:
        return self.cost_cells * CELL_M


def select_startup_planner(arguments, environment_value=None):
    """World controllerArgs take precedence over the optional test environment."""
    option = arguments[0] if arguments else (environment_value or "astar")
    normalized = option.strip().lower()
    if normalized == "dijkstra":
        return "Dijkstra"
    if normalized in ("astar", "a*"):
        return "A*"
    raise ValueError("Unknown planner %r; choose astar or dijkstra" % option)


def make_occupancy_map(inflate: bool = True) -> list[list[bool]]:
    """Create the 50x50 map used by the world and inflate obstacles by footprint."""
    occupied = [[False] * GRID for _ in range(GRID)]
    for i in range(GRID):
        occupied[0][i] = occupied[GRID - 1][i] = True
        occupied[i][0] = occupied[i][GRID - 1] = True
    for r0, r1, c0, c1 in OBSTACLES:
        for row in range(r0, r1):
            for col in range(c0, c1):
                occupied[row][col] = True
    if not inflate:
        return occupied

    radius_cells = math.ceil(ROBOT_RADIUS_M / CELL_M + 0.5)
    offsets = [
        (dr, dc)
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        # Measure from a candidate cell centre to the occupied cell's edges,
        # not its centre. A neighbouring cell is only 0.12 m from an obstacle
        # edge even though the cell centres are 0.24 m apart.
        if math.hypot(
            max(0.0, abs(dr) - 0.5) * CELL_M,
            max(0.0, abs(dc) - 0.5) * CELL_M,
        ) <= ROBOT_RADIUS_M + 1e-9
    ]
    inflated = [row[:] for row in occupied]
    for row in range(GRID):
        for col in range(GRID):
            if not occupied[row][col]:
                continue
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < GRID and 0 <= nc < GRID:
                    inflated[nr][nc] = True
    return inflated


def plan_path(
    occupied: list[list[bool]],
    start: tuple[int, int] = START,
    goal: tuple[int, int] = GOAL,
    algorithm: str = "astar",
) -> PlanResult:
    """Plan an 8-connected shortest route with Dijkstra or Euclidean A*."""
    mode = algorithm.lower().replace("*", "star")
    if mode not in {"astar", "dijkstra"}:
        raise ValueError("algorithm must be 'astar' or 'dijkstra'")
    for label, (row, col) in (("start", start), ("goal", goal)):
        if not (0 <= row < GRID and 0 <= col < GRID):
            raise ValueError(f"{label} cell is outside the map: {(row, col)}")
        if occupied[row][col]:
            raise ValueError(f"{label} cell is occupied: {(row, col)}")

    frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    cost = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    explored = 0
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost > cost.get(current, math.inf) + 1e-10:
            continue
        explored += 1
        if current == goal:
            break
        row, col = current
        for dr, dc, step_cost in NEIGHBOURS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < GRID and 0 <= nc < GRID) or occupied[nr][nc]:
                continue
            if dr and dc and (occupied[row + dr][col] or occupied[row][col + dc]):
                continue
            neighbour = (nr, nc)
            new_cost = current_cost + step_cost
            if new_cost >= cost.get(neighbour, math.inf):
                continue
            cost[neighbour] = new_cost
            parent[neighbour] = current
            heuristic = (
                math.hypot(goal[0] - nr, goal[1] - nc) if mode == "astar" else 0.0
            )
            heapq.heappush(frontier, (new_cost + heuristic, new_cost, neighbour))

    if goal not in parent:
        raise RuntimeError(f"{algorithm} could not find a path from {start} to {goal}")
    path: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return PlanResult(
        "A*" if mode == "astar" else "Dijkstra",
        tuple(path),
        cost[goal],
        explored,
    )


def world_from_cell(cell: tuple[int, int]) -> tuple[float, float]:
    """Convert (row, column) into the centre (x, z) of a map cell."""
    row, col = cell
    return (
        MAP_ORIGIN + (col + 0.5) * CELL_M,
        MAP_ORIGIN + (row + 0.5) * CELL_M,
    )


def cell_from_world(x: float, z: float) -> tuple[int, int]:
    """Convert world-plane coordinates to a clamped (row, column) cell."""
    col = int(math.floor((x - MAP_ORIGIN) / CELL_M))
    row = int(math.floor((z - MAP_ORIGIN) / CELL_M))
    return (
        min(GRID - 1, max(0, row)),
        min(GRID - 1, max(0, col)),
    )


def nearest_free_cell(
    occupied: list[list[bool]], cell: tuple[int, int]
) -> tuple[int, int]:
    """Find the closest traversable cell when localization lands in an inflated cell."""
    if not occupied[cell[0]][cell[1]]:
        return cell
    candidates = [
        (math.hypot(row - cell[0], col - cell[1]), (row, col))
        for row in range(GRID)
        for col in range(GRID)
        if not occupied[row][col]
    ]
    if not candidates:
        raise RuntimeError("the occupancy map contains no traversable cells")
    return min(candidates)[1]


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def lidar_safe_commands(
    linear_speed: float,
    heading_rate: float,
    front_range: float,
) -> tuple[float, float, str]:
    """Stop forward motion near obstacles while retaining steering authority."""
    if front_range < 0.24:
        return 0.0, heading_rate, "turn-only"
    if front_range < 0.60:
        return linear_speed * 0.25, heading_rate, "slow"
    return linear_speed, heading_rate, "clear"


class PID:
    """Discrete PID with integral clamping and derivative-on-error."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float,
        output_limit: float,
        angular: bool = False,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)
        self.output_limit = abs(output_limit)
        self.angular = angular
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            raise ValueError("PID timestep must be positive")
        if self.angular:
            error = wrap_angle(error)
        self.integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral + error * dt),
        )
        derivative = 0.0
        if self.previous_error is not None:
            difference = error - self.previous_error
            if self.angular:
                difference = wrap_angle(difference)
            derivative = difference / dt
        self.previous_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))
