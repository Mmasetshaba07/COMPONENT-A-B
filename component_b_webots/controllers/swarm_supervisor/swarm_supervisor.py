"""Three-agent search-and-rescue demonstration for Webots.

A Webots Supervisor implements a transparent kinematic model, consensus, discrete
PSO task allocation, A* route planning, and a trained small neural network that
selects cautious/cruise movement. It is a simulation model, not a hardware controller.
"""
from controller import Supervisor
import csv
import math
import os
import random

RNG = random.Random(27)
DT = 0.064
GRID = 0.25
BOUNDS = (-8.0, 8.0, -6.0, 6.0)  # x_min, x_max, z_min, z_max
# x_min, x_max, z_min, z_max, inflation already included for conservative planning
OBSTACLES = [(-1.95, -0.05, -1.85, 1.85),
             (1.55, 4.45, -3.05, -1.35),
             (1.30, 2.90, 1.10, 3.70),
             (-4.95, -3.05, -3.35, -1.65)]


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def inside_obstacle(x, z):
    return any(x0 <= x <= x1 and z0 <= z <= z1 for x0, x1, z0, z1 in OBSTACLES)


def clearance(x, z):
    best = 99.0
    for x0, x1, z0, z1 in OBSTACLES:
        dx = max(x0 - x, 0.0, x - x1)
        dz = max(z0 - z, 0.0, z - z1)
        best = min(best, math.hypot(dx, dz))
    # account for the courtyard perimeter
    best = min(best, x - BOUNDS[0], BOUNDS[1] - x, z - BOUNDS[2], BOUNDS[3] - z)
    return max(0.0, best)


def cell(point):
    return (round(point[0] / GRID), round(point[1] / GRID))


def cell_point(c):
    return (c[0] * GRID, c[1] * GRID)


def astar(start, goal):
    """Eight-connected A* on a 0.25 m occupancy grid."""
    s, g = cell(start), cell(goal)
    xmin, xmax, zmin, zmax = BOUNDS
    max_x, min_x = int(xmax / GRID), int(xmin / GRID)
    max_z, min_z = int(zmax / GRID), int(zmin / GRID)
    # 8 neighbors; avoid diagonal corner-cutting by checking the adjacent cardinals.
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    frontier = [(0.0, s)]
    came = {s: None}
    cost = {s: 0.0}
    while frontier:
        frontier.sort(key=lambda v: v[0])
        _, cur = frontier.pop(0)
        if cur == g:
            break
        for dx, dz, step_cost in moves:
            nxt = (cur[0] + dx, cur[1] + dz)
            px, pz = cell_point(nxt)
            if not (min_x <= nxt[0] <= max_x and min_z <= nxt[1] <= max_z):
                continue
            if inside_obstacle(px, pz):
                continue
            if dx and dz and (inside_obstacle(*cell_point((cur[0] + dx, cur[1]))) or
                              inside_obstacle(*cell_point((cur[0], cur[1] + dz)))):
                continue
            new_cost = cost[cur] + step_cost
            if nxt not in cost or new_cost < cost[nxt]:
                cost[nxt] = new_cost
                came[nxt] = cur
                heuristic = math.hypot(g[0] - nxt[0], g[1] - nxt[1])
                frontier.append((new_cost + heuristic, nxt))
    if g not in came:
        return [start, goal]
    route = []
    cursor = g
    while cursor is not None:
        route.append(cell_point(cursor))
        cursor = came[cursor]
    route.reverse()
    route[0] = start
    route[-1] = goal
    # Simplify a route by retaining direction-change points only.
    if len(route) <= 2:
        return route
    simplified = [route[0]]
    old_dir = None
    for i in range(1, len(route)):
        d = (round(route[i][0] - route[i-1][0], 4), round(route[i][1] - route[i-1][1], 4))
        if old_dir is not None and d != old_dir:
            simplified.append(route[i-1])
        old_dir = d
    simplified.append(route[-1])
    return simplified


# A tiny one-hidden-layer neural network; trained on synthetic examples generated
# from a transparent cautious-speed rule. Model features: normalized goal distance,
# obstacle clearance, and route progress. Outputs: cautious / cruise.
class TinyNetwork:
    def __init__(self, inputs=3, hidden=6, outputs=2):
        self.w1 = [[RNG.uniform(-0.5, 0.5) for _ in range(inputs)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[RNG.uniform(-0.5, 0.5) for _ in range(hidden)] for _ in range(outputs)]
        self.b2 = [0.0] * outputs

    @staticmethod
    def sigmoid(x):
        x = max(-30.0, min(30.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, x):
        h = [self.sigmoid(sum(w * v for w, v in zip(row, x)) + b)
             for row, b in zip(self.w1, self.b1)]
        logits = [sum(w * v for w, v in zip(row, h)) + b for row, b in zip(self.w2, self.b2)]
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        total = sum(exps)
        return h, [v / total for v in exps]

    def train(self, examples, epochs=100, lr=0.28):
        for _ in range(epochs):
            RNG.shuffle(examples)
            for x, label in examples:
                h, probs = self.forward(x)
                delta2 = [probs[j] - (1.0 if j == label else 0.0) for j in range(2)]
                delta1 = [h[k] * (1.0 - h[k]) * sum(self.w2[j][k] * delta2[j] for j in range(2))
                          for k in range(len(h))]
                for j in range(2):
                    for k in range(len(h)):
                        self.w2[j][k] -= lr * delta2[j] * h[k]
                    self.b2[j] -= lr * delta2[j]
                for k in range(len(h)):
                    for i in range(3):
                        self.w1[k][i] -= lr * delta1[k] * x[i]
                    self.b1[k] -= lr * delta1[k]

    def predict(self, x):
        return max(range(2), key=lambda j: self.forward(x)[1][j])


def oracle_label(distance, clear):
    return 0 if distance < 1.1 or clear < 0.65 else 1


def train_network():
    data = []
    for _ in range(450):
        dist = RNG.uniform(0.0, 8.0)
        clear = RNG.uniform(0.0, 3.0)
        progress = RNG.random()
        label = oracle_label(dist, clear)
        data.append(([dist / 8.0, clear / 3.0, progress], label))
    net = TinyNetwork()
    net.train(data)
    return net


def solve_assignment(starts, goals, particles=32, iterations=55):
    """Random-key discrete PSO: each particle's sorted keys encode a permutation."""
    n = len(starts)
    def decode(keys):
        return sorted(range(n), key=lambda i: keys[i])
    def objective(keys):
        order = decode(keys)
        return sum(euclidean(starts[r], goals[order[r]]) for r in range(n))
    swarm = []
    for _ in range(particles):
        pos = [RNG.random() for _ in range(n)]
        swarm.append({"p": pos, "v": [RNG.uniform(-0.1, 0.1) for _ in range(n)],
                      "best": pos[:], "score": objective(pos)})
    best = min(swarm, key=lambda item: item["score"])["best"][:]
    best_score = objective(best)
    for _ in range(iterations):
        for p in swarm:
            score = objective(p["p"])
            if score < p["score"]:
                p["score"], p["best"] = score, p["p"][:]
            if score < best_score:
                best_score, best = score, p["p"][:]
        for p in swarm:
            for d in range(n):
                p["v"][d] = (0.55 * p["v"][d] +
                             1.35 * RNG.random() * (p["best"][d] - p["p"][d]) +
                             1.45 * RNG.random() * (best[d] - p["p"][d]))
                p["v"][d] = max(-0.25, min(0.25, p["v"][d]))
                p["p"][d] = max(0.001, min(0.999, p["p"][d] + p["v"][d]))
    return decode(best), best_score


def main():
    supervisor = Supervisor()
    step_ms = int(supervisor.getBasicTimeStep())
    step_seconds = step_ms / 1000.0
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(root, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    robot_defs = ["ROBOT_1", "ROBOT_2", "ROBOT_3"]
    victim_defs = ["VICTIM_1", "VICTIM_2", "VICTIM_3"]
    robots = [supervisor.getFromDef(name) for name in robot_defs]
    victims = [supervisor.getFromDef(name) for name in victim_defs]
    if any(node is None for node in robots + victims):
        raise RuntimeError("A DEF robot or victim target is missing from the world file")
    robot_fields = [node.getField("translation") for node in robots]
    robot_rotations = [node.getField("rotation") for node in robots]
    victim_fields = [node.getField("translation") for node in victims]
    starts = [(field.getSFVec3f()[0], field.getSFVec3f()[2]) for field in robot_fields]
    goals = [(field.getSFVec3f()[0], field.getSFVec3f()[2]) for field in victim_fields]
    assignment, assignment_cost = solve_assignment(starts, goals)
    net = train_network()
    routes = [astar(starts[i], goals[assignment[i]]) for i in range(3)]
    route_indices = [1 if len(route) > 1 else 0 for route in routes]
    goal_distances = [max(0.01, euclidean(starts[i], goals[assignment[i]])) for i in range(3)]
    done = [False] * 3
    consensus = [0.0] * 3
    neighbor_graph = {0: [1, 2], 1: [0], 2: [0]}  # leader-centered star
    rows_written = 0
    sim_time = 0.0
    max_steps = 4200
    log_path = os.path.join(out_dir, "trajectory.csv")
    print("PSO assignment (robot -> victim):", [assignment[i] + 1 for i in range(3)])
    print("PSO route-cost objective: %.2f m; consensus topology: leader-centered star" % assignment_cost)
    print("Training neural speed policy on 450 generated examples ...")

    with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["sim_time", "robot", "x", "z", "target",
            "progress", "consensus", "decision", "oracle", "clearance", "distance_to_target",
            "path_error", "found"])
        writer.writeheader()
        for tick in range(max_steps):
            if supervisor.step(step_ms) == -1:
                break
            sim_time += step_seconds
            local_progress = []
            distances = []
            for i in range(3):
                pos3 = robot_fields[i].getSFVec3f()
                x, z = pos3[0], pos3[2]
                target = goals[assignment[i]]
                dgoal = euclidean((x, z), target)
                distances.append(dgoal)
                local_progress.append(max(0.0, min(1.0, 1.0 - dgoal / goal_distances[i])))
                if not done[i] and dgoal <= 0.32:
                    done[i] = True
                    victim_fields[assignment[i]].setSFVec3f([target[0], -4.0, target[1]])
                    near = clearance(x, z)
                    decision = net.predict([min(1.0, dgoal / 8.0),
                                            min(1.0, near / 3.0), local_progress[-1]])
                    oracle = oracle_label(dgoal, near)
                    writer.writerow({"sim_time": "%.3f" % sim_time,
                        "robot": "Robot %d" % (i+1), "x": "%.4f" % x, "z": "%.4f" % z,
                        "target": "Victim %d" % (assignment[i]+1),
                        "progress": "%.4f" % local_progress[-1], "consensus": "%.4f" % consensus[i],
                        "decision": decision, "oracle": oracle, "clearance": "%.4f" % near,
                        "distance_to_target": "%.4f" % dgoal, "path_error": "0.0000", "found": 1})
                    rows_written += 1
                if done[i]:
                    continue
                route = routes[i]
                idx = route_indices[i]
                while idx < len(route) - 1 and euclidean((x, z), route[idx]) < 0.13:
                    idx += 1
                route_indices[i] = idx
                waypoint = route[idx] if idx < len(route) else target
                path_error = euclidean((x, z), waypoint)
                near = clearance(x, z)
                features = [min(1.0, dgoal / 8.0), min(1.0, near / 3.0), local_progress[-1]]
                decision = net.predict(features)  # 0=cautious, 1=cruise
                oracle = oracle_label(dgoal, near)
                speed = 0.045 if decision == 0 else 0.075
                dx, dz = waypoint[0] - x, waypoint[1] - z
                length = math.hypot(dx, dz)
                if length > 1e-6:
                    move = min(speed, length)
                    nx, nz = x + dx / length * move, z + dz / length * move
                    # A* protects against obstacles; this final guard catches numerical drift.
                    if not inside_obstacle(nx, nz):
                        robot_fields[i].setSFVec3f([nx, 0.14, nz])
                        yaw = math.atan2(dx, dz)
                        robot_rotations[i].setSFRotation([0, 1, 0, yaw])
                if tick % 5 == 0:
                    writer.writerow({"sim_time": "%.3f" % sim_time, "robot": "Robot %d" % (i+1),
                        "x": "%.4f" % x, "z": "%.4f" % z, "target": "Victim %d" % (assignment[i]+1),
                        "progress": "%.4f" % local_progress[-1], "consensus": "%.4f" % consensus[i],
                        "decision": decision, "oracle": oracle, "clearance": "%.4f" % near,
                        "distance_to_target": "%.4f" % dgoal, "path_error": "%.4f" % path_error,
                        "found": int(done[i])})
                    rows_written += 1
            # One synchronous average-consensus update over the star graph.
            old = consensus[:]
            for i in range(3):
                consensus[i] = old[i] + 0.12 * sum(old[j] - old[i] for j in neighbor_graph[i])
                consensus[i] = 0.88 * consensus[i] + 0.12 * local_progress[i]
            if tick % 200 == 0:
                csv_file.flush()
                print("t=%5.1fs | found %d/3 | consensus %.2f %.2f %.2f" %
                      (sim_time, sum(done), consensus[0], consensus[1], consensus[2]))
            if all(done):
                break
        csv_file.flush()

    summary = os.path.join(out_dir, "run_summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("Multi-Agent Search and Rescue simulation\n")
        f.write("PSO assignment robot->victim: %s\n" % ", ".join(str(v+1) for v in assignment))
        f.write("PSO assignment distance objective: %.3f m\n" % assignment_cost)
        f.write("Targets found: %d/3\n" % sum(done))
        f.write("Simulation time: %.2f s\n" % sim_time)
        f.write("Rows logged: %d\n" % rows_written)
        f.write("Consensus communication graph: leader-centered star (Robot 1 connected to Robots 2 and 3)\n")
        f.write("Controller simulation: kinematic Supervisor pose updates; no wheel dynamics or hardware claims.\n")
    print("Mission complete: %d/3 targets found; log saved to %s" % (sum(done), log_path))
    print("Run python analysis.py from the project root to generate plots and evaluation.csv")


if __name__ == "__main__":
    main()
