import sys
import unittest
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "webots").is_dir():
    PROJECT_ROOT = PROJECT_ROOT / "webots"
CONTROLLER_DIR = PROJECT_ROOT / "controllers" / "navigation_controller"
sys.path.insert(0, str(CONTROLLER_DIR))

from navigation_algorithms import (  # noqa: E402
    GOAL,
    OBSTACLES,
    START,
    PID,
    cell_from_world,
    lidar_safe_commands,
    make_occupancy_map,
    plan_path,
    select_startup_planner,
    world_from_cell,
)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.grid = make_occupancy_map()

    def test_world_arguments_select_planner_without_keyboard(self):
        self.assertEqual(select_startup_planner([]), "A*")
        self.assertEqual(select_startup_planner(["dijkstra"]), "Dijkstra")
        self.assertEqual(select_startup_planner(["astar"]), "A*")
        self.assertEqual(select_startup_planner([], "dijkstra"), "Dijkstra")
        self.assertEqual(select_startup_planner(["dijkstra"], "astar"), "Dijkstra")
        self.assertEqual(select_startup_planner(["astar"], "dijkstra"), "A*")
        with self.assertRaises(ValueError):
            select_startup_planner(["invalid"])

    def test_dijkstra_and_astar_find_equal_cost_collision_free_routes(self):
        dijkstra = plan_path(self.grid, START, GOAL, "dijkstra")
        astar = plan_path(self.grid, START, GOAL, "astar")
        self.assertEqual(dijkstra.path[0], START)
        self.assertEqual(dijkstra.path[-1], GOAL)
        self.assertEqual(astar.path[0], START)
        self.assertEqual(astar.path[-1], GOAL)
        self.assertAlmostEqual(dijkstra.cost_cells, astar.cost_cells, places=9)
        self.assertLess(astar.explored, dijkstra.explored)
        for path in (dijkstra.path, astar.path):
            for row, col in path:
                self.assertFalse(self.grid[row][col])
            for (row1, col1), (row2, col2) in zip(path, path[1:]):
                self.assertLessEqual(max(abs(row2 - row1), abs(col2 - col1)), 1)
                if row1 != row2 and col1 != col2:
                    self.assertFalse(self.grid[row1][col2])
                    self.assertFalse(self.grid[row2][col1])

    def test_coordinate_conversions_match_start_and_goal_markers(self):
        self.assertEqual(cell_from_world(*world_from_cell(START)), START)
        self.assertEqual(cell_from_world(*world_from_cell(GOAL)), GOAL)
        goal_x, goal_z = world_from_cell(GOAL)
        self.assertLess(abs(goal_x - 4.65), 0.05)
        self.assertLess(abs(goal_z - 1.50), 0.10)

    def test_inflation_blocks_cells_too_close_to_first_barrier(self):
        raw = make_occupancy_map(inflate=False)
        inflated = make_occupancy_map()
        # Barrier 1 begins at row 5, column 8; row 4 is only 12 cm from it.
        self.assertFalse(raw[4][8])
        self.assertTrue(inflated[4][8])
        self.assertTrue(inflated[4][7])
        self.assertGreater(
            sum(sum(row) for row in inflated),
            sum(sum(row) for row in raw),
        )

    def test_pid_clamps_output_and_integral_and_resets(self):
        pid = PID(2.0, 1.0, 0.0, 0.1, 0.4)
        for _ in range(100):
            self.assertLessEqual(pid.update(1.0, 0.05), 0.4)
        self.assertLessEqual(abs(pid.integral), 0.1)
        pid.reset()
        self.assertEqual(pid.integral, 0.0)
        self.assertIsNone(pid.previous_error)

    def test_lidar_stop_preserves_turning_and_slows_forward_motion(self):
        self.assertEqual(lidar_safe_commands(0.2, 0.5, 0.23), (0.0, 0.5, "turn-only"))
        self.assertEqual(lidar_safe_commands(0.2, -0.5, 0.40), (0.05, -0.5, "slow"))
        self.assertEqual(lidar_safe_commands(0.2, 0.1, 1.0), (0.2, 0.1, "clear"))

    def test_webots_barrier_geometry_matches_planner_grid(self):
        world_file = CONTROLLER_DIR.parents[1] / "worlds" / "indoor_navigation.wbt"
        world = world_file.read_text(encoding="utf-8")
        pattern = re.compile(
            r"DEF OBSTACLE_(\d+) Solid \{\s*"
            r"translation ([\d.-]+) 0\.30 ([\d.-]+).*?"
            r"geometry Box \{ size ([\d.]+) 0\.60 ([\d.]+) \}",
            re.DOTALL,
        )
        found = []
        for match in pattern.finditer(world):
            _, x, z, size_x, size_z = match.groups()
            x, z, size_x, size_z = map(float, (x, z, size_x, size_z))
            c0 = round((x - size_x / 2 + 6.0) / 0.24)
            c1 = round((x + size_x / 2 + 6.0) / 0.24)
            r0 = round((z - size_z / 2 + 6.0) / 0.24)
            r1 = round((z + size_z / 2 + 6.0) / 0.24)
            found.append((r0, r1, c0, c1))
        self.assertEqual(len(found), 9)
        self.assertEqual(found, OBSTACLES)


class WorldLoadingTests(unittest.TestCase):
    def test_world_delimiters_are_nested_not_just_equal_in_number(self):
        world = (PROJECT_ROOT / "worlds" / "indoor_navigation.wbt").read_text()
        # Ignore comments and string contents, which may contain delimiters.
        tokens = re.finditer(r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}\[\]]', world)
        stack = []
        for token in tokens:
            value = token.group()
            if value in ("{", "["):
                stack.append(value)
            elif value in ("}", "]"):
                self.assertTrue(stack, f"Unexpected {value} at offset {token.start()}")
                self.assertEqual(
                    stack.pop(), {"}": "{", "]": "["}[value],
                    f"Incorrect nesting near line {world.count(chr(10), 0, token.start()) + 1}",
                )
        self.assertEqual(stack, [])

    def test_world_has_no_undeclared_protos_and_targets_r2025a(self):
        world = (PROJECT_ROOT / "worlds" / "indoor_navigation.wbt").read_text()
        self.assertEqual(world.splitlines()[0], "#VRML_SIM R2025a utf8")
        self.assertIn('coordinateSystem "NUE"', world)
        builtins = {
            "WorldInfo", "Viewpoint", "Background", "DirectionalLight",
            "Solid", "Shape", "PBRAppearance", "Box", "Transform", "Cylinder",
            "Sphere", "Robot", "GPS", "Compass", "Camera", "Recognition",
            "Lidar", "HingeJoint", "HingeJointParameters", "RotationalMotor",
            "PositionSensor", "Physics",
        }
        used = set(re.findall(r"\b([A-Z]\w*)\s*\{", world))
        self.assertFalse(used - builtins, f"Unresolved nodes: {used - builtins}")

    def test_wheel_joints_use_forward_rolling_axes(self):
        world = (PROJECT_ROOT / "worlds" / "indoor_navigation.wbt").read_text()
        axes = re.findall(
            r"HingeJointParameters\s*\{\s*axis\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)",
            world,
        )
        self.assertEqual(axes, [("-1", "0", "0"), ("-1", "0", "0")])


if __name__ == "__main__":
    unittest.main()
