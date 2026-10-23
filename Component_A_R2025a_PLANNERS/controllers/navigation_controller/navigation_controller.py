"""Webots controller for the Component A differential-drive rover.

Both grid planners are calculated at startup. Press A or D while the simulation
is focused to select A* or Dijkstra; distance and heading PID loops then track
the selected route. The route uses the known, static world map; LiDAR and camera
recognition provide an independent local slowdown/stop safety check.
"""

from controller import Robot
import math
import os
import sys

from navigation_algorithms import (
    GOAL,
    START,
    PID,
    cell_from_world,
    lidar_safe_commands,
    make_occupancy_map,
    nearest_free_cell,
    plan_path,
    select_startup_planner,
    world_from_cell,
    wrap_angle,
)

WHEEL_RADIUS_M = 0.05
WHEEL_TRACK_M = 0.30
MAX_WHEEL_SPEED = 6.28
MAX_LINEAR_SPEED = 0.29
WAYPOINT_TOLERANCE_M = 0.15
GOAL_TOLERANCE_M = 0.16


def camera_obstacle_nearby(camera):
    """Use Webots camera recognition as a supplemental forward hazard cue."""
    try:
        objects = camera.getRecognitionObjects()
    except (AttributeError, RuntimeError):
        return False
    for detected in objects:
        try:
            forward, lateral, _ = detected.getPosition()
            distance = math.hypot(forward, lateral)
            if forward > 0.0 and abs(lateral) < 0.42 and distance < 1.4:
                return True
        except (AttributeError, TypeError, ValueError):
            continue
    return False


def read_front_lidar(lidar):
    ranges = lidar.getRangeImage()
    if not ranges:
        return lidar.getMaxRange()
    center = len(ranges) // 2
    front = [
        value
        for value in ranges[max(0, center - 8) : center + 9]
        if math.isfinite(value) and value >= 0.0
    ]
    return min(front) if front else lidar.getMaxRange()


def compass_heading(compass):
    """Return the rover's forward bearing in world x/z coordinates.

    The rover faces local -Z. The world uses NUE (north +X, up +Y), so
    atan2(-north_x, -north_z) gives its forward bearing in the world X/Z plane.
    """
    north = compass.getValues()
    return math.atan2(-north[0], -north[2])


def set_wheel_speeds(left_motor, right_motor, linear, heading_rate):
    """Convert forward speed and world-bearing rate to wheel angular speeds."""
    # With the rover's local -Z forward axis, a positive map-frame bearing
    # change requires the opposite-sign wheel differential.
    left = (linear + heading_rate * WHEEL_TRACK_M / 2.0) / WHEEL_RADIUS_M
    right = (linear - heading_rate * WHEEL_TRACK_M / 2.0) / WHEEL_RADIUS_M
    left_motor.setVelocity(max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, left)))
    right_motor.setVelocity(max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, right)))


def main():
    robot = Robot()
    timestep_ms = int(robot.getBasicTimeStep())
    dt = timestep_ms / 1000.0
    keyboard = robot.getKeyboard()
    keyboard.enable(timestep_ms)

    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    gps = robot.getDevice("gps")
    compass = robot.getDevice("compass")
    lidar = robot.getDevice("lidar")
    camera = robot.getDevice("camera")
    left_encoder = robot.getDevice("left wheel sensor")
    right_encoder = robot.getDevice("right wheel sensor")

    for motor in (left_motor, right_motor):
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
    for sensor in (gps, compass, lidar, camera, left_encoder, right_encoder):
        sensor.enable(timestep_ms)
    camera.recognitionEnable(timestep_ms)

    occupied = make_occupancy_map()
    results = {
        "A*": plan_path(occupied, START, GOAL, "astar"),
        "Dijkstra": plan_path(occupied, START, GOAL, "dijkstra"),
    }
    planner_name = select_startup_planner(
        sys.argv[1:], os.environ.get("COMPONENT_A_PLANNER")
    )
    print("Component A: clearance-aware navigation (R2025a)")
    for name, result in results.items():
        print(
            "%s: path=%.2f m, expanded=%d cells"
            % (name, result.distance_m, result.explored)
        )
    print("RUNNING PLANNER: %s" % planner_name)
    print("Optional shortcuts: A or D (focus the Webots 3D view).")

    path = results[planner_name].path
    waypoints = [world_from_cell(cell) for cell in path]
    target_index = min(1, len(waypoints) - 1)
    heading_pid = PID(2.25, 0.025, 0.20, 0.8, 1.8, angular=True)
    distance_pid = PID(1.15, 0.025, 0.045, 0.50, MAX_LINEAR_SPEED)
    last_report_ms = 0
    elapsed_ms = 0
    completed = False

    while robot.step(timestep_ms) != -1:
        elapsed_ms += timestep_ms
        position = gps.getValues()
        x, z = position[0], position[2]
        heading = compass_heading(compass)

        key = keyboard.getKey()
        requested = "A*" if key in (ord("a"), ord("A")) else (
            "Dijkstra" if key in (ord("d"), ord("D")) else None
        )
        if requested and requested != planner_name:
            current = nearest_free_cell(occupied, cell_from_world(x, z))
            result = plan_path(occupied, current, GOAL, requested.lower().replace("*", "star"))
            planner_name = requested
            path = result.path
            waypoints = [world_from_cell(cell) for cell in path]
            target_index = min(1, len(waypoints) - 1)
            heading_pid.reset()
            distance_pid.reset()
            completed = False
            print(
                "Selected %s: replanned from %s; path=%.2f m, expanded=%d cells"
                % (planner_name, current, result.distance_m, result.explored)
            )

        if completed or target_index >= len(waypoints):
            set_wheel_speeds(left_motor, right_motor, 0.0, 0.0)
            continue

        target_x, target_z = waypoints[target_index]
        dx, dz = target_x - x, target_z - z
        distance = math.hypot(dx, dz)
        tolerance = GOAL_TOLERANCE_M if target_index == len(waypoints) - 1 else WAYPOINT_TOLERANCE_M
        if distance <= tolerance:
            if target_index == len(waypoints) - 1:
                if not completed:
                    print(
                        "Goal reached using %s at t=%.2f s; position=(%.3f, %.3f)m; error=%.3fm. Motors stopped."
                        % (planner_name, elapsed_ms / 1000.0, x, z, distance)
                    )
                    completed = True
                set_wheel_speeds(left_motor, right_motor, 0.0, 0.0)
                continue
            target_index += 1
            distance_pid.reset()
            continue

        desired_heading = math.atan2(dz, dx)
        heading_error = wrap_angle(desired_heading - heading)
        omega = heading_pid.update(heading_error, dt)
        speed = distance_pid.update(distance, dt)
        speed = max(0.0, min(MAX_LINEAR_SPEED, speed))
        # Turn in place instead of driving sideways through a waypoint corner.
        speed *= max(0.0, math.cos(heading_error))

        lidar_front = read_front_lidar(lidar)
        visual_hazard = camera_obstacle_nearby(camera)
        speed, omega, safety_mode = lidar_safe_commands(speed, omega, lidar_front)
        if visual_hazard and safety_mode != "turn-only":
            speed *= 0.25

        set_wheel_speeds(left_motor, right_motor, speed, omega)

        if elapsed_ms - last_report_ms >= 2000:
            left_distance = left_encoder.getValue() * WHEEL_RADIUS_M
            right_distance = right_encoder.getValue() * WHEEL_RADIUS_M
            print(
                "t=%.1fs planner=%s waypoint=%d/%d range=%.2fm safety=%s vision=%s "
                "wheel_travel=(%.2f, %.2f)m position=(%.2f, %.2f)m "
                "heading=%.1f target=%.1f error=%.1f drive=(%.2f, %.2f)"
                % (
                    elapsed_ms / 1000.0,
                    planner_name,
                    target_index + 1,
                    len(waypoints),
                    lidar_front,
                    safety_mode,
                    "hazard" if visual_hazard else "clear",
                    left_distance,
                    right_distance,
                    x,
                    z,
                    math.degrees(heading),
                    math.degrees(desired_heading),
                    math.degrees(heading_error),
                    speed,
                    omega,
                )
            )
            last_report_ms = elapsed_ms


if __name__ == "__main__":
    main()
