NAVIGATION CONTROLLER
=====================

Controller directory: controllers/navigation_controller
World controller name: navigation_controller
Entry point: navigation_controller.py
Local module: navigation_algorithms.py

The controller uses Webots' Python API and Python standard-library modules only.
It enables the GPS, Compass, LiDAR, Camera, and wheel encoders, then commands
the left and right wheel motors.

Planner selection:
- indoor_navigation.wbt starts with A*.
- indoor_navigation_dijkstra.wbt starts with Dijkstra; no key press is needed.
- The dedicated Dijkstra world sets Robot.controllerArgs to ["dijkstra"].
- The Console prints RUNNING PLANNER to confirm the selected algorithm.
- Focus the Webots 3D view and press A to select A*.
- Press D to select Dijkstra.

Do not rename the controller directory or the entry-point Python file without
also updating the Robot.controller field in the world file.
