COMPONENT A — WEBOTS R2025a ROBOT PROJECT
=========================================

This ZIP is a complete Webots R2025a project folder. It contains the world,
rover, all sensors and wheel devices, the navigation controller, planner/PID
utilities, and local algorithm tests. No assignment report files are required.

OPEN AND RUN IN WEBOTS R2025a
-----------------------------
1. Extract the ZIP. Keep the Component_A_R2025a_PLANNERS folder intact.
2. In Webots R2025a, choose File > Open World.
3. Open:
       Component_A_R2025a_PLANNERS/worlds/indoor_navigation.wbt
4. Press the simulation play button.
5. If Webots reports that Python is unavailable, configure a Python 3
   interpreter in Webots Preferences, then reopen the world.

The indoor arena is 12 m x 12 m. It includes the same nine barrier positions
and start/goal marker locations as the design preview. The rover uses its
specified physical dimensions. The design preview enlarged the robot visually
so it would be easier to inspect.

ROBOT DEVICES
-------------
- Differential-drive left and right wheel motors.
- Left and right wheel position sensors (encoders).
- 360-degree single-layer LiDAR.
- GPS and Compass.
- Forward Camera with object recognition enabled.

PLANNERS AND CONTROLLERS
------------------------
- To run A*, open worlds/indoor_navigation.wbt and press Play.
- To run Dijkstra, open worlds/indoor_navigation_dijkstra.wbt and press Play.
- The Console confirms RUNNING PLANNER: A* or RUNNING PLANNER: Dijkstra.
- No keyboard shortcut is required. Both worlds use the same robot and arena.
- Optional shortcuts: focus the Webots 3D view and press A or D.
- Open/reload each world from its initial pose for a fair planner comparison.
- Both algorithms are calculated and their path lengths and explored-cell
  counts are printed in the Webots console.
- A heading PID controls turning; a distance PID controls forward speed.
- LiDAR provides near-field stop/slow checks; camera object recognition adds a
  supplemental obstacle cue.

The planners use the known static map and do not remap a changed environment.
PID values are initial simulation values, not hardware-calibrated or
safety-certified.

TEST WITHOUT WEBOTS
-------------------
On Windows, double-click RUN_TESTS_WINDOWS.bat in the project folder.
Python 3.10+ must be installed. The window stays open so you can read the result.

Python 3.10+ is sufficient for the planner/PID checks; no packages are needed:
       cd Component_A_R2025a_PLANNERS
       python -m unittest discover -s tests -v

SIMULATOR VALIDATION
--------------------
Both planners were run from the initial pose to the goal in the actual Webots
R2025a Linux physics engine with software-rendered sensor images:
- A*: goal reached at 71.45 simulated seconds; final GPS error 0.128 m.
- Dijkstra: goal reached at 71.20 simulated seconds; final GPS error 0.139 m.
The motors remain stopped after goal completion. See VALIDATION.txt.
Windows is the intended user platform, but these engine checks ran on Linux;
Windows-specific Python setup and graphics drivers are not tested here.

FILES
-----
worlds/indoor_navigation.wbt
worlds/indoor_navigation_dijkstra.wbt
controllers/navigation_controller/navigation_controller.py
controllers/navigation_controller/navigation_algorithms.py
controllers/navigation_controller/README.txt
tests/test_webots_algorithms.py
RUN_TESTS_WINDOWS.bat
VALIDATION.txt
