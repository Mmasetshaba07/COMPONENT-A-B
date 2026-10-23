AUTONOMOUS SYSTEMS AND ROBOTICS 700 — COMPONENT B
Multi-Agent Robotic Swarm System with Learning-Based Decision-Making

PROJECT CONTENTS
  worlds/search_rescue.wbt                     Webots courtyard world
  controllers/swarm_supervisor/swarm_supervisor.py
                                               Multi-agent Supervisor controller
  analysis.py                                  Matplotlib plots and evaluation table
  outputs/                                     Run logs and generated plots

OPEN AND RUN IN WEBOTS
1. Extract this ZIP to a normal folder (do not open the WBT from inside the ZIP).
2. Install Webots R2023b or a newer release.
3. In Webots, select File > Open World and open worlds/search_rescue.wbt.
4. Press the green Run/Play button. The simulation is autonomous and runs until all
   three targets are found or the controller reaches its maximum step count.
5. From a terminal opened in the extracted project root, run:
       python analysis.py
   This creates trajectory.png, consensus.png, confusion_matrix.png, and evaluation.csv
   inside outputs/.

PYTHON / DEPENDENCIES
- The controller uses Webots' built-in Python controller API and Python standard library.
- For the plots: Python 3.9+ and Matplotlib (pip install matplotlib).
- No ROS, Gazebo, NumPy, or network connection is required by the simulation.
- Run analysis.py with the same Python environment as the Matplotlib installation.

WHAT THE DEMO IMPLEMENTS
- Three Webots Robot nodes with distinct roles: Robot 1 is the leader/explorer;
  Robots 2 and 3 are searchers.
- A leader-centered star communication graph. A synchronous average-consensus update
  shares and smooths mission-progress estimates.
- Random-key Particle Swarm Optimisation (PSO) assigns three victim-search tasks to
  three robots while minimizing total start-to-target distance.
- Eight-connected A* on a 0.25 m occupancy grid routes agents around four obstacles.
- A small one-hidden-layer neural network is trained in the controller on generated
  examples. It chooses cautious or cruise movement from distance, obstacle clearance,
  and progress features. The reference labels are an explicit distance/clearance rule.
- The controller saves sampled robot trajectories, decisions, progress, and target
  status to outputs/trajectory.csv. analysis.py plots the results and computes metrics.

MODEL LIMITATIONS / REPORTING NOTE
This is a reproducible kinematic demonstration: a Webots Supervisor updates the pose
of each Robot node directly. It does not model wheel slip, motor dynamics, real LiDAR or
camera sensors, radio packet loss, or real rescue operations. Treat the NN confusion
matrix as evaluation against the synthetic rule-generated labels, not as evidence of
real-world safety or generalisation. State these limits in the report and discuss how
hardware-in-the-loop tests, sensor uncertainty, communication failures, human oversight,
and local South African safety/privacy requirements would be addressed before deployment.

SIMULATION SETTINGS
- World file uses Webots R2023b syntax and a 64 ms basic time step.
- The Supervisor controller is set in the WBT world; no manual controller selection is needed.
- Typical run time depends on machine speed and Webots real-time settings.
