"""Create Matplotlib plots and a compact evaluation table from a Webots run."""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
LOG = OUT / "trajectory.csv"


def load_rows():
    if not LOG.exists():
        raise SystemExit("No outputs/trajectory.csv found. Run the Webots world first.")
    with LOG.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    robots = defaultdict(list)
    for row in rows:
        row["sim_time"] = float(row["sim_time"])
        for key in ("x", "z", "progress", "consensus", "decision", "oracle", "distance_to_target", "path_error", "found"):
            row[key] = float(row[key])
        robots[row["robot"]].append(row)

    # Trajectory and courtyard map
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor("#e8e7df")
    obstacle_boxes = [(-1.7, -1.6, 1.4, 3.2), (1.9, -2.75, 2.2, 1.1),
                      (1.6, 1.4, 1.0, 2.0), (-4.7, -3.1, 1.4, 1.2)]
    for x, z, w, h in obstacle_boxes:
        ax.add_patch(plt.Rectangle((x, z), w, h, color="#555b5c", alpha=0.9))
    colors = {"Robot 1": "#d62728", "Robot 2": "#1f77b4", "Robot 3": "#2ca02c"}
    for name, data in sorted(robots.items()):
        c = colors.get(name, "black")
        ax.plot([r["x"] for r in data], [r["z"] for r in data], color=c, lw=2.3, label=name + " actual path")
        if data:
            ax.scatter(data[0]["x"], data[0]["z"], c=c, marker="o", s=55)
    for i, z in enumerate((3.8, 0.0, -3.8), start=1):
        ax.scatter(6, z, c="#e31a1c", marker="^", edgecolors="black", s=125)
        ax.text(6.12, z + 0.10, "Victim %d" % i, fontsize=9)
    ax.scatter(-6.1, 0, marker="s", color="#39a852", s=100, label="Start area")
    ax.set(xlim=(-8.5, 8.5), ylim=(-6.5, 6.5), xlabel="x position (m)", ylabel="z position (m)",
           title="Webots multi-robot search-and-rescue trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory.png", dpi=180)
    plt.close(fig)

    # Local progress vs the distributed consensus estimate.
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, data in sorted(robots.items()):
        if data:
            ax.plot([r["sim_time"] for r in data], [r["consensus"] for r in data],
                    color=colors.get(name), lw=2, label=name)
    ax.set(xlabel="Simulation time (s)", ylabel="Consensus progress estimate",
           ylim=(-0.03, 1.03), title="Leader-centered average-consensus convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "consensus.png", dpi=180)
    plt.close(fig)

    # 2 x 2 confusion matrix for the NN's cautious/cruise action prediction.
    cm = [[0, 0], [0, 0]]
    for row in rows:
        true = int(row["oracle"])
        pred = int(row["decision"])
        cm[true][pred] += 1
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Cautious", "Cruise"])
    ax.set_yticks([0, 1], labels=["Cautious", "Cruise"])
    ax.set_xlabel("Predicted action")
    ax.set_ylabel("Reference action")
    ax.set_title("Neural speed-policy confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Evaluation table: distance actually travelled, time-to-goal, and mean path-following error.
    eval_rows = []
    for name, data in sorted(robots.items()):
        if not data:
            continue
        travelled = sum(math.hypot(b["x"]-a["x"], b["z"]-a["z"]) for a, b in zip(data, data[1:]))
        reached = [r for r in data if r["found"] >= 1 or r["distance_to_target"] <= 0.32]
        t_goal = reached[0]["sim_time"] if reached else "not reached"
        mean_error = sum(r["path_error"] for r in data) / len(data)
        avg_progress = sum(r["progress"] for r in data) / len(data)
        eval_rows.append({"robot": name, "logged_path_length_m": "%.3f" % travelled,
                          "time_to_goal_s": ("%.2f" % t_goal if isinstance(t_goal, float) else t_goal),
                          "mean_waypoint_error_m": "%.3f" % mean_error,
                          "mean_mission_progress": "%.3f" % avg_progress,
                          "samples": len(data)})
    with (OUT / "evaluation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["robot", "logged_path_length_m", "time_to_goal_s",
            "mean_waypoint_error_m", "mean_mission_progress", "samples"])
        writer.writeheader()
        writer.writerows(eval_rows)
    print("Created outputs/trajectory.png, consensus.png, confusion_matrix.png, evaluation.csv")
    print("Confusion matrix (reference rows x predicted columns):", cm)
    print("Note: path length is calculated from logged positions (sampled every five controller steps).")


if __name__ == "__main__":
    main()
