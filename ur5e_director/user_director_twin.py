"""User Instruction and Digital Twin Coaching Module for Antigravity.

This module implements the human-in-the-loop "Director & Digital Twin Coach"
component of the Antigravity autonomous robotic cameraman system. It is
responsible for:
  1. Translating high-level cinematic shot primitives (e.g., 'orbit',
     'dolly_in', 'low_angle_hero', 'tracking') into actionable stage directions
     for the human creator standing in front of the camera.
  2. Defining the time-parameterized 3D reference trajectory ("Digital Twin")
     that represents where the human creator should ideally be positioned at
     any timestamp `t`.
  3. Computing real-time spatial deviation and posture metrics by comparing live
     user tracking coordinates (from mocap, ToF LiDAR, IMU, or MediaPipe)
     against the ideal digital twin trajectory.

Coordinate System Conventions:
  - World frame is right-handed Z-up (in meters), with the UR5e robot base at
    the origin `(0.0, 0.0, 0.0)` and the floor plane at `Z = 0.0`.
  - The creator staging zone is centered around `X = [0.8, 1.3]`, `Y = [-0.5, 0.5]`.
  - In the MuJoCo `humanoid.xml` model, the root mocap body is attached at the
    torso center, which rests at `Z = 1.282` meters when the humanoid stands
    upright with its feet flat on the floor (`Z = 0.0`).

Typical usage example:
  ```python
  director = UserDirectorDigitalTwin(shot_type="orbit")
  print(director.instruction)
  report = director.evaluate_user_performance(actual_user_pos=[1.05, 0.1, 1.25], t=2.0)
  ```
"""

import sys
import time
from typing import Any, Dict, List, Sequence, Union
import numpy as np

# Ensure Windows terminals handle Unicode output cleanly without encoding crashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Vertical height (in meters) of the MuJoCo humanoid torso center when standing
# upright with feet touching the floor plane at Z = 0.0m.
HUMANOID_TORSO_Z: float = 1.282


class UserDirectorDigitalTwin:
    """Manages ideal shot choreography, creator instructions, and motion coaching.

    This class maintains the active cinematic shot state, computes the expected
    3D position of the human subject (the "ideal digital twin") as a function of
    elapsed shot time, and evaluates real-time physical deviations to generate
    coaching feedback for the UI and HUD overlays.

    Attributes:
        shot_type: A string identifier for the active cinematic shot primitive
            (one of `'orbit'`, `'dolly_in'`, `'low_angle_hero'`, `'tracking'`).
        instruction: A human-readable stage direction string instructing the
            creator how to move or pose for the current shot.
        start_time: Unix timestamp (in seconds) when the instance was initialized.
    """

    def __init__(self, shot_type: str = "orbit") -> None:
        """Initializes the Digital Twin Coach with a specified shot primitive.

        Args:
            shot_type: Initial cinematic shot primitive identifier. Defaults to
                `'orbit'`.
        """
        self.shot_type: str = shot_type
        self.instruction: str = self.get_agent_instruction(shot_type)
        self.start_time: float = time.time()

    def set_shot_type(self, shot_type: str) -> None:
        """Updates the active shot primitive and refreshes creator stage guidance.

        Args:
            shot_type: The new shot primitive identifier (e.g., `'dolly_in'`).
        """
        self.shot_type = shot_type
        self.instruction = self.get_agent_instruction(shot_type)

    def get_agent_instruction(self, shot_type: str) -> str:
        """Maps a shot primitive identifier to clear human stage directions.

        Args:
            shot_type: The cinematic shot primitive selected by the Gemma 4
                director parser.

        Returns:
            A concise instructional string telling the creator where to stand
            and how to move during the shot.
        """
        instructions: Dict[str, str] = {
            "orbit": (
                "Stand in the center zone. Slowly walk in a small circle "
                "clockwise while maintaining high energy."
            ),
            "dolly_in": (
                "Start at the back marker, step forward steadily toward the "
                "camera over 5 seconds, then hold pose."
            ),
            "low_angle_hero": (
                "Stand tall, plant your feet firmly on the green mat, and "
                "perform a confident power pose."
            ),
            "tracking": (
                "Walk smoothly from left to right across the frame at a "
                "consistent pace."
            ),
        }
        return instructions.get(shot_type, "Hold steady and face the camera lens.")

    def get_ideal_twin_position(self, t: float) -> np.ndarray:
        """Computes the mathematically ideal 3D subject position at time `t`.

        This method defines the reference trajectory of the "Digital Twin" in
        MuJoCo world coordinates `(X, Y, Z)` at torso height (`Z = 1.282m`).

        Args:
            t: Elapsed simulation or shot time in seconds.

        Returns:
            A 1D NumPy float array of shape `(3,)` representing the ideal
            `[X, Y, Z]` world coordinates in meters.
        """
        if self.shot_type == "orbit":
            # Ideal circular creator path of radius 0.15m centered at (1.0, 0.0).
            return np.array(
                [
                    1.0 + 0.15 * np.cos(t * 0.5),
                    0.15 * np.sin(t * 0.5),
                    HUMANOID_TORSO_Z,
                ]
            )
        elif self.shot_type == "dolly_in":
            # Ideal linear walk forward along the -X axis toward the camera,
            # clamping at X = 0.8m to stay outside the robot's safety boundary.
            pos_x = max(0.8, 1.3 - (t * 0.05))
            return np.array([pos_x, 0.0, HUMANOID_TORSO_Z])
        elif self.shot_type == "low_angle_hero":
            # Confident stationary pose centered on the creator floor mat.
            return np.array([1.2, 0.0, HUMANOID_TORSO_Z])
        elif self.shot_type == "tracking":
            # Smooth lateral oscillation across the camera frame along the Y axis.
            return np.array(
                [1.1, 0.3 * np.sin(0.4 * t), 0.8 * 0.0 + HUMANOID_TORSO_Z]
            )
        else:
            # Default stationary reference point in front of the robot arm.
            return np.array([1.0, 0.0, HUMANOID_TORSO_Z])

    def evaluate_user_performance(
        self,
        actual_user_pos: Union[Sequence[float], np.ndarray],
        t: float,
    ) -> Dict[str, Any]:
        """Compares the user's actual 3D position against the ideal digital twin.

        Normalizes incoming height coordinates if they are referenced from the
        floor (`Z < 0.4m`) or waist/pelvis (`0.4m <= Z < 1.1m`), calculates the
        Euclidean distance error from the ideal trajectory, and checks vertical
        posture for slouching.

        Args:
            actual_user_pos: A 3-element sequence or NumPy array `[X, Y, Z]`
                representing the tracked human subject position in meters.
            t: Elapsed shot time in seconds used to query the ideal twin pose.

        Returns:
            A dictionary containing telemetry and coaching diagnostics:
                - `'ideal_position'`: `List[float]` of `[X, Y, Z]` ideal coordinates.
                - `'actual_position'`: `List[float]` of normalized `[X, Y, Z]` user
                  coordinates.
                - `'deviation_score'`: `float` Euclidean distance error in meters,
                  rounded to 3 decimal places.
                - `'coaching_tips'`: `List[str]` of actionable human-readable
                  feedback messages.
        """
        ideal_pos = self.get_ideal_twin_position(t)
        actual_pos = np.asarray(actual_user_pos, dtype=float).copy()

        # Adapt floor-referenced or waist-referenced sensor streams to torso height
        # so comparisons against HUMANOID_TORSO_Z (1.282m) remain consistent.
        if actual_pos[2] < 0.4:
            actual_pos[2] += HUMANOID_TORSO_Z
        elif actual_pos[2] < 1.1:
            actual_pos[2] += HUMANOID_TORSO_Z - 0.857

        # Compute 3D Euclidean distance between actual user and ideal digital twin.
        deviation = np.linalg.norm(actual_pos - ideal_pos)

        feedback: List[str] = []
        if deviation > 0.15:
            feedback.append(
                "[!] Deviation detected: Adjust position closer to the ideal digital twin path."
            )
        else:
            feedback.append(
                "[OK] Form and pacing match the ideal director trajectory!"
            )

        # Check vertical torso alignment to detect slouching (> 8cm drop).
        if actual_pos[2] < (HUMANOID_TORSO_Z - 0.08):
            feedback.append(
                "[!] Posture check: Stand up straighter for the camera frame."
            )

        return {
            "ideal_position": ideal_pos.tolist(),
            "actual_position": actual_pos.tolist(),
            "deviation_score": round(float(deviation), 3),
            "coaching_tips": feedback,
        }


if __name__ == "__main__":
    # Standalone verification demo for the Digital Twin Coach.
    director = UserDirectorDigitalTwin(shot_type="orbit")
    print("=" * 60)
    print("=== AGENT DIRECTOR INSTRUCTION ===")
    print(director.instruction)
    print("=" * 60 + "\n")

    # Simulate a 3-step coaching loop with a slight trajectory deviation and slouch.
    for step in range(3):
        sim_t = step * 1.0
        simulated_user_pos = np.array([1.05 + (step * 0.02), 0.1, 1.25])
        report = director.evaluate_user_performance(simulated_user_pos, sim_t)
        print(f"Time {sim_t}s | Deviation: {report['deviation_score']}m")
        for tip in report["coaching_tips"]:
            print(f"  -> {tip}")
        time.sleep(0.1)
