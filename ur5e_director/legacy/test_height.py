"""Humanoid Floor-Clearance Verification Utility (`test_height.py`).

This diagnostic script loads `scene.xml`, runs forward kinematics (`mj_forward`),
and prints the exact world-space vertical Z coordinates (in meters) of the
humanoid's torso, head, and left/right foot soles (`foot1_right`, `foot1_left`).

Usage:
  Run this script after modifying `scene.xml` or `humanoid.xml` to verify that
  the humanoid's feet rest cleanly on top of the floor plane (`Z >= 0.0m`,
  standing torso height `Z = 1.282m`).
"""

import mujoco


def verify_humanoid_floor_clearance(scene_path: str = "scene.xml") -> bool:
    """Computes and prints humanoid geom elevations to verify floor contact.

    Args:
        scene_path: Path to the MuJoCo XML scene file. Defaults to `'scene.xml'`.

    Returns:
        `True` if both left and right foot bottom soles are at or above `Z = -0.001m`;
        `False` if the feet penetrate below the floor plane.
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Subtract the 0.027m foot capsule radius to get the bottom sole elevation.
    foot1_r = data.geom_xpos[model.geom("foot1_right").id][2] - 0.027
    foot1_l = data.geom_xpos[model.geom("foot1_left").id][2] - 0.027
    torso_z = data.geom_xpos[model.geom("torso").id][2]
    head_z = data.geom_xpos[model.geom("head").id][2]

    print("=" * 60)
    print(f"HUMANOID HEIGHT VERIFICATION REPORT ({scene_path})")
    print("=" * 60)
    print(f"Mocap Body Position:       {data.mocap_pos[0]}")
    print(f"Torso Center Height:       Z = {torso_z:.4f} m")
    print(
        f"Head Top Height:           Z = {head_z + 0.09:.4f} m (Eye level: {head_z:.3f} m)"
    )
    print(f"Right Foot Bottom Sole:    Z = {foot1_r:.4f} m")
    print(f"Left Foot Bottom Sole:     Z = {foot1_l:.4f} m")
    print("-" * 60)
    is_grounded = bool(foot1_r >= -0.001 and foot1_l >= -0.001)
    if is_grounded:
        print("[SUCCESS] Humanoid feet are fully resting ON TOP OF THE FLOOR!")
    else:
        print("[FAILURE] Humanoid feet are still below floor level!")
    print("=" * 60)
    return is_grounded


if __name__ == "__main__":
    verify_humanoid_floor_clearance()
