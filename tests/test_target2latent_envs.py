"""Frame / kinematics checks for target2latent.envs that need no dataset."""

import numpy as np

from target2latent.envs import SPECS, ReacherSpec


def test_every_spec_has_a_consistent_layout():
    for spec in SPECS.values():
        assert spec.coord_dim % 3 == 0
        assert len(spec.columns) == len(spec.goal_keys)
        for col, key in zip(spec.columns, spec.goal_keys):
            # swm flattens '/' in column names when it builds the goal_ keys
            assert key == "goal_" + col.replace("/", "_")


def test_reacher_fk_reproduces_the_table_finger_pos():
    # (qpos, finger_pos) rows copied from reacher.lance (rows 0 and 2)
    spec = SPECS["reacher"]
    qpos = np.array([[-1.4907618, 2.0985866], [-1.3686203, 1.934317]])
    finger_pos = np.array([[0.10810096, -0.05108587], [0.12540203, -0.05323534]])
    wrist, finger = spec._fk_world(qpos)
    np.testing.assert_allclose(finger[:, :2], finger_pos, atol=1e-6)
    np.testing.assert_allclose(wrist[:, 2], ReacherSpec.ARM_Z)
    np.testing.assert_allclose(np.linalg.norm(wrist[:, :2], axis=1), ReacherSpec.LINK1)
    np.testing.assert_allclose(
        np.linalg.norm(finger[:, :2] - wrist[:, :2], axis=1), ReacherSpec.LINK2
    )


def test_reacher_goal_pose_is_two_sensor_frame_points():
    spec = SPECS["reacher"]
    coords, rot = spec.goal_pose({"qpos": np.zeros((4, 2), dtype=np.float32)})
    assert rot is None
    assert coords.shape == (4, 6) and coords.dtype == np.float32
    # straight arm along +x: wrist (0.12, 0, 0.05), fingertip (0.24, 0, 0.05)
    back = (coords[0].reshape(2, 3) @ spec.sensor_rot.T) + spec.sensor_origin
    np.testing.assert_allclose(back, [[0.12, 0, 0.05], [0.24, 0, 0.05]], atol=1e-6)
    # the sensor sits 0.80 m out and looks at the workspace: both points lie
    # ahead of it (sensor x forward) at roughly that range
    assert np.all(coords[0, [0, 3]] > 0.5) and np.all(coords[0, [0, 3]] < 1.0)


def test_reacher_fingertip_alone_would_be_ambiguous():
    # the two IK branches of one fingertip position give different wrists --
    # the reason the goal carries both points
    spec = SPECS["reacher"]
    q = np.array([[0.3, 1.2], [0.3 + 1.2, -1.2]])  # mirrored elbow, same fingertip
    wrist, finger = spec._fk_world(q)
    np.testing.assert_allclose(finger[0], finger[1], atol=1e-9)
    assert np.linalg.norm(wrist[0] - wrist[1]) > 0.05
