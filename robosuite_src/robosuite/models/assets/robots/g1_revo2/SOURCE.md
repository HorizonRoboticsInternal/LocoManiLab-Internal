# G1 Revo2 robot provenance

Migrated from SIMPLE `data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_v3.xml`, revision `ab818408b10efa5ea150e1453ef54e6bf4d6abee`. `SOURCE.json` retains source hashes, selected assets and transformations. The source XML is retained for comparison. `robot.xml` removes scene-owned floor/light/visual settings, uses an equivalent `freejoint` element for robosuite, and explicitly declares the two zero-mass camera mounts for MuJoCo 3.3.1. All named physical joints, inertias, actuators, coupled constraints and tactile sensors are retained. The private workstation prefix in camera metadata is removed.

The robot has 29 body and 22 hand actuators; free-base nq=58, nv=57, nu=51, 10 equalities and 122 sensors. Camera names include `hbvcam_left_pinhole`, automatically prefixed by robosuite. Hand PD and open poses use the source calibration.

Robot meshes, camera mounts, calibration and robot descriptions retain their source owners' terms. No asset-specific redistribution license was found in the selected source directory; the benchmark or LocoManiLab MIT license must not be applied to these assets.
