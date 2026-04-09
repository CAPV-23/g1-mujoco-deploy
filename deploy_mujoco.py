import time
import csv
import argparse

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name inside ./configs/")
    args = parser.parse_args()

    # Load config from local repo folder
    with open(f"./configs/{args.config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Read portable relative paths directly from YAML
    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    simulation_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)

    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale = float(config["ang_vel_scale"])
    dof_pos_scale = float(config["dof_pos_scale"])
    dof_vel_scale = float(config["dof_vel_scale"])
    action_scale = float(config["action_scale"])
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = int(config["num_actions"])
    num_obs = int(config["num_obs"])

    cmd = np.array(config["cmd_init"], dtype=np.float32)

    # Context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy().astype(np.float32)
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load TorchScript policy
    policy = torch.jit.load(policy_path)
    policy.eval()

    # CSV logging
    flush_every = 200  # control frames

    with open("g1_joint_data2.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)

        header = (
            ["time"]
            + [f"obs_{i}" for i in range(num_obs)]
            + [f"action_{i}" for i in range(num_actions)]
        )
        writer.writerow(header)

        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # PD torque to track current target positions
                tau = pd_control(
                    target_dof_pos,
                    d.qpos[7:7 + num_actions],
                    kps[:num_actions],
                    np.zeros(num_actions, dtype=np.float32),
                    d.qvel[6:6 + num_actions],
                    kds[:num_actions],
                )
                d.ctrl[:] = tau

                # Step physics
                mujoco.mj_step(m, d)
                counter += 1

                # Update control and log at control rate
                if counter % control_decimation == 0:
                    qj = d.qpos[7:7 + num_actions].copy()
                    dqj = d.qvel[6:6 + num_actions].copy()
                    quat = d.qpos[3:7].copy()
                    omega = d.qvel[3:6].copy()

                    qj_obs = (qj - default_angles) * dof_pos_scale
                    dqj_obs = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega_obs = omega * ang_vel_scale

                    period = 0.8
                    phase = (d.time % period) / period
                    sin_phase = np.sin(2 * np.pi * phase).astype(np.float32)
                    cos_phase = np.cos(2 * np.pi * phase).astype(np.float32)

                    obs[:3] = omega_obs
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale
                    obs[9:9 + num_actions] = qj_obs
                    obs[9 + num_actions:9 + 2 * num_actions] = dqj_obs
                    obs[9 + 2 * num_actions:9 + 3 * num_actions] = action
                    obs[9 + 3 * num_actions:9 + 3 * num_actions + 2] = np.array(
                        [sin_phase, cos_phase], dtype=np.float32
                    )

                    # Policy inference
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                        action = (
                            policy(obs_tensor)
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                            .astype(np.float32)
                        )

                    # Log data
                    writer.writerow([float(d.time)] + obs.tolist() + action.tolist())

                    if (counter // control_decimation) % flush_every == 0:
                        fcsv.flush()

                    # Convert action to target joint positions
                    target_dof_pos = action * action_scale + default_angles

                viewer.sync()

                # Real-time pacing
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
