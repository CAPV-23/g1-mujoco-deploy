export const STAND_POSE = new Float32Array([
  -0.10,  0.00,  0.00,  0.30, -0.20,  0.00,
  -0.10,  0.00,  0.00,  0.30, -0.20,  0.00,
]);

export const ACTION_SCALE = 0.25;
export const BASE_ANG_VEL_SCALE = 0.25;
export const JOINT_VEL_SCALE = 0.05;
export const CMD_SCALED = new Float32Array([0.5, 0.0, 0.0]);
export const PHASE_PERIOD = 0.8;

const BASE_QUAT_OFFSET = 3;
const JOINT_QPOS_OFFSET = 7;
const BASE_ANGVEL_OFFSET = 3;
const JOINT_QVEL_OFFSET = 6;

function quatToGravity(qw, qx, qy, qz) {
  const gx = 2 * (-qz * qx + qw * qy);
  const gy = -2 * (qz * qy + qw * qx);
  const gz = 1 - 2 * (qx * qx + qy * qy);
  return [gx, gy, gz];
}

function standardizeVector(x, mean, std) {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    const s = Math.abs(std[i]) < 1e-8 ? 1.0 : std[i];
    y[i] = (x[i] - mean[i]) / s;
  }
  return y;
}

export function buildRawObservation(state, prevAction, simTimeSec) {
  const { qpos, qvel } = state;

  const qw = qpos[BASE_QUAT_OFFSET + 0];
  const qx = qpos[BASE_QUAT_OFFSET + 1];
  const qy = qpos[BASE_QUAT_OFFSET + 2];
  const qz = qpos[BASE_QUAT_OFFSET + 3];

  const [gx, gy, gz] = quatToGravity(qw, qx, qy, qz);

  const obs = new Float32Array(47);
  let k = 0;

  obs[k++] = qvel[BASE_ANGVEL_OFFSET + 0] * BASE_ANG_VEL_SCALE;
  obs[k++] = qvel[BASE_ANGVEL_OFFSET + 1] * BASE_ANG_VEL_SCALE;
  obs[k++] = qvel[BASE_ANGVEL_OFFSET + 2] * BASE_ANG_VEL_SCALE;

  obs[k++] = gx;
  obs[k++] = gy;
  obs[k++] = gz;

  obs[k++] = CMD_SCALED[0];
  obs[k++] = CMD_SCALED[1];
  obs[k++] = CMD_SCALED[2];

  for (let i = 0; i < 12; i += 1) {
    const q = qpos[JOINT_QPOS_OFFSET + i];
    obs[k++] = q - STAND_POSE[i];
  }

  for (let i = 0; i < 12; i += 1) {
    const dq = qvel[JOINT_QVEL_OFFSET + i];
    obs[k++] = dq * JOINT_VEL_SCALE;
  }

  for (let i = 0; i < 12; i += 1) {
    obs[k++] = prevAction[i];
  }

  const phase = (2.0 * Math.PI * simTimeSec) / PHASE_PERIOD;
  obs[k++] = Math.sin(phase);
  obs[k++] = Math.cos(phase);

  return obs;
}

export function buildPolicyObservation(state, prevAction, simTimeSec, norm) {
  const raw = buildRawObservation(state, prevAction, simTimeSec);
  return standardizeVector(raw, norm.x_mean, norm.x_std);
}
