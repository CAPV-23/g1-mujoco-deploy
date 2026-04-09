import { ACTION_SCALE, STAND_POSE } from "./g1_obs.js";

const LEG_CTRL_LIMITS = new Float32Array([
   88,  88,  88, 139, 50, 50,
   88,  88,  88, 139, 50, 50,
]);

const KP = new Float32Array([
  120, 100, 100, 180,  60, 40,
  120, 100, 100, 180,  60, 40,
]);

const KD = new Float32Array([
   3.0, 2.5, 2.5, 4.0, 1.5, 1.0,
   3.0, 2.5, 2.5, 4.0, 1.5, 1.0,
]);

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

export function applyLegControl(state, action12) {
  const { qpos, qvel, ctrl } = state;

  for (let i = 0; i < 12; i += 1) {
    const q = qpos[7 + i];
    const dq = qvel[6 + i];
    const targetQ = STAND_POSE[i] + ACTION_SCALE * action12[i];
    const rawTau = KP[i] * (targetQ - q) - KD[i] * dq;
    ctrl[i] = clamp(rawTau, -LEG_CTRL_LIMITS[i], LEG_CTRL_LIMITS[i]);
  }

  for (let i = 12; i < ctrl.length; i += 1) {
    ctrl[i] = 0.0;
  }
}

