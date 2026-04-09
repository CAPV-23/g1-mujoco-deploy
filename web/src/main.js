import './style.css'
import { G1LSTMPolicy } from './g1_policy.js'
import { buildPolicyObservation } from './g1_obs.js'
import { applyLegControl } from './g1_controller.js'

const base = import.meta.env.BASE_URL

const POLICY_PATH = base + 'policies/g1/g1_lstm_policy.onnx'
const NORM_PATH = base + 'policies/g1/normalization.json'

document.querySelector('#app').innerHTML = `
  <div style="padding:24px;font-family:Arial,sans-serif">
    <h1>G1 Web Demo</h1>
    <p id="status">Loading policy...</p>
    <p id="time">Sim time: 0.00</p>
    <p id="action0">Action[0]: 0.0000</p>
  </div>
`

const statusEl = document.querySelector('#status')
const timeEl = document.querySelector('#time')
const action0El = document.querySelector('#action0')

const state = {
  qpos: new Float32Array(19),
  qvel: new Float32Array(18),
  ctrl: new Float32Array(12),
}

state.qpos[3] = 1.0

let prevAction = new Float32Array(12)
let simTime = 0

async function main() {
  try {
    const policy = await G1LSTMPolicy.create(POLICY_PATH, NORM_PATH)
    policy.reset()
    statusEl.textContent = 'Policy loaded. Running loop...'

    async function step() {
      const obs = buildPolicyObservation(state, prevAction, simTime, policy.norm)
      const action = await policy.step(obs)

      applyLegControl(state, action)
      prevAction = action
      simTime += 0.02

      timeEl.textContent = `Sim time: ${simTime.toFixed(2)}`
      action0El.textContent = `Action[0]: ${action[0].toFixed(4)}`

      requestAnimationFrame(step)
    }

    step()
  } catch (err) {
    console.error(err)
    statusEl.textContent = `Error: ${err.message}`
  }
}

main()
