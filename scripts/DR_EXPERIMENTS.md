## Domain randomization experiment sweep (Leap touch rotate)

This folder contains JSON DR configs intended to help diagnose “delicate fingers” by sweeping only the DR knobs that are already supported.

### How to run an experiment

Pick one JSON file and pass it via `--dr_config_path`:

```bash
python learning/train_jax_ppo.py \
  --mask_path=scripts/sensor_masks.json \
  --env_name=LeapCubeRotateZAxisTouchObjectsMask16 \
  --domain_randomization \
  --dr_config_path=scripts/<EXPERIMENT_JSON>
```

Notes:
- Each experiment JSON is a **single object** mapping `param_key -> {min,max}`.
- Keys omitted from a given experiment fall back to the **hardcoded defaults** in the environment DR code.
- The “schema” / defaults reference is in `scripts/dr_leap_touch_objects_schema.json` (metadata only; not consumed by training).

**XML vs DR:** Palm and finger **link** collision friction (MuJoCo triplet) come from the compiled model defaults in `mujoco_playground/_src/manipulation/leap_hand/xmls/leap_rh_touch_mjx.xml` (class `leap_rh`). Fingertip pad geoms use the `tip` / `thumb_tip` classes there. None of the JSON files below set `hand_geom_friction_*`; optional overrides for those keys are documented in the schema if you add a custom JSON.

### Experiment list

### Suggested priority (most → least important, friction-first)

1. **Combo (aggressive fingers)**: `scripts/dr_exp10_combo_aggressive_fingers.json`
2. **Fingertip sliding friction (high)**: `scripts/dr_exp03_fingertip_friction_high.json`
3. **Fingertip sliding friction (medium)**: `scripts/dr_exp02_fingertip_friction_med.json`
4. **Higher actuator gain (kp)**: `scripts/dr_exp06_high_actuator_kp.json`
5. **Lower hand joint damping**: `scripts/dr_exp04_low_hand_damping.json`
6. **Lower hand joint frictionloss**: `scripts/dr_exp05_low_hand_frictionloss.json`
7. **Lower cube inertia**: `scripts/dr_exp09_low_cube_inertia.json`
8. **Wider cube COM offset**: `scripts/dr_exp08_wide_cube_com_offset.json`
9. **Higher hand armature** (control): `scripts/dr_exp07_high_hand_armature.json`
10. **Baseline (explicit defaults)**: `scripts/dr_exp01_baseline_defaults.json`

#### 1) Baseline (explicit defaults)
- **File**: `scripts/dr_exp01_baseline_defaults.json`
- **What it does**: Sets *all* supported DR keys to the current built-in defaults (useful for reproducibility / sanity checks).

#### 2) Fingertip sliding friction (medium)
- **File**: `scripts/dr_exp02_fingertip_friction_med.json`
- **Changes**:
  - `geom_friction_fingertips`: \(U[1.0, 1.5]\)

#### 3) Fingertip sliding friction (high)
- **File**: `scripts/dr_exp03_fingertip_friction_high.json`
- **Changes**:
  - `geom_friction_fingertips`: \(U[1.5, 2.5]\)

#### 4) Lower hand joint damping
- **File**: `scripts/dr_exp04_low_hand_damping.json`
- **Hypothesis**: Less velocity damping can make motion feel more “snappy” / aggressive.
- **Changes**:
  - `hand_damping_scale`: \(U[0.3, 0.8]\)

#### 5) Lower hand joint frictionloss
- **File**: `scripts/dr_exp05_low_hand_frictionloss.json`
- **Hypothesis**: Reduces internal frictionloss that can suppress small motions and make the hand feel “sticky”.
- **Changes**:
  - `hand_frictionloss_scale`: \(U[0.2, 0.8]\)

#### 6) Higher actuator gain (kp)
- **File**: `scripts/dr_exp06_high_actuator_kp.json`
- **Hypothesis**: Stronger actuator tracking can increase contact “assertiveness”.
- **Changes**:
  - `actuator_kp_scale`: \(U[1.2, 2.0]\)

#### 7) Higher hand armature
- **File**: `scripts/dr_exp07_high_hand_armature.json`
- **Hypothesis**: Increased armature changes joint dynamics (often slows acceleration); included as a control sweep.
- **Changes**:
  - `hand_armature_scale`: \(U[1.05, 1.3]\)

#### 8) Wider cube COM offset (cube `body_ipos`)
- **File**: `scripts/dr_exp08_wide_cube_com_offset.json`
- **Hypothesis**: Larger COM variation can force more robust, firmer contact strategies.
- **Changes**:
  - `cube_ipos_add`: \(U[-0.01, 0.01]\) (per axis, meters)

#### 9) Lower cube inertia
- **File**: `scripts/dr_exp09_low_cube_inertia.json`
- **Hypothesis**: Easier-to-spin objects may reward more decisive contacts for rotation.
- **Changes**:
  - `cube_inertia_scale`: \(U[0.5, 0.9]\)

#### 10) Combo: aggressive fingers
- **File**: `scripts/dr_exp10_combo_aggressive_fingers.json`
- **What it does**: Combines the most directly “aggressiveness” aligned knobs:
  - higher fingertip friction
  - lower damping
  - lower frictionloss
  - higher actuator kp
- **Changes**:
  - `geom_friction_fingertips`: \(U[1.5, 2.5]\)
  - `hand_damping_scale`: \(U[0.3, 0.8]\)
  - `hand_frictionloss_scale`: \(U[0.2, 0.8]\)
  - `actuator_kp_scale`: \(U[1.2, 2.0]\)
