"""Constants for WF_TRON1A wheeled-legged biped."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "wf_tron1a"
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_flat_terrain.xml"


def task_to_xml(task_name: str):
  return {
      "flat_terrain": FLAT_TERRAIN_XML,
  }[task_name]


# Sensor names.
GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

# Body/site names.
ROOT_BODY = "base_Link"
WHEEL_SITES = ["wheel_L", "wheel_R"]

# Joint ordering (matches actuator order).
# Leg joints: abad, hip, knee (per side) — these control leg pose.
# Wheel joints: wheel (per side) — these drive locomotion.
LEG_JOINT_NAMES = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint",
]
WHEEL_JOINT_NAMES = ["wheel_L_Joint", "wheel_R_Joint"]
ALL_JOINT_NAMES = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
]
