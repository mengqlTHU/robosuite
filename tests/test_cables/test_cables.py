import numpy as np
import robosuite as suite
from robosuite import load_part_controller_config, load_composite_controller_config
from robosuite.utils.transform_utils import quat2axisangle, quat_multiply
import time
import mujoco

# Load the desired controller config with default Basic controller
# config = load_part_controller_config(default_controller="OSC_POSE")
config = load_composite_controller_config(controller="BASIC")
config['body_parts']['right']["input_type"] = 'absolute'
config['body_parts']['right']["input_ref_frame"] = 'world'
config['body_parts']['right']["kp"] = 40
config['body_parts']['right']["damping_ratio"] = 2


print(config)

# create environment instance
env = suite.make(
    env_name="CableManipulation", # try with other tasks like "Stack" and "Door"
    robots="Aubo_I10",  # try with other robots like "Sawyer" and "Jaco"
    renderer="mjviewer",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    controller_configs = config
)


mj_viewer = env.viewer

# reset the environment
env.reset()
obs = None
phases = ['approach','close','lift','move','down','open']
now_phase = 1
original_cube_pos = None
original_cube_quat = None
finished = False

start_time = time.perf_counter()
last_phase_time = 0
while True:
    if obs:
        action = np.zeros(7)
        sim_time = env.sim.data.time
        if now_phase == 1:
            action[0:3] = obs['cube_pos']
            action[1] += 0.1
            action[2] += 0.04
            action[3:6] = quat2axisangle(quat_multiply(obs['cube_quat'],[1,0,0,0]))
            action[6] = -1
            # if max(obs['robot0_joint_vel'])< 0.005 and np.linalg.norm(obs['gripper_to_cube_pos'])<0.1:
            if max(obs['robot0_joint_vel'])< 0.005 and sim_time>1.0:
                last_phase_time = sim_time
                now_phase = 2
        elif now_phase == 2:
            action[0:3] = obs['cube_pos']
            action[1] += 0.1
            action[2] += 0.005
            action[3:6] = quat2axisangle(quat_multiply(obs['cube_quat'],[1,0,0,0]))
            action[6] = -1
            if max(obs['robot0_joint_vel'])< 0.005 and sim_time - last_phase_time>1.0:
                now_phase = 3
        elif now_phase == 3:
            action[0:3] = obs['cube_pos']
            action[1] += 0.1
            action[2] += 0.005
            action[3:6] = quat2axisangle(quat_multiply(obs['cube_quat'],[1,0,0,0]))
            action[6] = 1
            if obs['robot0_gripper_qpos'][0] > 0.48:
                original_cube_pos = obs['cube_pos']
                original_cube_quat = obs['cube_quat']
                now_phase = 4
        elif now_phase == 4:
            action[0:3] = original_cube_pos
            action[1] += 0.1
            action[0] += 0.02*np.sin(2*sim_time)
            action[2] += 0.2
            action[3:6] = quat2axisangle(quat_multiply(original_cube_quat,[1,0,0,0]))
            action[6] = 1
            
    else:
        action = np.zeros(7)

    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display

    sim_time = env.sim.data.time
    elapsed_time = time.perf_counter() - start_time

    # while elapsed_time < sim_time:
    #     elapsed_time = time.perf_counter() - start_time