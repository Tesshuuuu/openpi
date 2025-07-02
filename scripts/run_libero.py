from serve_policy import DEFAULT_CHECKPOINT, EnvMode
import dataclasses
import tyro
from pathlib import Path

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import cv2
import mujoco
from libero.libero import get_libero_path
import os
import numpy as np
from openpi_client import image_tools

import collections

import matplotlib.pyplot as plt
from typing import Dict
import torch
import math
import copy

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    task_id: int = 0

    task_suite_name: str = "libero_10"

    instruction: str = ""

    max_steps: int = 160

    num_runs: int = 1


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class FrameSaver:
    def __init__(self, should_viz, recording_enabled_getter=None):
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.should_viz = should_viz
        self.frames = []
        self.recording_enabled_getter = recording_enabled_getter

    def update(self, img: np.ndarray):
        # Only record if recording_enabled is True (if provided)
        if self.recording_enabled_getter is not None and not self.recording_enabled_getter():
            return
        if self.should_viz:
            if not self.im:
                self.im = self.ax.imshow(img)
            else:
                self.im.set_data(img)
            plt.draw()
            plt.pause(0.1)
        self.frames.append(img)

    def write_video(self, filename, fps=10):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = self.frames[-1].shape[:2]
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        for frame in self.frames:
            # Convert from RGB to BGR if needed
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)

        self.writer.release()


class LiberoWrapper:
    resize_size: int = 224
    num_steps_wait: int = 10
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

    def __init__(
        self,
        task_suite_name: str = "libero_10",
        task_id: int = 0,
        language_instruction: str = "",
        viz=False,
    ):
        benchmark_dict = benchmark.get_benchmark_dict()
        # task_suite_name = "libero_10"  # can also choose libero_spatial, libero_object, etc.
        self.task_suite = benchmark_dict[task_suite_name]()
        self.task_id = task_id

        # retrieve a specific task
        task = self.task_suite.get_task(self.task_id)
        task_name = task.name
        self.task_description = task.language

        if language_instruction != "":
            self.task_description = language_instruction

        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        print(
            f"\n[info] retrieving task {self.task_id} from suite {task_suite_name}\n"
            + f"\nlanguage instruction is: {self.task_description}"
            + f"\n\nand the bddl file is {task_bddl_file}\n"
        )

        # step over the environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 224,  #128,
            "camera_widths": 224, #128,
            # "camera_heights": 128,
            # "camera_widths": 128,
        }
        self.env = OffScreenRenderEnv(**env_args)
        # self.env = DemoRenderEnv(**env_args)

        self.t = 0

        self.should_viz = viz
        self.recording_enabled = True
        self.viz = FrameSaver(should_viz=viz, recording_enabled_getter=lambda: self.recording_enabled)

    def reset(self):
        self.env.seed(0)
        self.env.reset()
        init_states = self.task_suite.get_task_init_states(
            self.task_id
        )  # for benchmarking purpose, we fix the a set of initial states
        init_state_id = 0
        self.env.set_init_state(init_states[init_state_id])

        # wait for sim to stabilize
        for i in range(self.num_steps_wait):
            obs, reward, done, info = self.step(self.LIBERO_DUMMY_ACTION)
            self.t += 1

        return obs

    def form_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, self.resize_size, self.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, self.resize_size, self.resize_size)
        )
        pi_observation = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(self.task_description),
        }

        return pi_observation

    def step(self, action) -> Dict[str, torch.Tensor]:
        # dummy_action = [0.0] * 7
        obs, reward, done, info = self.env.step(action)

        self.viz.update(obs["agentview_image"][::-1, ::-1])

        return obs, reward, done, info

    def close(self):
        self.env.close()


def run_task(
    *,
    policy: _policy.Policy,
    instruction: str,
    task_suite_name: str,
    task_id: str,
    replan_steps: int,
    max_steps: int,
):
    action_plan = collections.deque()
    env = LiberoWrapper(
        task_suite_name=task_suite_name,
        task_id=task_id,
        language_instruction=instruction,
    )
    obs = env.reset()
    for i in range(max_steps):
        if not action_plan:
            print(f"stepping")
            pi_obs = env.form_observation(obs)
            action_chunk = policy.infer(pi_obs)["actions"]

            action_plan.extend(action_chunk[:replan_steps])

        action = action_plan.popleft()
        obs, reward, done, info = env.step(action)

    print(f"Task finished with {i} steps and success: {done}")
    env.viz.write_video("video_.mp4")
    
    # Explicitly close the environment to avoid EGL cleanup errors
    env.close()

    return True

from typing import Set, Tuple

def collision_detect(env, model, data) -> Set[Tuple[str, str]]:
    """
    Returns a set of (geom1_name, geom2_name) for every contact 
    in data.contact that does *not* involve 'table'.
    """
    geom_pairs = set()
    sim_orig = env.env.sim

    if data.ncon > 0:
        for n in range(data.ncon):
            c = data.contact[n]
            geom1 = sim_orig.model.geom_id2name(c.geom1)
            geom2 = sim_orig.model.geom_id2name(c.geom2)    


            # Provide additional information for unnamed geoms
            if geom1 is None:
                geom1_body_id = sim_orig.model.geom_bodyid[c.geom1]
                geom1_body_name = sim_orig.model.body_id2name(geom1_body_id)
                geom1_type = sim_orig.model.geom_type[c.geom1]
                
                geom1 = geom1_body_name
            
            if geom2 is None:
                geom2_body_id = sim_orig.model.geom_bodyid[c.geom2]
                geom2_body_name = sim_orig.model.body_id2name(geom2_body_id)
                geom2_type = sim_orig.model.geom_type[c.geom2]

                geom2 = geom2_body_name
            
            if "table" not in geom1 and "table" not in geom2:
                geom_pairs.add((geom1, geom2))

                if "milk" in geom1 or "milk" in geom2:
                    print(f"Milk collision detected: {geom1} and {geom2}")

    return geom_pairs

def collision_aware_action_plan(env, nominal_plan, pi_obs, policy, replan_steps, n_samples=15, sigma=0.2):
    """
    Try your nominal_plan first; if it ever collides with 'milk', 
    sample up to n_samples noisy variants and pick the one that
    avoids milk the furthest (or is fully collision-free).
    Uses the real environment's step and state save/restore for exact matching.
    """
    sim = env.env.sim
    best_plan = copy.deepcopy(nominal_plan)
    best_score = -1   # how many steps we got before a milk collision
    all_traces = {}

    prev_recording_enabled = env.recording_enabled
    env.recording_enabled = False

    for sample_idx in range(n_samples):
        # build candidate
        if sample_idx == 0:
            candidate = copy.deepcopy(nominal_plan)
        else:
            # # randomize the action plan
            # candidate = [
            #     a + np.random.randn(*a.shape) * sigma
            #     for a in nominal_plan
            # ]

            # sample from the policy
            candidate_chunk = policy.infer(pi_obs)["actions"]
            candidate = candidate_chunk[:replan_steps]

        # Save the sim state before rolling out the candidate
        saved_state = env.env.sim.get_state()
        steps_safe = 0
        trace = []
        for t, action in enumerate(candidate):
            trace.append(env.env.sim.data.qpos.copy()[:1])
            obs, reward, done, info = env.step(action)
            contacts = collision_detect(env, env.env.sim.model, env.env.sim.data)
            if any("milk" in g1 or "milk" in g2 for g1,g2 in contacts) or any("orange" in g1 or "orange" in g2 for g1,g2 in contacts):
                print(f"Simulated Contacts: {contacts}")
                break
            steps_safe += 1
        trace.append(env.env.sim.data.qpos.copy()[:1])
        all_traces[sample_idx] = trace
        print(f"Trace {sample_idx}: {trace}")
        # Restore the sim state after candidate rollout
        env.env.sim.set_state(saved_state)
        env.env.sim.forward()
        # if it survived *all* steps, we're done
        if steps_safe == len(candidate):
            print(f"✅ Sample {sample_idx} was fully milk and orange juice free.")
            env.recording_enabled = prev_recording_enabled
            return collections.deque(candidate)
        # otherwise keep the plan that got the furthest
        if steps_safe > best_score:
            best_score = steps_safe
            best_plan  = candidate
            print(f"⚠️ Sample {sample_idx} survived {best_score}/{len(candidate)} steps.")
    print(f"▶️ Returning best plan (avoids milk for {best_score} steps).")
    env.recording_enabled = prev_recording_enabled
    return collections.deque(best_plan)



def main(args: Args):
    replan_steps = 8
    policy_config = DEFAULT_CHECKPOINT[EnvMode.LIBERO]
    print(f"Replan steps: {replan_steps}")

    policy = _policy_config.create_trained_policy(
        _config.get_config(policy_config.config), policy_config.dir
    )

    action_plan = collections.deque()
    env = LiberoWrapper(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        language_instruction=args.instruction,
    )
    obs = env.reset()
    # TODO: add a loop to run the task for multiple times
    for j in range(args.num_runs):
        for i in range(args.max_steps):
            if not action_plan:
                pi_obs = env.form_observation(obs)
                action_chunk = policy.infer(pi_obs)["actions"]

                action_plan.extend(action_chunk[:replan_steps])

                # check if the action plan has no collision
                print(f"--- Collision Aware Action Plan ---")
                action_plan = collision_aware_action_plan(env, action_plan, pi_obs, policy, replan_steps)

            action = action_plan.popleft()
            # action = collision_aware_action(env, action)
            obs, reward, done, info = env.step(action)

            # print the state of the sim
            print(f"--- Real Simulation ---")
            print(f"Step {i}: State: {env.env.sim.data.qpos[:1]}")

            geom_lst = collision_detect(env, env.env.sim.model, env.env.sim.data)
            # print only the geom pairs that contain "milk" and "cheese"
            if "milk" in geom_lst:
                print(f"Collision detected: {geom_lst}")

        print(f"Iteration {j}: Finished at step {i} with status {done}")

        Path("./videos").mkdir(exist_ok=True)
        env.viz.write_video(f"videos/video_{env.task_description.replace(' ', '_')}_{j}.mp4")
    
    # Explicitly close the environment to avoid EGL cleanup errors
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))