import pickle
import time

import grpc
import numpy as np
import torch

from .base import Policy, WebsocketServicePolicy, ZMQServicePolicy
from .lerobot.helpers import RemotePolicyConfig, TimedObservation
from .lerobot.transport import services_pb2, services_pb2_grpc
from .lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from .openpi import image_tools


class Gr00t16ServicePolicyClient(ZMQServicePolicy):
    """
    Service policy client for GR00T N1.6: https://github.com/NVIDIA/Isaac-GR00T
    Target commit: https://github.com/NVIDIA/Isaac-GR00T/commit/e8e625f4f21898c506a1d8f7d20a289c97a52acf
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 5000,
        camera_keys: list[str] = ["base", "wrist"],
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            timeout_ms: Timeout of the policy server.
            modality_keys: Keys of the modality.
        """
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
        self.camera_keys = camera_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        # Build the 'video' dictionary: {camera_name: (B, T, H, W, 3), dtype=uint8}
        video = {
            camera_key: np.expand_dims(observation_dict[camera_key].cpu().numpy().astype(np.uint8), axis=0)
            for camera_key in self.camera_keys
        }
        # Build the 'state' dictionary (single_arm, gripper)
        state = {}
        joint_pos = observation_dict["joint_pos"][:, :6]
        # joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        # Add a new axis at the front (batch dim)
        joint_pos = np.expand_dims(joint_pos, axis=0)
        # Ensure joint_pos shape is (B, T, 6), we need (B, T, D) for each stream
        # e.g., single_arm: first 5 dims, gripper: last dim
        state["single_arm"] = joint_pos[..., 0:5].astype(np.float32)
        state["gripper"] = joint_pos[..., 5:6].astype(np.float32)

        # Build the 'language' dictionary
        language = {
            "annotation.human.task_description": [[observation_dict["task_description"]]],
        }

        # Compose the final observation dictionary as required
        obs_dict = {
            "video": video,
            "state": state,
            "language": language,
        }

        # the following structure is defined in modality_configs on the policy server side
        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video": {
                    "front": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8),
                    "wrist": np.zeros((1, 1, 480, 640, 3), dtype=np.uint8),
                },
                "state": {
                    "single_arm": np.zeros((1, 1, 5)),
                    "gripper": np.zeros((1, 1, 1)),
                },
                "language": {
                    "task": [["pick and place"]],
                }
            }
        """
        obs_dict = {"observation": obs_dict}
        # get the action chunk via the policy server
        action_chunk = self.call_endpoint("get_action", obs_dict)

        """
            Example of action_chunk for single arm task:
            action_chunk = [{
                "single_arm": np.zeros((1, 16, 5)),
                "gripper": np.zeros((1, 16, 1)),
            }]
        """
        action_chunk = action_chunk[0]
        concat_action = np.concatenate(
            [action_chunk["single_arm"], action_chunk["gripper"]],
            axis=-1,
        )
        # squeeze the first dimension
        concat_action = concat_action.squeeze(0)
        # concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])


class LeRobotServicePolicyClient(Policy):
    """
    Service policy client for Lerobot: https://github.com/huggingface/lerobot
    Target Commit: https://github.com/huggingface/lerobot/tree/v0.3.3

    # Change manually 01/12/2025
    Target Commit: https://github.com/huggingface/lerobot/tree/v0.4.2
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 5000,
        camera_infos: dict[str, tuple[int, int]] = {},
        policy_type: str = "smolvla",
        pretrained_name_or_path: str = "checkpoints/last/pretrained_model",
        actions_per_chunk: int = 50,
        device: str = "cuda",
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            timeout_ms: Timeout of the policy server.
            camera_infos: List of camera information. {camera_key: (height, width)}
            task_type: Type of task.
            policy_type: Type of policy.
            pretrained_name_or_path: Path to the pretrained model in the remote policy server.
            actions_per_chunk: Number of actions per chunk.
            device: Device to use.
        """
        super().__init__("service")
        service_address = f"{host}:{port}"
        self.timeout_ms = timeout_ms
        self.actions_per_chunk = actions_per_chunk

        lerobot_features = {}
        self.last_action = None
        self.rrl_m3_state_names = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'left_carriage_joint', 
                                  'lin_vel_x', 'lin_vel_y', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z'] # full ang_vel for tipping information
        
        lerobot_features["observation.state"] = {
            "dtype": "float32",
            "shape": (12,),
            "names": self.rrl_m3_state_names 
        }
        self.last_action = np.zeros((1, 10))

        self.camera_mapping = {"base": "camera1", "wrist": "camera2"} # shoulde match the renamemap in training
        for camera_key, camera_image_shape in camera_infos.items():
            mapped_key = self.camera_mapping.get(camera_key, camera_key)
            lerobot_features[f"observation.images.{mapped_key}"] = {
                "dtype": "image",
                "shape": (camera_image_shape[0], camera_image_shape[1], 3),
                "names": ["height", "width", "channels"],
            }
            # lerobot_features[f"observation.images.{camera_key}"] = {
            #     "dtype": "image",
            #     "shape": (camera_image_shape[0], camera_image_shape[1], 3),
            #     "names": ["height", "width", "channels"],
            # }
        self.camera_keys = list(camera_infos.keys())

        self.policy_config = RemotePolicyConfig(
            policy_type,
            pretrained_name_or_path,
            lerobot_features,
            actions_per_chunk,
            device,
        )
        self.channel = grpc.insecure_channel(service_address, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.latest_action_step = 0
        self.skip_send_observation = False

        self._init_service()

    def _init_service(self):
        try:
            self.stub.Ready(services_pb2.Empty())

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            print("Sending policy instructions to policy server, it may take a while...")
            self.stub.SendPolicyInstructions(policy_setup)
            print("Policy server is ready.")

        except grpc.RpcError:
            raise RuntimeError("Failed to connect to policy server")

    def _send_observation(self, observation_dict: dict):
        # raw_observation = {
        #     f"{key}": observation_dict[key].cpu().numpy().astype(np.uint8)[0] for key in self.camera_keys
        # }
        raw_observation = {
            self.camera_mapping.get(key, key): observation_dict[key].cpu().numpy().astype(np.uint8)[0] 
            for key in self.camera_keys
        }
        raw_observation["task"] = observation_dict["task_description"]

        # joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        joint_pos = observation_dict["joint_pos"]
        for joint_name in self.rrl_m3_state_names:
            raw_observation[f"{joint_name}"] = joint_pos[0, self.rrl_m3_state_names.index(joint_name)].item()

        """
            Example of raw_observation for single arm task:
            raw_observation = {
                "front": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
                "task": "pick_and_place",
            }
        """
        self.latest_action_step += 1
        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self.latest_action_step,
            must_go=True,
        )

        # send observation to policy server
        observation_bytes = pickle.dumps(observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        _ = self.stub.SendObservations(observation_iterator)

    def _receive_action(self) -> dict:
        actions_chunk = self.stub.GetActions(services_pb2.Empty())
        if len(actions_chunk.data) == 0:
            print("Received `Empty` from policy server, waiting for next call")
            return None
        return pickle.loads(actions_chunk.data)

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        if not self.skip_send_observation:
            self._send_observation(observation_dict)
        action_chunk = self._receive_action()
        if action_chunk is None:
            self.skip_send_observation = True
            return torch.from_numpy(self.last_action).repeat(self.actions_per_chunk, 1)[:, None, :]

        action_list = [action.get_action()[None, :] for action in action_chunk]
        concat_action = torch.cat(action_list, dim=0)
        # concat_action = convert_lerobot_action_to_leisaac(concat_action)

        self.last_action = concat_action[-1, :]
        self.skip_send_observation = False

        return torch.from_numpy(concat_action[:, None, :])
    

class OpenPIServicePolicyClient(WebsocketServicePolicy):
    """
    Service policy client for OpenPI: https://github.com/Physical-Intelligence/openpi
    Target Commit: https://github.com/Physical-Intelligence/openpi/commit/5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d
    Reference: https://github.com/EverNorif/openpi/tree/lerobot-v0.3.3
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        camera_keys: list[str] = ["base", "wrist"],
        api_key: str = None,
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            task_type: Type of task.
            api_key: API key of the policy server.
        """
        super().__init__(host=host, port=port, api_key=api_key)
        self.camera_keys = camera_keys

        self.camera_mapping = {"base": "base_image", "wrist": "wrist_image"}

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        
        obs_dict = {
            f"images/{self.camera_mapping.get(key, key)}": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(observation_dict[key].cpu().squeeze().numpy(), 224, 224)
            )
            for key in self.camera_keys
        }

        # joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        joint_pos = observation_dict["joint_pos"].cpu().numpy()
        obs_dict["state"] = joint_pos.squeeze().astype(np.float64)

        obs_dict["prompt"] = observation_dict["task_description"]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "images/base_image": np.zeros((224, 224, 3), dtype=np.uint8),
                "images/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                "state": np.zeros(12),
                "prompt": observation_dict["task_description"],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.infer(obs_dict)["actions"]

        """
            Example of action_chunk for single arm task:
            action_chunk: np.zeros((10, 6))
        """
        action_chunk = action_chunk[:, :10]
        processed_action = np.concatenate((action_chunk[:, -3:], 
                                           action_chunk[:, :-3]), axis=-1)

        return torch.from_numpy(processed_action[:, None, :])
