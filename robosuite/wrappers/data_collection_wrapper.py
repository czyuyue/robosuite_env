"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import json
import os
import time
import cv2

import numpy as np

from robosuite.utils.mjcf_utils import save_sim_model
from robosuite.wrappers import Wrapper
from robosuite.utils.camera_utils import transform_from_pixels_to_world, get_camera_extrinsic_matrix,get_pointcloud_from_image_and_depth
from robosuite.utils.camera_utils import project_points_from_world_to_camera,get_camera_intrinsic_matrix

def pixel_to_3d_with_depth(pixel_coord, depth, camera_matrix, camera_pose):
    """
    使用深度信息将单个相机的像素坐标映射到3D空间坐标

    Args:
        pixel_coord (np.ndarray): 像素坐标 [u, v]
        depth (float): 深度值（单位：米）
        camera_matrix (np.ndarray): 相机内参矩阵 3x3
        camera_pose (np.ndarray): 相机外参矩阵 4x4

    Returns:
        np.ndarray: 3D空间坐标 [x, y, z]
    """
    # 获取相机内参矩阵的参数
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 将像素坐标转换为相机坐标系下的归一化坐标
    x = (pixel_coord[0] - cx) / fx
    y = (pixel_coord[1] - cy) / fy

    # 构建相机坐标系下的3D点
    point_camera = np.array([x * depth, y * depth, depth])

    # 获取相机外参矩阵的旋转和平移
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]

    # 将点从相机坐标系转换到世界坐标系
    point_world = R @ point_camera + t

    return point_world

# def example_usage():
#     """
#     示例用法
#     """
#     # 示例相机内参矩阵
#     K1 = np.array([
#         [800, 0, 320],
#         [0, 800, 240],
#         [0, 0, 1]
#     ])

#     K2 = np.array([
#         [800, 0, 320],
#         [0, 800, 240],
#         [0, 0, 1]
#     ])

#     K3 = np.array([
#         [800, 0, 320],
#         [0, 800, 240],
#         [0, 0, 1]
#     ])

#     # 示例相机外参矩阵
#     R1 = np.eye(3)
#     t1 = np.array([0, 0, 0])
#     T1 = np.vstack([np.hstack([R1, t1.reshape(3, 1)]), [0, 0, 0, 1]])

#     R2 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
#     t2 = np.array([1, 0, 0])
#     T2 = np.vstack([np.hstack([R2, t2.reshape(3, 1)]), [0, 0, 0, 1]])

#     R3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#     t3 = np.array([0, 1, 0])
#     T3 = np.vstack([np.hstack([R3, t3.reshape(3, 1)]), [0, 0, 0, 1]])

#     # 示例像素坐标
#     pixel1 = [320, 240]  # 相机1的像素坐标
#     pixel2 = [320, 240]  # 相机2的像素坐标
#     pixel3 = [320, 240]  # 相机3的像素坐标

#     # 调用函数
#     point_3d = pixel_to_3d(
#         [pixel1, pixel2, pixel3],
#         [K1, K2, K3],
#         [T1, T2, T3]
#     )

#     print(f"3D点坐标: {point_3d}")

def example_usage_with_depth():
    """
    使用深度信息的示例用法
    """
    # 示例相机内参矩阵
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])

    # 示例相机外参矩阵
    R = np.eye(3)
    t = np.array([0, 0, 0])
    T = np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]])

    # 示例像素坐标和深度值
    pixel = np.array([320, 240])  # 图像中心点
    depth = 1.0  # 1米深度

    # 调用函数
    point_3d = pixel_to_3d_with_depth(pixel, depth, K, T)
    print(f"3D点坐标: {point_3d}")

class DataCollectionWrapper(Wrapper):
    def __init__(self, env, directory, collect_freq=1, flush_freq=100, env_id=0):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)
        
        self.env_id = env_id  # environment ID for multi-env setups

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.states = []
        self.action_infos = []  # stores information about actions taken
        self.keypoints_pos = []  # stores keypoint positions
        self.images = []  # stores collected images
        self.depths = []  # stores collected depth maps
        self.successful = False  # stores success state of demonstration

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        # 关键点相关的属性
        self.keypoints = None
        self._keypoint_registry = None
        self._keypoint2object = None

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None
        self._current_task_instance_xml = None

        ## get all camera names
        self.cam_names = [cam for cam in self.env.sim.model.camera_names]
        print("cam_names: ", self.cam_names)

        ### get camera position


        # import pdb; pdb.set_trace()

        #### i want to test the transform_from_pixels_to_world function
        # pixel = np.array([320, 240])  # 图像中心点
        # depth = 1.0  # 1米深度
        # ###

        # # 获取相机内参矩阵
        # camera_height = self.env.sim.model.vis.global_.offheight
        # camera_width = self.env.sim.model.vis.global_.offwidth
        # camera_name = self.cam_names[0]  # 使用第一个相机

        # # 获取相机到世界坐标系的变换矩阵
        # camera_to_world_transform = get_camera_extrinsic_matrix(self.env.sim, camera_name)

        # # 将深度值转换为深度图格式
        # depth_map = np.ones((camera_height, camera_width, 1)) * depth

        # # 将像素坐标转换为世界坐标
        # pixels = np.array([[pixel]])  # 需要匹配函数期望的形状 [..., 2]
        # point_3d = transform_from_pixels_to_world(pixels, depth_map, camera_to_world_transform)[0]
        # print("point_3d: ", point_3d)



    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)

        # NOTE: was previously self.env.model.get_xml(). Was causing the following issue in rare cases:
        # ValueError: Error: eigenvalues of mesh inertia violate A + B >= C
        # switching to self.env.sim.model.get_xml() does not create this issue
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())

        # trick for ensuring that we can play MuJoCo demonstrations back
        # deterministically by using the recorded actions open loop
        # self.env.set_ep_meta(self.env.get_ep_meta())
        # self.env.reset_from_xml_string(self._current_task_instance_xml)
        # self.env.sim.reset()
        # self.env.sim.set_state_from_flattened(self._current_task_instance_state)
        # self.env.sim.forward()

    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, "ep_id{}_{}".format(self.env_id, t1))
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

        # save the episode info to json file
        ep_meta_path = os.path.join(self.ep_directory, "ep_meta.json")
        with open(ep_meta_path, "w") as f:
            json.dump(self.env.get_ep_meta(), f)

        # save initial state and action
        assert len(self.states) == 0
        self.states.append(self._current_task_instance_state)

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        ### only flush when the episode is done
        print("flush????\n")
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, "state_id{}_{}.npz".format(self.env_id, t1))
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        print("env_name: ", env_name)
        print("saving to: ", state_path)
        
        # Prepare data dictionary for saving
        save_data = {
            "states": np.array(self.states),
            "action_infos": self.action_infos,
            "successful": self.successful,
            "keypoint_positions": np.array(self.keypoints_pos),
            "env": env_name,
        }
        ### save image and depth data as png
        # if len(self.images) > 0:
        #     for i in range(len(self.images)):
        #         img = self.images[i]["agentview"]
        #         depth = self.depths[i]["agentview"]
        #         cv2.imwrite(os.path.join(self.ep_directory, f"image_{i}.png"), img)
        #         cv2.imwrite(os.path.join(self.ep_directory, f"depth_{i}.png"), depth)
        ###### TEMPORARY DISABLED IMAGE AND DEPTH DATA
        # Add image and depth data if available 
        # if len(self.images) > 0:
        #     save_data["images"] = np.array(self.images)
        #     print(f"Saving {len(self.images)} images with shape: {np.array(self.images).shape}")
        
        ## convert the depth data to uint16
        depth_agentview = [d["agentview"] for d in self.depths]
        depth_agentview = np.array(depth_agentview)
        print("depth_agentview: ", depth_agentview.shape)
        # import pdb; pdb.set_trace()
        depth_min = np.min(depth_agentview)
        depth_max = np.max(depth_agentview)
        
        # Convert depth from min-max range to uint16 (0-65535)
        depth_agentview_normalized = ((depth_agentview - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)

        ## save the depth_min and depth_max to the depth_min_max.npz
        np.savez(os.path.join(self.ep_directory, "depth_min_max.npz"), depth_min=depth_min, depth_max=depth_max)
        ## save it to avi use FFv1
        video_path = os.path.join(self.ep_directory, "depth.avi")
        import imageio
        imageio.mimwrite(video_path, depth_agentview_normalized, fps=10.0, codec='ffv1', pixelformat='gray16le')
        ## save the depth_agentview to the depth_agentview.npz
        # np.savez(os.path.join(self.ep_directory, "depth_agentview.npz"), depth_agentview=depth_agentview)
        
        

        # if len(self.depths) > 0:
        #     save_data["depths"] = np.array(self.depths)
        #     print(f"Saving {len(self.depths)} depth maps with shape: {np.array(self.depths).shape}")
        
        np.savez(state_path, **save_data)
        ## save keypoint_registry (convert numpy types to python types for JSON serialization)
        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        
        keypoint_registry_serializable = convert_numpy_types(self._keypoint_registry)
        keypoint2object_serializable = convert_numpy_types(self._keypoint2object)
        
        with open(os.path.join(self.ep_directory, "keypoint_registry.json"), "w") as f:
            json.dump(keypoint_registry_serializable, f, indent=2)
        ## save keypoint2object
        with open(os.path.join(self.ep_directory, "keypoint2object.json"), "w") as f:
            json.dump(keypoint2object_serializable, f, indent=2)
        # Save video of images
        if len(self.images) > 0:
            # Get video writer
            video_path = os.path.join(self.ep_directory, "video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
            
            # Write each frame
            for i in range(len(self.images)):
                img = self.images[i]["agentview"]
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writer.write(img_bgr)
            
            # Release video writer
            video_writer.release()
            print(f"Saved video to {video_path}")
        # Clear all data
        self.states = []
        self.action_infos = []
        self.keypoints_pos = []
        self.images = []
        self.depths = []
        self.successful = False

    def register_keypoints(self, pixels,points_3d,segmentation):
        """
        注册关键点到物体上，找到最接近的几何体

        Args:
            keypoints (np.ndarray): 世界坐标系中的关键点，形状为 (N, 3)
        Returns:
            None
        """
        if not isinstance(points_3d, np.ndarray):
            points_3d = np.array(points_3d)
        self.keypoints = points_3d
        self._keypoint_registry = dict()
        self._keypoint2object = dict()
        # 不考虑这些物体作为关键点关联对象
        # exclude_names = ['wall', 'floor', 'fixed', 'table', 'robot', 'target']
        exclude_names = []
        # import pdb; pdb.set_trace()


        ## get the segmentation mask
        # segmentation, _ = self.env.sim.render(width=640, height=480, camera_name="agentview", segmentation=True,depth=True)
        # segmentation = segmentation[:,:,1]
        ### init image
        segmentation = segmentation[:,:,1]
        image = self.env.sim.render(width=640, height=480, camera_name="agentview")

        world_to_camera_transform = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        # print("world_to_camera_transform: ", world_to_camera_transform)
        K = get_camera_intrinsic_matrix(self.env.sim, "agentview",480,640)
        
        obs = self.env._get_observations()
        print("obs: ", obs)
        ### print pos of all cubes
        # for key in obs:
        #     if "cube" in key and "pos" in key:
        #         print("cube: ", key)
        #         print("cube pos: ", obs[key])
        #         import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        import cv2
        for i in range(len(pixels)):
            pixel = pixels[i]
            keypoint = points_3d[i]
            y, x = pixel[0], pixel[1]
            
            ### add a circle to the image
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            # print("y, x: ", y, x)
            # print("segmentation: ", segmentation.shape)
            # import pdb; pdb.set_trace()
            geom_id = segmentation[y, x]
            points_2d = project_points_from_world_to_camera(keypoint, world_to_camera_transform,480,640, K)
            # print("points_2d: ", points_2d)
            # print("pixel: ", pixel)
            # print("keypoint: ", keypoint)
            # import pdb; pdb.set_trace()
            if geom_id < len(self.env.sim.model.geom_names):
                geom_name = self.env.sim.model.geom_id2name(geom_id)
                if "table" in geom_name:
                    continue
                if "floor" in geom_name:
                    continue
                if "wall" in geom_name:
                    continue
                if "fixed" in geom_name:
                    continue
                if "robot" in geom_name:
                    continue
              
                # if "cube" in geom_name:
                # print(f"Pixel ({x}, {y}):")
                # print(f"  - geom_id: {geom_id}")
                # print(f"  - geom_name: {geom_name}")
                # print(f"  - body_id: {self.env.sim.model.geom_bodyid[geom_id]}")
                body_id = self.env.sim.model.geom_bodyid[geom_id]
                body_name = self.env.sim.model.body_id2name(body_id)
                ### add text to the image
                # cv2.putText(image, geom_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # print(f"  - body_name: {body_name}")
                ## register the keypoint
                print("geom_name: ", geom_name)
                # import pdb; pdb.set_trace()
                self._keypoint_registry[i] = {
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "body_id": body_id,
                    "body_name": body_name,
                    "initial_keypoint": keypoint,
                    "initial_pos": np.copy(self.env.sim.data.get_geom_xpos(geom_name)),
                    "initial_rot": np.copy(self.env.sim.data.get_geom_xmat(geom_name).reshape(3, 3)),
                    "keypoint_offset": keypoint - self.env.sim.data.get_geom_xpos(geom_name)
                }
                distance = np.linalg.norm(keypoint - self.env.sim.data.get_geom_xpos(geom_name))
                # if "cube" in geom_name:
                #     print("cube: ", geom_name)
                #     print("distance: ", distance)
                #     print("keypoint: ", keypoint)
                #     print("initial_pos: ", self.env.sim.data.get_geom_xpos(geom_name))
                #     import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
            if i % 10 == 0:
                cv2.imwrite("image_vis.png", image)
                image = self.env.sim.render(width=640, height=480, camera_name="agentview")
                # import pdb; pdb.set_trace()
        ######



        # 获取所有几何体的ID和名称
        # geom_ids = []
        # geom_names = []
        # for i in range(self.env.sim.model.ngeom):
        #     geom_ids.append(i)
        #     geom_names.append(self.env.sim.model.geom_id2name(i))

        # # 获取所有物体的ID和名称
        # body_ids = []
        # body_names = []
        # for i in range(self.env.sim.model.nbody):
        #     body_ids.append(i)
        #     body_names.append(self.env.sim.model.body_id2name(i))

        # # 为每个关键点找到最近的几何体
        # for idx, keypoint in enumerate(keypoints):
        #     closest_distance = np.inf
        #     closest_geom_id = None
        #     closest_body_id = None

        #     # 遍历所有几何体
        #     for i, geom_id in enumerate(geom_ids):
        #         geom_name = geom_names[i]
        #         # 跳过不需要考虑的几何体
        #         if any([name in geom_name.lower() for name in exclude_names if name != ""]):
        #             continue

        #         # 获取几何体当前的世界坐标位置
        #         geom_pos = self.env.sim.data.get_geom_xpos(geom_name)

        #         # 计算距离
        #         distance = np.linalg.norm(geom_pos - keypoint)
        #         if distance < closest_distance:
        #             closest_distance = distance
        #             closest_geom_id = geom_id
        #             closest_geom_name = geom_name

        #     # 找到几何体所属的身体
        #     if closest_geom_id is not None:
        #         # 获取这个几何体所属的身体ID
        #         body_id = self.env.sim.model.geom_bodyid[closest_geom_id]
        #         closest_body_id = body_id
        #         closest_body_name = body_names[body_ids.index(body_id)]
        #         ##
        #         print("keypoint: ", keypoint)
        #         print("distance: ", distance)
        #         print("closest_geom_pos: ", self.env.sim.data.get_geom_xpos(closest_geom_name))
        #         print("closest_body_name: ", closest_body_name)

        #         # 记录关键点对应的几何体和初始位姿
        #         self._keypoint_registry[idx] = {
        #             "geom_id": closest_geom_id,
        #             "geom_name": closest_geom_name,
        #             "body_id": closest_body_id,
        #             "body_name": closest_body_name,
        #             "initial_pos": np.copy(self.env.sim.data.get_geom_xpos(closest_geom_name)),
        #             "initial_rot": np.copy(self.env.sim.data.get_geom_xmat(closest_geom_name).reshape(3, 3)),
        #             "keypoint_offset": keypoint - self.env.sim.data.get_geom_xpos(closest_geom_name)
        #         }
        #         self._keypoint2object[idx] = closest_body_name
        # import pdb; pdb.set_trace()
        # 更新关键点位置为对应几何体的位置
        for idx in self._keypoint_registry:
            self.keypoints[idx] = self._keypoint_registry[idx]["initial_pos"] + self._keypoint_registry[idx]["keypoint_offset"]

    def get_keypoint_positions(self, check=False):
        """
        获取已注册关键点的当前位置

        Returns:
            np.ndarray: 世界坐标系中的关键点位置，形状为 (N, 3)
        """
        assert hasattr(self, '_keypoint_registry') and self._keypoint_registry is not None, "关键点尚未注册"
        keypoint_positions = []
        for idx in self._keypoint_registry:
            reg = self._keypoint_registry[idx]
            current_pos = self.env.sim.data.get_geom_xpos(reg["geom_name"])
            current_rot = self.env.sim.data.get_geom_xmat(reg["geom_name"]).reshape(3, 3)

            # 计算关键点相对位置的旋转
            initial_rot_inv = np.linalg.inv(reg["initial_rot"])
            relative_rot = np.dot(current_rot, initial_rot_inv)

            # 旋转关键点的偏移量并加到当前位置上
            rotated_offset = np.dot(relative_rot, reg["keypoint_offset"])
            keypoint_pos = current_pos + rotated_offset

            keypoint_positions.append(keypoint_pos)
            initial_pos = reg["initial_pos"]
            initial_keypoint = reg["initial_keypoint"]

            # if check and np.linalg.norm(keypoint_pos - initial_keypoint) > 0.01:

            #     print("object name: ", reg["geom_name"])
            #     print("current_pos: ", current_pos)
            #     print("initial_pos: ", initial_pos)
            #     print("keypoint_pos: ", keypoint_pos)
            #     print("initial_keypoint: ", initial_keypoint)
            #     distance_now = np.linalg.norm(keypoint_pos - current_pos)
            #     distance_initial = np.linalg.norm(initial_keypoint - initial_pos)
            #     print("distance_now: ", distance_now)
            #     print("distance_initial: ", distance_initial)
            #     import pdb; pdb.set_trace()
        return np.array(keypoint_positions)

    def get_object_by_keypoint(self, keypoint_idx):
        """
        获取与关键点关联的物体名称

        Args:
            keypoint_idx (int): 关键点的索引
        Returns:
            str: 与关键点关联的物体名称
        """
        assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "关键点尚未注册"
        return self._keypoint2object[keypoint_idx]

    def get_keypoints_from_pixels(self, pixels, depths, camera_name=None):
        """
        从像素坐标和深度信息获取世界坐标系中的关键点

        Args:
            pixels (np.ndarray): 像素坐标，形状为 (N, 2)
            depths (np.ndarray): 对应的深度值，形状为 (N,)
            camera_name (str, optional): 相机名称，如果为None则使用第一个相机

        Returns:
            np.ndarray: 世界坐标系中的关键点，形状为 (N, 3)
        """
        if camera_name is None:
            camera_name = self.cam_names[0]

        # 获取相机到世界坐标系的变换矩阵
        world_to_camera_transform = get_camera_extrinsic_matrix(self.env.sim, camera_name)

        camera_to_world_transform = np.linalg.inv(world_to_camera_transform)

        # 将深度值转换为深度图格式

        # 将像素坐标转换为世界坐标
        points_3d = transform_from_pixels_to_world(pixels, depths, camera_to_world_transform)
        return points_3d


    def render_with_keypoints(self, check=False):
        """
        渲染环境并绘制关键点
        """
        # 获取当前关键点位置
        keypoints = self.get_keypoint_positions(check)
        ## 绘制关键点
        import cv2
        import numpy as np

        # 获取当前图像
        image = self.env.sim.render(width=640, height=480, camera_name="agentview")
        ## flip vertically
        image = cv2.flip(image, 0)
        ## 绘制关键点
        ## project the keypoints to the image
        ## 使用相机内参矩阵和外参矩阵将关键点投影到图像上
        world_to_camera_transform = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        # print("world_to_camera_transform: ", world_to_camera_transform)
        K = get_camera_intrinsic_matrix(self.env.sim, "agentview",480,640)
        points_2d = project_points_from_world_to_camera(keypoints, world_to_camera_transform,480,640, K)
        # print("points_2d: ", points_2d[0][:100])
        if len(points_2d.shape) > 2:
            points_2d = points_2d.squeeze()
        # import pdb; pdb.set_trace()
        ## 绘制关键点
        ## flip the image vertically
        # image = cv2.flip(image, 0)
        ## flip the image horizontally
        # image = cv2.flip(image, 1)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for point in points_2d:
            # print("point: ", point)
            cv2.circle(image, (int(point[1]), int(point[0])), 5, (0, 0, 255), -1)
        ## save the image
        cv2.imwrite("keypoints.png", image)
        
        # Convert back to RGB for consistency with video frames
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb







    def reset(self):
        """
        重置环境并注册关键点

        Returns:
            OrderedDict: 重置后的环境观察空间
        """
        self.env.unset_ep_meta()  # 清除之前设置的episode元数据
        ret = super().reset()

        ##get agentview image and depth
        self._pixels, self._depths = self.env.sim.render(width=640, height=480, camera_name="agentview", depth=True)
        self._segmentation, _ = self.env.sim.render(width=640, height=480, camera_name="agentview", segmentation=True,depth=True)
        ### 翻转图像 VERY IMPORTANT
        self._pixels = cv2.flip(self._pixels, 0)
        self._depths = cv2.flip(self._depths, 0)
        self._segmentation = cv2.flip(self._segmentation, 0)
        
        # world_to_camera_transform = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        # camera_to_world_transform = np.linalg.inv(world_to_camera_transform)


        geom_id = self.env.sim.model.geom_name2id("table_leg3_visual")
        geom_rgba = self.env.sim.model.geom_rgba[geom_id]
        ## modify its group id
        # self.env.sim.model.geom_group[geom_id] = 4
#         import mujoco

# # 设置 camera 渲染选项，只显示 group 0 的 geom
#         opt = mujoco.MjvOption()
#         mujoco.mjv_defaultOption(opt)
#         opt.geomgroup[4] = False 
        # print(f"table_leg3_visual RGBA: {geom_rgba}")
        # print(f"table_leg3_visual group id: {self.env.sim.model.geom_group[geom_id]}")

        
        # import pdb; pdb.set_trace()
        ## save the segmentation map
        # print("self._segmentation: ", self._segmentation.shape)
        ## shape (480, 640, 2)
        ## save the segmentation map
        # cv2.imwrite("segmentation_mask1.png", self._segmentation[:,:,0]*20)
        # cv2.imwrite("segmentation_mask2.png", self._segmentation[:,:,1]*20)
        # Get unique values in mask2
        # unique_values = np.unique(self._segmentation[:,:,1])
        # ## unique_values for another half
        # unique_values_2 = np.unique(self._segmentation[:,:,0])
        # print("unique_values_2: ", unique_values_2)
        # for body_id in unique_values_2:
        #     name = self.env.sim.model.body_id2name(body_id)
        #     print(f"Body ID {body_id}: {name}")
        # # Get all geoms that belong to body ID 6
        # body_id = 6
        # print(f"\n=== Geoms belonging to body ID {body_id} ===")
        # for geom_id in range(self.env.sim.model.ngeom):
        #     if self.env.sim.model.geom_bodyid[geom_id] == body_id:
        #         geom_name = self.env.sim.model.geom_id2name(geom_id)
        #         print(f"Geom ID {geom_id}: {geom_name}")
        # # Get all pixels where body_id is 6
        # body_6_pixels = np.where(self._segmentation[:,:,0] == 6)
        
        # Sample a few points from these pixels
        # num_samples = 5
        # if len(body_6_pixels[0]) > 0:
        #     indices = np.random.choice(len(body_6_pixels[0]), min(num_samples, len(body_6_pixels[0])), replace=False)
            
        #     print("\n=== Sampling points from body_id 6 ===")
        #     for idx in indices:
        #         y, x = body_6_pixels[0][idx], body_6_pixels[1][idx]
        #         # Get the geom_id at this pixel from the second channel
        #         geom_id = self._segmentation[y, x, 1]
        #         print(f"Pixel ({x}, {y}):")
        #         print(f"  - geom_id: {geom_id}")
        #         if geom_id < len(self.env.sim.model.geom_names):
        #             geom_name = self.env.sim.model.geom_id2name(geom_id)
        #             print(f"  - geom_name: {geom_name}")
        # import pdb; pdb.set_trace()
        
        # Create a copy of the image for visualization
        # mask2_vis = self._segmentation[:,:,1].copy()*20
        
        # # For each unique value (region)
        # for val in unique_values:
        #     if val == 0:  # Skip background
        #         continue
                
        #     # Find all pixels with this value
        #     region_pixels = np.where(self._segmentation[:,:,1] == val)
            
        #     # Find middle point of the region
        #     y_middle = region_pixels[0][len(region_pixels[0])//2]
        #     x_middle = region_pixels[1][len(region_pixels[1])//2]
            
        #     # Add number label at middle point
        #     cv2.putText(mask2_vis, str(int(val)), (x_middle, y_middle), 
        #                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        
        # Save the labeled mask
        # cv2.imwrite("segmentation_mask2_labeled.png", mask2_vis)

        # 分析所有可见的几何体
        # print("\n=== 分析所有可见几何体 ===")
        
        # for target_geom_id in unique_values:
        #     if target_geom_id == 0:  # Skip background
        #         continue
                
        #     print(f"\n=== 分析 Geom ID {target_geom_id} ===")
            
        #     try:
        #         # 获取几何体名称
        #         if target_geom_id < len(self.env.sim.model.geom_names):
        #             geom_name = self.env.sim.model.geom_id2name(target_geom_id)
        #             print(f"Geom ID {target_geom_id} 名称: {geom_name}")
                    
        #             # 获取几何体类型
        #             geom_type = self.env.sim.model.geom_type[target_geom_id]
        #             geom_type_names = {0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule', 
        #                                4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'}
        #             print(f"几何体类型: {geom_type_names.get(geom_type, 'unknown')} (code: {geom_type})")
                    
        #             # 获取几何体位置
        #             geom_pos = self.env.sim.data.geom_xpos[target_geom_id]
        #             print(f"世界坐标位置: {geom_pos}")
                    
        #             # 获取几何体大小
        #             geom_size = self.env.sim.model.geom_size[target_geom_id]
        #             print(f"几何体大小: {geom_size}")
                    
        #             # 获取几何体所属的body
        #             geom_bodyid = self.env.sim.model.geom_bodyid[target_geom_id]
        #             if geom_bodyid < len(self.env.sim.model.body_names):
        #                 body_name = self.env.sim.model.body_id2name(geom_bodyid)
        #                 print(f"所属Body: {body_name} (ID: {geom_bodyid})")
                    
        #             # 获取几何体颜色/材质
        #             geom_rgba = self.env.sim.model.geom_rgba[target_geom_id]
        #             print(f"颜色 (RGBA): {geom_rgba}")
                    
        #             # 查看这个geom在分割图中占据的像素数量
        #             geom_pixels = np.sum(self._segmentation[:,:,1] == target_geom_id)
        #             print(f"在分割图中占据的像素数: {geom_pixels}")
                    
        #         else:
        #             print(f"Geom ID {target_geom_id} 超出范围")
                    
        #     except Exception as e:
        #         print(f"获取geom信息时出错: {e}")
        
        # print("=== 所有几何体列表 ===")
        # try:
        #     for i in range(min(20, len(self.env.sim.model.geom_names))):  # 只显示前20个
        #         geom_name = self.env.sim.model.geom_id2name(i)
        #         geom_pixels = np.sum(self._segmentation[:,:,1] == i)
        #         if geom_pixels > 0:  # 只显示在图像中可见的几何体
        #             print(f"ID {i}: {geom_name} ({geom_pixels} pixels)")
        # except Exception as e:
        #     print(f"列出几何体时出错: {e}")

        # import pdb; pdb.set_trace()
        ## save the picture of agentview
        cv2.imwrite("agentview.png", self._pixels)
        ## save the depth map
        self._start_new_episode()

        # 如果有像素坐标和深度信息，则注册关键点
        # print("self._pixels: ", self._pixels.shape)
        # print("self._depths: ", self._depths.shape)
        # print("HERE!!!!\n")
        # import pdb; pdb.set_trace()



        #####
        points_3d , cam_points = get_pointcloud_from_image_and_depth(self.env.sim, self._pixels, self._depths,"agentview")
        print("points_3d: ", points_3d.shape)
        points_3d = points_3d.reshape(480, 640, -1)
        cam_points = cam_points.reshape(480, 640, -1)
        points_3d = points_3d[:,:, :3]
        # cam_points = cam_points[:,:, :3]
        # cam_points = cam_points.reshape(-1, 3)
        ## 可视化点云
        ## 添加 import 路径

        middle_point = points_3d[240, 320, :]
        middle_cam_point = cam_points[240, 320, :]
        print("middle_cam_point: ", middle_cam_point)
        print("middle_point: ", middle_point)
        print("middle depth: ", self._depths[240, 320])


        ## transform the middle point into pixels
        world_to_camera_transform = get_camera_extrinsic_matrix(self.env.sim, "agentview")
        K = get_camera_intrinsic_matrix(self.env.sim, "agentview",480,640)
        pixels = project_points_from_world_to_camera(middle_point.reshape(1, 3), world_to_camera_transform,480,640, K)
        print("pixels transformed from middle point: ", pixels)
        # import pdb; pdb.set_trace()



        # import pdb; pdb.set_trace()

        print("type of points_3d: ", type(points_3d))
        print("dtype of points_3d: ", points_3d.dtype)
        points_3d = points_3d.reshape(-1, 3)
        ## 输出每一维的min max 一共三维 N
        print("points_3d.min(): ", points_3d.min(axis=0))
        print("points_3d.max(): ", points_3d.max(axis=0))
        ## downsampling to 50000 points
        points_3d_vis = points_3d[np.random.choice(points_3d.shape[0], 50000, replace=False)]
        print("points_3d_vis.shape: ", points_3d_vis.shape)
        import sys
        sys.path.append("/home/yunzhe/zzzzzworkspaceyy/robosuite_data/robosuite")

        colors = np.zeros((points_3d_vis.shape[0], 3))
        ## grey 
        colors[:, 0] = 0.5
        colors[:, 1] = 0.5
        colors[:, 2] = 0.5
        ### 可视化的时候 把 cube的 位置也画出来 red color
        obs = self.env._get_observations()
        for key in obs:
            if "cube" in key and "pos" in key and (not "gripper" in key):
                pos = obs[key]
                print("key: ", key)
                print("pos: ", pos)
                ## make it large ball by adding multiple points around the center
                # Create a small sphere of points around the cube position
                num_points = 100  # Number of points to create the sphere
                radius = 0.02  # Radius of the sphere
                
                # Generate random points on a sphere
                phi = np.random.uniform(0, 2*np.pi, num_points)
                theta = np.random.uniform(0, np.pi, num_points)
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                
                # Add the sphere points to the point cloud
                sphere_points = np.stack([x, y, z], axis=1) + pos
                points_3d_vis = np.concatenate([points_3d_vis, sphere_points], axis=0)
                sphere_colors = np.tile(np.array([1, 0, 0]), (num_points, 1))  # Red color for all sphere points
                colors = np.concatenate([colors, sphere_colors], axis=0)
        # import pdb; pdb.set_trace()
        print("points_3d_vis: ", points_3d_vis.shape)
        # import pdb; pdb.set_trace()
        from visualizer import visualize_ndarray
        # visualize_ndarray(points_3d_vis, colors=colors, output_dir="pointcloud_visualizer", port=8000, max_points=100000, point_size=0.002)

        # import pdb; pdb.set_trace()
        print(f"current directory: {os.getcwd()}")
        ## 将当前目录设置为 /home/yunzhe/zzzzzworkspaceyy/robosuite_data/robosuite
        os.chdir("/data_new/yueyu/zz/robosuite_env/robosuite")
        print(f"current directory: {os.getcwd()}")
        # import pdb; pdb.set_trace()
        ###  visualize the points_3d using cv2
        ## dont use open3d
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # 假设你的点云是 Nx3 的 numpy 数组
        points = points_3d  # 示例数据

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ## save it as a png
        plt.savefig("points_3d.png")



        ## 480 * 640
        points_3d = points_3d.reshape(480, 640, 3)


        ## sample 20*20 points by grid
        
        ### (#keypoint,2)

        rows = np.arange(0, 640, 20)
        cols = np.arange(0, 480, 20)
        grid_x, grid_y = np.meshgrid(cols, rows)
        # print("grid_x: ", grid_x)
        # print("grid_y: ", grid_y)
        keypoints_3d = points_3d[grid_x.flatten(), grid_y.flatten(), :]
        keypoints_pixel_ids = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
        # import pdb; pdb.set_trace()
        ## flatten
        # keypoints_3d = keypoints_3d.reshape(-1, 3) 
        
        # filter with bounds
        bounds_min = [-0.35, -0.5, 0.3]
        bounds_max = [0.3, 0.5, 1.0]

        # mask = (keypoints_3d[:, 0] >= bounds_min[0]) & (keypoints_3d[:, 0] <= bounds_max[0]) & \
        #     (keypoints_3d[:, 1] >= bounds_min[1]) & (keypoints_3d[:, 1] <= bounds_max[1]) & \
        #     (keypoints_3d[:, 2] >= bounds_min[2]) & (keypoints_3d[:, 2] <= bounds_max[2])
        # keypoints_3d = keypoints_3d[mask]
        # keypoints_pixel_ids = keypoints_pixel_ids[mask]
        ## save the points_3d


        # import pdb; pdb.set_trace()

        ## add keypoints_3d to points_3d_vis also a ball
        for pixel_id in keypoints_pixel_ids:
            num_points = 20  # Number of points to create the sphere
            radius = 0.005  # Radius of the sphere
            
            # Generate random points on a sphere
            phi = np.random.uniform(0, 2*np.pi, num_points)
            theta = np.random.uniform(0, np.pi, num_points)
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            pos = points_3d[pixel_id[0], pixel_id[1], :].reshape(1, 3)
            ## geom_id
            geom_id = self._segmentation[pixel_id[0], pixel_id[1], 1]
            ## get the geom_name
            geom_name = self.env.sim.model.geom_id2name(geom_id)
            if "cube" in geom_name:
                print("cube found")
                print("geom_name: ", geom_name, "geom_id: ", geom_id)
                ## get geom_pos
                geom_pos = self.env.sim.data.geom_xpos[geom_id]
                dist = np.linalg.norm(geom_pos - pos)
                print("geom_pos: ", geom_pos)
                print("pos: ", pos)
                print("dist: ", dist)
                # import pdb; pdb.set_trace()
            else:
                continue
            ## get the geom_rgba
            # Add the sphere points to the point cloud
            sphere_points = np.stack([x, y, z], axis=1) + pos
            points_3d_vis = np.concatenate([points_3d_vis, sphere_points], axis=0)
            sphere_colors = np.tile(np.array([0, 0, 1]), (num_points, 1))  #  color for all sphere points
            colors = np.concatenate([colors, sphere_colors], axis=0)
        

        # visualize_ndarray(points_3d_vis, colors=colors, output_dir="pointcloud_visualizer", port=8000, max_points=100000, point_size=0.002)

        # import pdb; pdb.set_trace()
        print(f"current directory: {os.getcwd()}")
        ## 将当前目录设置为 /home/yunzhe/zzzzzworkspaceyy/robosuite_data/robosuite
        os.chdir("/data_new/yueyu/zz/robosuite_env/robosuite")
        print(f"current directory: {os.getcwd()}")


        self.register_keypoints(keypoints_pixel_ids, keypoints_3d, self._segmentation)
        self.render_with_keypoints()
        # import pdb; pdb.set_trace()
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
    
            self.states.append(state)

            info = {}
            info["actions"] = np.array(action)

            # (if applicable) store absolute actions
            step_info = ret[3]
            if "action_abs" in step_info.keys():
                info["actions_abs"] = np.array(step_info["action_abs"])

            self.action_infos.append(info)
            
            keypoints_pos = self.get_keypoint_positions()
            self.keypoints_pos.append(keypoints_pos)
            
            # Collect image and depth data from agentview camera
            images_step = {}
            depths_step = {}
            try:
                # Get image and depth for this camera
                img, depth = self.env.sim.render(width=640, height=480, camera_name="agentview", depth=True)
                ## vertical flip the image
                img = cv2.flip(img, 0)
                depth = cv2.flip(depth, 0)
                images_step["agentview"] = img
                depths_step["agentview"] = depth
            except Exception as e:
                print(f"Warning: Could not collect data from camera agentview: {e}")
            
            # Store the collected images and depths
            self.images.append(images_step)
            self.depths.append(depths_step)

        # check if the demonstration is successful
        if self.env._check_success():
            self.successful = True

        # flush collected data to disk if necessary
        # if self.t % self.flush_freq == 0:
            # self._flush()

        return ret

    def close(self):
        """
        Override close method in order to flush left over data
        """
        if self.has_interaction:
            self._flush()
        self.env.close()
