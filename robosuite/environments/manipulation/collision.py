from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects import BottleObject
from robosuite.models.objects import CanObject
from robosuite.models.objects import MilkObject
from robosuite.models.objects import BreadObject
from robosuite.models.objects import CerealObject
from robosuite.models.objects import SquareNutObject
from robosuite.models.objects import RoundNutObject
from robosuite.models.objects import PlateWithHoleObject
from robosuite.models.objects import DoorObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from robosuite.utils.camera_utils import transform_from_pixels_to_world, get_camera_extrinsic_matrix
import open3d as o3d


class Collision(ManipulationEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

        track_object (str or None): Name of the object to track. If None, no tracking is performed.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        camera_depths=True,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
        track_object=None,  # 要跟踪的物体名称
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # 设置要跟踪的物体
        self.track_object = track_object
        self.trajectory_history = []  # 存储轨迹历史
        self.max_history_length = 10  # 最大历史长度

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        # self.obje
        # object placement initializer
        self.placement_initializer = placement_initializer
        print("placement_initializer",self.placement_initializer)
        # print(" right now in collision\n")
        # import pdb; pdb.set_trace()
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )
        # print(self._observables, end=" _observables\n")
        # import pdb; pdb.set_trace()

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0
        return reward
        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.cube.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        ### output table pos

        print(mujoco_arena, end=" table_pos\n")
        # import pdb; pdb.set_trace()

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])


        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="darkwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # 定义5种颜色的立方体
        self.red_cube = BoxObject(
            name="red_cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        self.blue_cube = BoxObject(
            name="blue_cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0, 0, 1, 1],
            material=bluewood,
        )
        self.green_cube = BoxObject(
            name="green_cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )
        self.light_cube = BoxObject(
            name="light_cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[1, 1, 0, 1],
            material=lightwood,
        )
        self.dark_cube = BoxObject(
            name="dark_cube",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.022, 0.022, 0.022],
            rgba=[0.5, 0, 0.5, 1],
            material=darkwood,
        )
        # import xml.etree.ElementTree as ET

        # print(ET.tostring(self.red_cube.get_obj(), encoding='utf-8').decode('utf-8'))
        # import pdb; pdb.set_trace()
        all_objects = [self.red_cube, self.blue_cube, self.green_cube, self.light_cube, self.dark_cube]

        ## 随机选择2-5个不同颜色的立方体
        num_objects = np.random.randint(2, 6)
        if not hasattr(self, 'objects'):
            self.objects = np.random.choice(all_objects, num_objects, replace=False)
            self.objects_name = [obj.name for obj in self.objects]

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.objects)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-0.15, 0.15],
                y_range=[-0.15, 0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        # self.cube_body_id = self.sim.model.body_name2id(self.cube.root_body)
        self.object_body_ids = [self.sim.model.body_name2id(obj.root_body) for obj in self.objects]

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        #### TODO: add object information
        ## ADD OBJECT INFORMATION for each object


        # 定义可观察对象的模态
        modality = "object"

        # 为每个对象创建传感器函数
        sensors = []
        names = []

        for i, obj in enumerate(self.objects):
            # 对象位置传感器
            @sensor(modality=modality)
            def obj_pos(obs_cache, obj_id=i):
                return np.array(self.sim.data.body_xpos[self.object_body_ids[obj_id]])

            # 对象旋转四元数传感器
            @sensor(modality=modality)
            def obj_quat(obs_cache, obj_id=i):
                return convert_quat(np.array(self.sim.data.body_xquat[self.object_body_ids[obj_id]]), to="xyzw")

            # # 对象速度传感器
            # @sensor(modality=modality)
            # def obj_vel(obs_cache, obj_id=i):
            #     return np.array(self.sim.data.get_body_xvelp)

            # 设置传感器名称
            obj_name = self.objects[i].name
            obj_pos.__name__ = f"{obj_name}_pos"
            obj_quat.__name__ = f"{obj_name}_quat"
            # obj_vel.__name__ = f"{obj_name}_vel"

            # 添加到传感器列表
            sensors.extend([obj_pos, obj_quat])
            names.extend([f"{obj_name}_pos", f"{obj_name}_quat"])

        # 为机器人手臂添加传感器
        if self.robots:
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            # 为每个对象和机器人手臂添加抓取器到对象位置的传感器
            for i, obj in enumerate(self.objects):
                obj_name = self.objects[i].name
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes):
                    sensors.append(
                        self._get_obj_eef_sensor(full_pf, f"{obj_name}_pos", f"{arm_pf}gripper_to_{obj_name}_pos", modality)
                    )
                    names.append(f"{arm_pf}gripper_to_{obj_name}_pos")

        # 创建可观察对象
        for name, s in zip(names, sensors):
            # print(name,s, end=" name,s\n")
            # import pdb; pdb.set_trace()
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        # low-level object information
        # if self.use_object_obs:
        #     # define observables modality
        #     modality = "object"

        #     # cube-related observables
        #     @sensor(modality=modality)
        #     def cube_pos(obs_cache):
        #         return np.array(self.sim.data.body_xpos[self.cube_body_id])

        #     @sensor(modality=modality)
        #     def cube_quat(obs_cache):
        #         return convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")

        #     sensors = [cube_pos, cube_quat]

        #     arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        #     full_prefixes = self._get_arm_prefixes(self.robots[0])

        #     # gripper to cube position sensor; one for each arm
        #     sensors += [
        #         self._get_obj_eef_sensor(full_pf, "cube_pos", f"{arm_pf}gripper_to_cube_pos", modality)
        #         for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        #     ]
        #     names = [s.__name__ for s in sensors]

        #     # Create observables
        #     for name, s in zip(names, sensors):
        #         observables[name] = Observable(
        #             name=name,
        #             sensor=s,
        #             sampling_rate=self.control_freq,
        #         )
        # import pdb; pdb.set_trace()
        # print(observables, end=" observables\n")
        return observables

    @staticmethod
    def _pcd_crop(pcd, bounds):
        mask = np.all(pcd[:, :3] > bounds[:3], axis=1) & np.all(pcd[:, :3] < bounds[3:], axis=1)
        return pcd[mask]

    # def process_obs_dict(self, obs_dict):
    #     keys = obs_dict.keys()
    #     # print("keys:", keys)
    #     processed_obs = {
    #         "agent_pos": np.zeros(7),
    #         "point_cloud": np.zeros((self.n_points, 6)),
    #     }

    #     # process robot state
    #     robot_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    #     if all([k in keys for k in robot_keys]):
    #         pos = obs_dict["robot0_eef_pos"]
    #         rot = self._convert_state_quat_to_action_rotvec(obs_dict["robot0_eef_quat"])
    #         gripper = obs_dict["robot0_gripper_qpos"][0] - obs_dict["robot0_gripper_qpos"][1]
    #         robot_state = np.concatenate([pos, rot, [gripper]])
    #         processed_obs["agent_pos"] = robot_state

    #     # process point cloud
    #     pcd_list = []
    #     for cam_name in self.cam_names:
    #         rgb_name = cam_name + "_image"
    #         depth_name = cam_name + "_depth"
    #         if all([k in keys for k in [rgb_name, depth_name]]):
    #             # print("cam_name:", cam_name)
    #             rgb = obs_dict[rgb_name]
    #             depth = obs_dict[depth_name]
    #             pcd = self.get_pcd_from_rgbd(cam_name, rgb, depth)
    #             pcd_list.append(pcd)
    #         else:
    #             break
    #     if len(pcd_list) > 0:
    #         pcd = np.concatenate(pcd_list, axis=0)
    #         if USE_CROP:
    #             pcd = self._pcd_crop(pcd, TASK_BOUDNS[self.task_name])
    #         pcd = point_cloud_sampling(pcd, self.n_points)
    #         processed_obs["point_cloud"] = pcd

    #         # import pcd_visualizer
    #         # pcd_visualizer.visualize_pointcloud(pcd)

    #     return processed_obs

    def get_pcd_from_rgbd(self, cam_name, rgb, depth):
        def verticalFlip(img):
            return np.flip(img, axis=0)

        def get_o3d_cammat():
            cam_mat = self.env.get_camera_intrinsic_matrix(cam_name, self.cam_width, self.cam_height)
            cx = cam_mat[0,2]
            fx = cam_mat[0,0]
            cy = cam_mat[1,2]
            fy = cam_mat[1,1]
            return o3d.camera.PinholeCameraIntrinsic(self.cam_width, self.cam_height, fx, fy, cx, cy)

        rgb = verticalFlip(rgb)
        depth = self.env.get_real_depth_map(verticalFlip(depth))
        o3d_cammat = get_o3d_cammat()
        o3d_depth = o3d.geometry.Image(depth)
        o3d_pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_cammat)
        world_T_cam = self.env.get_camera_extrinsic_matrix(cam_name)
        o3d_pcd.transform(world_T_cam)
        points = np.asarray(o3d_pcd.points)
        colors = rgb.reshape(-1, 3)
        pcd = np.concatenate([points, colors], axis=1)
        return pcd



    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # 清空轨迹历史
        self.trajectory_history = []

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                print(obj,obj.joints, end=" obj\n")
                print(obj_pos,obj_quat, end=" obj_pos,obj_quat\n")
                # import pdb; pdb.set_trace()
                if len(obj.joints) > 0:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

            # 在物体放置完成后，设置轨迹点的位置
            # 获取相机内参矩阵
            # camera_name = "frontview"  # 使用前视图相机

            # # 获取相机外参矩阵
            # camera_extrinsic = get_camera_extrinsic_matrix(self.sim, camera_name)

            # # 设置一些示例点（这里需要根据您的需求修改）
            # pixel_points = np.array([
            #     [100, 100],  # 示例像素坐标
            #     [200, 200],
            #     [300, 300],
            #     # ... 添加更多点
            # ])

            # # 将像素坐标转换为世界坐标
            # world_points = transform_from_pixels_to_world(
            #     self.sim,
            #     camera_id,
            #     pixel_points,
            #     camera_extrinsic
            # )

            # 更新轨迹点的位置
            # self.update_trajectory_points(world_points)

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube)

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        # cube_height = self.sim.data.body_xpos[self.cube_body_id][2]
        # table_height = self.model.mujoco_arena.table_offset[2]

        # cube is higher than the table top above a margin
        # return cube_height > table_height + 0.04
        return False

    def update_trajectory_points(self, positions):
        """
        更新轨迹跟踪点的位置

        Args:
            positions (list): 包含每个跟踪点新位置的列表，每个位置是一个3D坐标 [x, y, z]
        """
        if len(positions) > len(self.trajectory_sites):
            print(f"Warning: 提供的位置数量({len(positions)})超过了跟踪点数量({len(self.trajectory_sites)})")
            positions = positions[:len(self.trajectory_sites)]

        for i, (site, pos) in enumerate(zip(self.trajectory_sites, positions)):
            site_id = self.sim.model.site_name2id(site.name)
            self.sim.model.site_pos[site_id] = pos

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        """
        # 执行原始step
        obs, reward, done, info = super().step(action)

        # 如果设置了跟踪物体，更新轨迹点
        if self.track_object is not None:
            # 获取物体当前位置
            obj_pos = self.sim.data.get_body_xpos(self.track_object)

            # 添加到历史记录
            self.trajectory_history.append(obj_pos)
            if len(self.trajectory_history) > self.max_history_length:
                self.trajectory_history.pop(0)

            # 更新轨迹点位置
            self.update_trajectory_points(self.trajectory_history)

        return obs, reward, done, info










