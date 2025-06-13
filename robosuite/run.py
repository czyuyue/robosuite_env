import os
robotsuite_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(robotsuite_dir)

import numpy as np
import robosuite as suite
import imageio
import cv2
from scipy.spatial.transform import Rotation as R
from robosuite.wrappers import DataCollectionWrapper

def direction_to_axis_angle(direction_xy):
    # 目标方向，假设在xy平面
    direction_xy = np.array(direction_xy)
    direction_xy = direction_xy / np.linalg.norm(direction_xy)

    # 将 z 轴方向旋转到目标方向对应的旋转矩阵
    current_dir = np.array([1.0, 0.0, 0.0])  # 假设gripper默认朝x轴
    target_dir = np.array([direction_xy[0], direction_xy[1], 0.0])

    rot = R.align_vectors([target_dir], [current_dir])[0]
    return rot.as_rotvec()  # 返回 axis-angle

def quaternion_distance(q1, q2):
    """计算两个四元数之间的角度距离"""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # 计算四元数点积的绝对值
    dot_product = abs(np.dot(q1, q2))
    # 确保dot_product在有效范围内
    dot_product = min(1.0, max(-1.0, dot_product))

    # 返回角度差异（弧度）
    return 2.0 * np.arccos(dot_product)
def calculate_table_orientation(direction_xy, format='xyzw'):
    """
    Calculate quaternion orientation for an end-effector to be:
    1. Parallel to the table (z-axis pointing down)
    2. Facing a specific direction in the xy-plane

    Parameters:
    -----------
    direction_xy : tuple or list or np.ndarray
        Direction vector in xy-plane (x, y)
    format : str, optional
        Output quaternion format, either 'wxyz' or 'xyzw'

    Returns:
    --------
    quaternion : np.ndarray
        Orientation as quaternion in specified format
    """
    # Normalize the xy direction vector
    direction = np.array([direction_xy[0], direction_xy[1], 0])
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    else:
        raise ValueError("Direction vector cannot be zero")

    # For a parallel-to-table orientation, we assume:
    # - The z-axis of the end-effector points down (towards table)
    # - The x-axis of the end-effector points in the specified xy direction
    # - The y-axis follows from the right-hand rule

    # Step 1: Define the axes of our end-effector's coordinate frame
    z_axis = np.array([0, 0, -1])  # Pointing down towards table
    x_axis = direction  # Pointing in the specified xy direction
    y_axis = np.cross(z_axis, x_axis)  # Using right-hand rule

    # Handle the special case when direction is close to vertical
    if np.linalg.norm(y_axis) < 1e-6:
        # Choose an arbitrary perpendicular direction
        y_axis = np.array([0, 1, 0])
        x_axis = np.cross(y_axis, z_axis)
        y_axis = np.cross(z_axis, x_axis)

    # Normalize axes
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Step 2: Create rotation matrix
    # The columns of the rotation matrix are the basis vectors of the new frame
    # expressed in the original frame
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    # Step 3: Convert rotation matrix to quaternion
    rotation = R.from_matrix(rotation_matrix)
    quat_xyzw = rotation.as_quat()  # scipy returns in xyzw format

    # Convert to desired format
    print(f"quat_xyzw: {quat_xyzw}")
    # import pdb; pdb.set_trace()

    if format.lower() == 'wxyz':
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    else:
        return quat_xyzw
def orientation_control(current_quat, target_quat, gain=2.0):
    """
    计算方向控制信号
    target_quat is in xyzw format
    current_quat is in xyzw format
    """
    # 使用scipy的Rotation来处理四元数
    current_rot = R.from_quat([current_quat[0], current_quat[1], current_quat[2], current_quat[3]])  # 注意：scipy使用的是[x,y,z,w]顺序
    # target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]])
    target_rot = R.from_quat([target_quat[0], target_quat[1], target_quat[2], target_quat[3]])
    # 计算从当前旋转到目标旋转的相对旋转
    rel_rot = target_rot * current_rot.inv()

    # 从相对旋转中提取轴角表示
    rotvec = rel_rot.as_rotvec()

    # 应用增益 - 这会同时缩放角度大小，保持旋转轴方向不变
    return gain * rotvec

class ScriptedPolicy:
    def __init__(self, env):
        self.env = env
        self.target_pos = None  # 推的目标方向或位置
        self.state = "lift"  # 状态机：approach / push / done
        self.gain = 5
        self.directed = False
        self.frames = []
        self.keypoint_frames = []  # 添加关键点视频帧收集
        self.step_count = 0
    def set_object_and_target_pos(self, object_name, target_object_name, target_pos):
        self.object_name = object_name
        self.target_object_name = target_object_name
        self.object_pos = self._get_object_pos(obs)
        self.target_pos = target_pos
        self.target_pos[2] = 0.82
        self.state = "lift"
        self.frames = []
        self.keypoint_frames = []  # 清空关键点视频帧
        ### offset is opposite direction of target_pos
        def normalize(vec):
            return vec / np.linalg.norm(vec)


        norm_dir =  normalize (self.target_pos - self.object_pos)
        ## check if nan
        if np.isnan(norm_dir).any():
            import pdb; pdb.set_trace()
        self.offset = -norm_dir * 0.05

        self.target_pos = self.target_pos + norm_dir * 0.05

        ### calc the end effector direction make the endeffector prependicular to normdir
        self.end_effector_direction = np.array([norm_dir[0], norm_dir[1]])

        self.target_quat = calculate_table_orientation(self.end_effector_direction)

        self.directed = False


    def get_action(self, obs):

        ee_pos = self._get_end_effector_pos(obs)
        obj_pos = self._get_object_pos(obs)

        ## get current end effector direction from obs
        # current_dir = obs["robot0_eef_quat"]
        # print(f"current_dir: {current_dir}")
        # import pdb; pdb.set_trace()

        obj_pos[2] = 0.82
        print(f"current state: {self.state}")
        if self.state == "lift":
            # 首先检查是否需要抬高机械臂
            if ee_pos[2] < 1.1-0.05:
                target_pos = ee_pos.copy()
                target_pos[2] = 1.1
                action = self._move_to(target_pos)

                print(f"lifting ee to height {target_pos}")
            else:
                self.state = "approach"
        if self.state == "approach":
            # 然后检查是否需要水平移动到目标位置上方
            if np.linalg.norm(ee_pos[:2] - (obj_pos[:2] + self.offset[:2])) > 0.02:
                target_pos = obj_pos.copy() + self.offset
                target_pos[2] = 1.1  # 保持高度
                action = self._move_to(target_pos)
                print(f"moving horizontally to {target_pos} current ee_pos: {ee_pos}  current object_pos: {obj_pos+ self.offset} current delta: {np.linalg.norm(ee_pos[:2] - (obj_pos[:2] + self.offset[:2]))}")

            else:
                self.state = "lower"
            # 最后下降到目标高度
        if self.state == "lower":
            if ee_pos[2] > 0.82+0.005:
                target_pos = obj_pos.copy() + self.offset
                target_pos[2] = 0.82  # 下降到目标高度
                action = self._move_to(target_pos)
                print(f"lowering to {target_pos} current ee_pos: {ee_pos}")
            else:
                self.state = "push"
        if self.state == "push":
            target_pos = self.target_pos.copy()
            target_pos[2] = 0.81  # 保持高度
            action = self._move_to(target_pos)
            print(f"PUSH move ee to {target_pos}")
            if np.linalg.norm(ee_pos[:2] - self.target_pos[:2]) < 0.01:
                self.state = "done"
                print(f"done {self.object_name} to {self.target_pos} !!!!!!!!!!")
                print(f"ee_pos: {ee_pos}, target_pos: {self.target_pos}")

        return action

    def _get_end_effector_pos(self, obs):
        return obs["robot0_eef_pos"]  # 根据你的obs定义提取末端执行器位置
    def _get_end_effector_ori(self, obs):
        return obs["robot0_eef_quat"]  # 根据你的obs定义提取末端执行器方向

    def _get_object_pos(self, obs):
        return obs[self.object_name + "_pos"]  # 提取物体的位置

    def _move_to(self, target_pos):
        # 这里可以用差值或控制器计算出动作
        direction = target_pos - self._get_end_effector_pos(obs)

        quat_delta = np.array([0, 0, 0])
        if not self.directed:
            quat_delta = self.direction_to_axis_angle
            self.directed = True
        action = np.concatenate([direction * self.gain, quat_delta])
        return action

    def move_to_pose(self, env, target_pos, target_quat, gripper_action=0,
                    p_gain=10.0, d_gain=5.0, max_steps=100, pos_threshold=0.005,
                    ori_threshold=0.01, orient_only=False, pos_only=False):
        """
        使用简单的直接位置控制将机器人移动到目标位置和姿态

        参数:
            env: robosuite环境实例
            target_pos: 目标位置，3维numpy数组 [x, y, z]
            target_quat: 目标四元数，4维numpy数组 [x, y, z, w]
            gripper_action: 夹爪动作 (0: 关闭, 1: 打开)
            p_gain: 位置比例增益
            d_gain: 位置微分增益
            max_steps: 最大尝试步数
            pos_threshold: 位置误差阈值
            ori_threshold: 方向误差阈值

        返回:
            bool: 是否成功到达目标位置
        """
        target_pos = np.array(target_pos)
        target_quat = np.array(target_quat)
        print(f"target_quat: {target_quat}")
        # 归一化目标四元数
        target_quat = target_quat / np.linalg.norm(target_quat)
        print(f"target_quat after norm: {target_quat}")
        # import pdb; pdb.set_trace()
        last_error = None

        for step in range(max_steps):
            # 获取当前末端执行器状态
            obs = env._get_observations()
            current_pos = self._get_end_effector_pos(obs)
            current_quat = self._get_end_effector_ori(obs)


            # 计算位置误差
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            # print(f"current_pos: {current_pos}, current_quat: {current_quat}")
            # print(f"target_pos: {target_pos}, target_quat: {target_quat}")
            # print(f"last_error: {last_error}, pos_error: {pos_error}")
            # import pdb; pdb.set_trace()
            # 计算方向误差 (四元数差)
            quat_error = quaternion_distance(current_quat, target_quat)

            # 检查是否达到目标
            if (pos_error_norm < pos_threshold or orient_only) and (quat_error < ori_threshold or pos_only):
                print("已到达目标位置和姿态!")
                return True

            # 简单的直接位置控制
            action = np.zeros(6)
            if not orient_only:
                action[:3] = pos_error * 5.0  # 直接使用位置误差，乘以一个简单的增益
            if not pos_only:
                action[3:6] = orientation_control(current_quat, target_quat)
            # 执行动作
            obs, reward, done, info = env.step(action)
            frame_front = obs["frontview_image"]
            frame_agent = obs["agentview_image"]
            frame_bird = obs["birdview_image"]
            ### add text to the frame
            frame = np.concatenate([frame_front, frame_agent, frame_bird], axis=1)
            frame = cv2.putText(frame, f"push {self.object_name} to {self.target_object_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.frames.append(frame)
            
            # 收集关键点可视化帧
            self.step_count += 1
            check = self.step_count % 150 == 0
            keypoint_frame = env.render_with_keypoints(check)
            # 添加文本到关键点帧
            keypoint_frame = cv2.putText(keypoint_frame, f"push {self.object_name} to {self.target_object_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.keypoint_frames.append(keypoint_frame)
            

        print("未能在最大步数内到达目标位置和姿态")
        return False
    def execute_policy(self, env, gripper_action=0):

        ### then rotate to the target direction
        obs = env._get_observations()
        ee_pos = self._get_end_effector_pos(obs)
        target_pos = np.array([ee_pos[0], ee_pos[1], 1.1])
        target_quat = self.target_quat
        self.move_to_pose(env, target_pos, target_quat, orient_only=True)
        print("\033[92mrotate to the target direction\033[0m")
        # import pdb; pdb.set_trace()
        ### first move up to 1.1m
        obs = env._get_observations()
        ee_pos = self._get_end_effector_pos(obs)
        target_pos = np.array([ee_pos[0], ee_pos[1], 1.1])
        target_quat = self._get_end_effector_ori(obs)
        self.move_to_pose(env, target_pos, target_quat, pos_only=True)
        print("\033[92mmove up to 1.1m\033[0m")
        # import pdb; pdb.set_trace()
        # return

        # return
        ### then move to the target position
        obs = env._get_observations()
        obj_pos = self._get_object_pos(obs)
        target_pos = obj_pos.copy() + self.offset
        target_pos[2] = 1.1
        self.move_to_pose(env, target_pos, target_quat, pos_only=True)
        print("\033[92mmove above the target position\033[0m")
        # import pdb; pdb.set_trace()
        ### then lower down
        obs = env._get_observations()
        obj_pos = self._get_object_pos(obs)
        target_pos = obj_pos.copy() + self.offset
        target_pos[2] = 0.809
        self.move_to_pose(env, target_pos, target_quat, pos_only=True)
        print("\033[92mmove to the target position lower down\033[0m")
        # import pdb; pdb.set_trace()


        ### then push the object
        target_pos = self.target_pos
        target_pos[2] = 0.809
        self.move_to_pose(env, target_pos, target_quat, pos_only=True)
        print("\033[92mpush the object\033[0m")
        # import pdb; pdb.set_trace()

        ### move up
        obs = env._get_observations()
        ee_pos = self._get_end_effector_pos(obs)
        target_pos = np.array([ee_pos[0], ee_pos[1], 1.1])
        target_quat = self._get_end_effector_ori(obs)
        self.move_to_pose(env, target_pos, target_quat, pos_only=True)
        print("\033[92mmove up\033[0m")


        return NotImplementedError
    def save_video(self, video_path):
        print(f"current directory: {os.getcwd()}")

        print(f"saving video to {video_path}")
        ## save /home/yunzhe/zzzzzworkspaceyy/robosuite_data/robosuite/robosuite_videos/
        # import pdb; pdb.set_trace()
        imageio.mimsave(video_path, self.frames, fps=20)
        
        # 保存关键点可视化视频
        if len(self.keypoint_frames) > 0:
            keypoint_video_path = video_path.replace('.mp4', '_keypoints.mp4')
            print(f"saving keypoint video to {keypoint_video_path}")
            imageio.mimsave(keypoint_video_path, self.keypoint_frames, fps=20)
        else:
            print("No keypoint frames to save")
        
# 配置控制器
env = suite.make(
    env_name="Collision",  # 尝试简单的抓取任务
    robots="Panda",   # 使用Panda机器人
    gripper_types="WipingGripper",
    has_renderer=False,  # 关闭实时渲染
    has_offscreen_renderer=True,  # 启用离屏渲染以录制视频
    render_camera="frontview",
    #'frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
    camera_names=["frontview","agentview","birdview"],
    renderer="mujoco",
    control_freq=20,
    ignore_done=True,
    camera_segmentations=["frontview","agentview","birdview"],
)

env = DataCollectionWrapper(env, "robosuite_data/robosuite/data/collision_data")

### 输出当前目录
print(f"current directory: {os.getcwd()}")
import pdb; pdb.set_trace()


# 创建保存视频的目录
video_dir = "robosuite_videos"
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, "robosuite_test.mp4")
# env.unset_ep_meta()
# 重置环境
# import pdb; pdb.set_trace()
obs = env.reset()
print(obs.keys(), end=" obs.keys\n")
obs = env._get_observations()
print(obs.keys(), end=" obs.keys\n")
# import pdb; pdb.set_trace()

# print("观察空间形状:", obs.shape)
# print("动作空间形状:", env.action_spec[0].shape)

# 收集帧用于视频
frames = []
# 添加初始帧
# frames.append(env.render(mode="rgb_array", camera_name="frontview"))

# 执行随机动作的简单循环

### push the first object to the second object
policy = ScriptedPolicy(env)
print(obs.keys(), end=" obs.keys\n")
# import pdb; pdb.set_trace()

first_object = env.objects_name[0]
second_object = env.objects_name[1]
## save the initial picture
agent_view_image = obs["agentview_image"]
imageio.imwrite("agent_view_image.png", agent_view_image)
# print(f" push {first_object} to {second_object} !!!!!!!!!!")
# import pdb; pdb.set_trace()


first_object_pos = obs[first_object + "_pos"]
second_object_pos = obs[second_object + "_pos"]

print(obs,end=" obs\n")

print(f"first_object_pos: {first_object_pos}, second_object_pos: {second_object_pos}")

policy.set_object_and_target_pos(first_object, second_object, second_object_pos)

policy.execute_policy(env)

policy.save_video(video_path)
env.close()
exit(0)

for i in range(2):
    ##  choose 2 different objects indices
    perm = np.random.permutation(len(env.objects_name))
    first_object = env.objects_name[perm[0]]
    second_object = env.objects_name[perm[1]]
    first_object_pos = obs[first_object + "_pos"]
    second_object_pos = obs[second_object + "_pos"]

    print(f" push {first_object} to {second_object} !!!!!!!!!!")
    import pdb; pdb.set_trace()
    policy.set_object_and_target_pos(first_object, second_object_pos)
    step = 0
    while policy.state != "done" and step < 200:
        action = policy.get_action(obs)
        if policy.state == "done":
            break
        obs, reward, done, info = env.step(action)
        frame_front = obs["frontview_image"]
        frame_agent = obs["agentview_image"]
        frame_bird = obs["birdview_image"]

        frame = np.concatenate([frame_front, frame_agent, frame_bird], axis=1)

        ### add text to the frame : push first_object to second_object
        frame = cv2.putText(frame, f"push {first_object} to {second_object}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        frames.append(frame)
        step += 1
    print(f" push {first_object} to {second_object} !!!!!!!!!!")






# for i in range(200):  # 减少步数以加快视频生成
#     action = np.random.uniform(low=-1, high=1, size=env.action_dim)
#     obs, reward, done, info = env.step(action)

#     # print(obs.keys(), end=" obs.keys\n")
#     # import pdb; pdb.set_trace()
#     # 获取当前帧并添加到帧列表

#     frame = obs["birdview_image"]
#     # frame = env.render()
#     # print(frame.shape, end=" frame.shape\n")
#     frames.append(frame)

#     if done:
#         obs = env.reset()
#         print(f"回合 {i} 结束，重置环境")

# 保存视频
print(f"正在保存视频到 {video_path}...")
imageio.mimsave(video_path, frames, fps=20)

# 关闭环境
env.close()
print(f"测试完成! 视频已保存到 {video_path}")

