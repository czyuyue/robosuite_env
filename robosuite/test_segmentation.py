
import os
robotsuite_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.append(robotsuite_dir)
import robosuite as suite
from robosuite.wrappers import DataCollectionWrapper
import numpy as np
import cv2

# 创建环境
env = suite.make(
    env_name="Stack",  # 使用Lift环境
    robots="Panda",   # 使用Panda机器人
    has_renderer=False,  # 关闭实时渲染
    has_offscreen_renderer=True,  # 启用离屏渲染
    render_camera="frontview",
    camera_names=["frontview"],
    renderer="mujoco",
    control_freq=20,
    ignore_done=True,
    camera_segmentations=["agentview"],
)

# 包装环境以获取分割信息

# 重置环境
obs = env.reset()

rgb = env.sim.render(width=640, height=480, camera_name="agentview")
cv2.imwrite("agentview_test.png", rgb)


# 获取分割图
segmentation, _ = env.sim.render(width=640, height=480, camera_name="agentview", segmentation=True,depth=True)

# 保存分割图
cv2.imwrite("segmentation_mask1_test.png", segmentation[:,:,0]*20)  # 第一个通道
cv2.imwrite("segmentation_mask2_test.png", segmentation[:,:,1]*20)  # 第二个通道

# 分析分割结果
unique_values_1 = np.unique(segmentation[:,:,0])
unique_values_2 = np.unique(segmentation[:,:,1])

print("Unique values in first channel:", unique_values_1)
print("Unique values in second channel:", unique_values_2)

# 为每个唯一值打印对应的物体名称
for body_id in unique_values_1:
    if body_id == 0:  # 跳过背景
        continue
    name = env.sim.model.body_id2name(body_id)
    print(f"Body ID {body_id}: {name}")

# 可视化分割结果
mask_vis = segmentation[:,:,1].copy()*20
for val in unique_values_2:
    if val == 0:  # 跳过背景
        continue
    region_pixels = np.where(segmentation[:,:,1] == val)
    y_middle = region_pixels[0][len(region_pixels[0])//2]
    x_middle = region_pixels[1][len(region_pixels[1])//2]
    cv2.putText(mask_vis, str(int(val)), (x_middle, y_middle), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

cv2.imwrite("segmentation_visualized_test.png", mask_vis)

env.close()
