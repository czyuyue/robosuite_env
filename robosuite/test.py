import numpy as np

# Load the .npz file
data = np.load('/home/yunzhe/zzzzzworkspaceyy/robosuite_data/robosuite/robosuite_data/robosuite/data/collision_data/ep_id0_1749404037/state_id0_1749404073.npz',allow_pickle=True)

# Print the structure of the .npz file
print("Keys in the .npz file:")
print(data.files)

keypoint_pos = data['keypoint_positions']
images = data['images']
print(images[0].keys(),end="keys images\n")
print(images[0]['frontview'].shape,end="shape images\n")
depths = data['depths']
print(depths[0].keys(),end="keys depths\n")
print(depths[0]['frontview'].shape,end="shape depths\n")

states = data['states']
print(images.shape,end="shape images\n")
print(depths.shape,end="shape depths\n")
print(states.shape , end="shape states\n")
print(keypoint_pos.shape,end="shape keypoint_pos\n")
print(keypoint_pos[0])


# Print the shape and data type of each array
# for key in data.files:
#     print(f"\nArray '{key}':")
#     print(f"Shape: {data[key].shape}")
#     print(f"Data type: {data[key].dtype}")
#     print(f"First few elements: {data[key][:5]}")  # Print first 5 elements as a sample
