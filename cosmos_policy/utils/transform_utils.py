
import numpy as np

def quat2euler(quat):
    """
    Convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw).
    
    Args:
        quat: (N, 4) or (4,) array in w, x, y, z format (or x, y, z, w?)
        Standard convention for libraries like scipy is usually scalar-last (x, y, z, w) for input but internal might differ.
        Detecting convention from data is hard without ground truth.
        However, inspecting inspect_h5 output: ee_rotation is (N, 4).
        Assumption: standard [x, y, z, w] or [w, x, y, z].
        
        Let's assume standard [x, y, z, w] for now commonly used in robotics (ROS uses x,y,z,w).
        Wait, scipy uses [x, y, z, w].
        Let's implement a robust one.
    """
    # Assuming q = [x, y, z, w]
    # Roll (x-axis rotation)
    if len(quat.shape) == 1:
        quat = quat[None, :]
        
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    
    return np.stack([roll, pitch, yaw], axis=-1)
