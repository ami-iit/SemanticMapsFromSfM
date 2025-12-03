# Code to test the intermediate steps and libraries needed for the depth estimation and voxel building process

import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from colmap_utils import read_model, qvec2rotmat
from depth_estimator import predict_depth, load_da_v2
# Utilities and visualization tools

def plot3D(
    pts: np.ndarray,
    point_size: float = 1.0,
):
    """
    Quick 3D scatter of a 3D pointcloud using matplotlib.
    """
    import matplotlib.pyplot as plt

    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(xs, ys, zs, s=point_size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Voxel Representation")

    # Equal aspect ratio
    max_range = np.array(
        [xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]
    ).max() / 2.0
    mid_x = (xs.max()+xs.min()) * 0.5
    mid_y = (ys.max()+ys.min()) * 0.5
    mid_z = (zs.max()+zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def get_intrinsics_matrix(cam):
    """
    cam: COLMAP Camera
    Handles PINHOLE and SIMPLE_PINHOLE without distortion.
    Returns:
        K: (3,3) torch.float32
    """
    params = cam.params
    if cam.model in ["PINHOLE"]:
        fx, fy, cx, cy = params[:4]
    elif cam.model in ["SIMPLE_PINHOLE"]:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise NotImplementedError(f"Camera model {cam.model} not handled here; "
                                  "use undistorted images or extend this.")
    K = torch.tensor([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    return K


def get_world_from_cam(img):
    """
    img: COLMAP Image (qvec, tvec)
    Returns:
        R_wc: (3,3) rotation from camera -> world
        t_wc: (3,)   translation of camera origin in world
    """
    R_wc = torch.from_numpy(qvec2rotmat(img.qvec)).float().T  # R_wc = R_cw^T
    t_cw = torch.from_numpy(img.tvec).float()                 # t_cw = t in X_c = R X_w + t
    C_w  = - R_wc @ t_cw                                      # camera center
    return R_wc, C_w


if __name__ == "__main__":

    # loading one image and corresponding data for testing purposes.

    root_dir = os.getcwd() # '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/'
    colmap_model_path = root_dir + '/Data/colmap_model'
    image_path = root_dir + '/Data/images/'

    image_name = 'PXL_20250930_080549279.jpg'

    # extract the COLMAP information (camera, images and pointcloud)

    cameras, images, points3D = read_model(colmap_model_path)

    img_id = [i for i in images if images[i].name == image_name]

    img = images[img_id[0]]
    cam = cameras[img.camera_id]

    # predict the depth

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = root_dir + "/Data/images" 
    output_dir = root_dir + "/Data/depth_maps"
    
    visualize = False

    processor, model = load_da_v2()
    model = model.to(device)
    model.eval()

    img_rgb = Image.open(image_path+ image_name)
    depth = predict_depth(img_rgb, processor, model)
    plt.imshow(depth.squeeze().cpu().numpy(), cmap='plasma')
    plt.show()
 
 
    # Reproject the points w.r.t. the camera
    R_wc, C_w = get_world_from_cam(img)
    R_wc = R_wc.to(device)
    C_w  = C_w.to(device)

    K = get_intrinsics_matrix(cam).to(device)
    K_inv = torch.inverse(K) 

    ## Build a grid over the image
    H, W = depth[0].shape
    # pixel grid
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")   # (W,H) if indexing="xy"
    uu = uu.T  # to (H,W)
    vv = vv.T

    # get pixel coordinates for valid depths
    u_valid = uu.float()
    v_valid = vv.float()
    d_valid = depth.float()

    # backproject to camera frame
    ones = torch.ones_like(u_valid)
    pix = torch.stack([u_valid, v_valid, ones], dim=-1)  # (N,3)
    rays_c = pix.reshape(W,H,1,3) @ K_inv.T # (K_inv @ pix.T).T                           # (N,3) directions
    X_c = rays_c.T.squeeze() * d_valid 
    # X_w = (R_wc @ X_c.T.unsqueeze(-1)).T + C_w.reshape(1,1,1,-1) # (R_wc @ X_c.T).T + C_w.unsqueeze(0)
    plot3D(X_c)

    print('pippo')    
    
dd