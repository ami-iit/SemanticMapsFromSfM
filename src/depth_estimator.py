# This files takes an SfM model obtained from COLMAP, and uses monocular depht estimation (based on the DepthAnything model) to predict a depth map for each image. 
# Leveraging on the known camera intrinsics and keypont/3D correspondences from COLMAP, the depth map is then re-scaled to be consistent with a common reference frame

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from colmap_utils import read_model, qvec2rotmat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle


# Code to generate the depth map

def load_da_v2(model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    """
    Returns (processor, model) ready for inference on a given device.
    """
    processor = image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)

    model.eval()
    return processor, model

@torch.no_grad()
def predict_depth(images_pil, processor, model):
    """
    images_pil: list[PIL.Image], length N
    returns: depth_rel (N,1,H,W) float32 on `device`
    Depth maps are *relative* (arbitrary scale).
    """
    
    h_i, w_i = images_pil.size[1], images_pil.size[0]
    
    inputs = processor(images=images_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # outputs.predicted_depth: (N, H, W) relative depth
    depth = outputs.predicted_depth.unsqueeze(1)  # (N,1,H,W)

    depth_hw = resize_depth_to_image(depth, h_i, w_i)

    return depth_hw


def check_depth_estimator(img_path="Data/images/PXL_20250930_080552967.jpg", save = False):
    p, m = load_da_v2()
    img = Image.open(img_path)
    depth = predict_depth(img, p, m)

    # Visualize the original image and the depth map side by side
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1,2,2)
    plt.imshow(depth.squeeze().cpu().numpy(), cmap='plasma')
    plt.axis('off')
    plt.title('Depth Map')
    if save:
        plt.savefig('depth_estimation_result_{}_.png'.format(img_path.split('/')[-1].split('.')[0]))
    else:
        plt.show()
    plt.close()

# Code to load the COLMAP model and re-scale the depth maps accordingly


def get_image_rt(img):
    """
    img: colmap_utils.Image (qvec, tvec)
    Returns:
        R: (3,3) torch.float32
        t: (3,)   torch.float32
    COLMAP convention: X_c = R * X_w + t
    """
    R = qvec2rotmat(img.qvec)  # (3,3) numpy
    t = img.tvec               # (3,)  numpy
    R = torch.from_numpy(R).float()
    t = torch.from_numpy(t).float()
    return R, t


def get_2d3d_correspondences_for_image(img, points3D):
    """
    img: colmap_utils.Image
    points3D: dict[point3D_id -> Point3D]

    Returns:
        xys:  (K,2) float32   pixel coords (u,v)
        Xws:  (K,3) float32   world coords
    """
    xys = img.xys                 # (N,2) numpy
    pids = img.point3D_ids        # (N,) numpy

    # Keep only valid 3D points (pid != -1 and exists in dict)
    mask_valid = (pids != -1)
    pids_valid = pids[mask_valid]
    xys_valid = xys[mask_valid]

    Xws_list = []
    xys_list = []
    for xy, pid in zip(xys_valid, pids_valid):
        if pid in points3D:
            Xws_list.append(points3D[pid].xyz)
            xys_list.append(xy)

    if len(Xws_list) == 0:
        return None, None

    xys_arr = np.stack(xys_list, axis=0).astype(np.float32)  # (K,2)
    Xws_arr = np.stack(Xws_list, axis=0).astype(np.float32)  # (K,3)
    return xys_arr, Xws_arr


def resize_depth_to_image(depth_rel_i, target_h, target_w):
    """
    depth_rel_i: (1,1,Hd,Wd) or (1,Hd,Wd)
    returns: (1,H,W)
    """
    if depth_rel_i.ndim == 3:
        depth_rel_i = depth_rel_i.unsqueeze(0)  # (1,1,Hd,Wd)
    depth_resized = F.interpolate(
        depth_rel_i,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=True
    )
    return depth_resized[0]



def compute_image_depth_scale(depth_rel_i, img, points3D, H, W, device="cuda", min_corr=10):
    """
    depth_rel_i: (1,H,W) torch.float32 relative depth map for this image,
                 same size as the original RGB.
    img:        colmap_utils.Image
    points3D:   dict point3D_id -> Point3D
    device:     "cuda" or "cpu"
    min_corr:   minimum # of valid correspondences to trust the scale

    Returns:
        scale: float or None if not enough valid correspondences
    """
    # 1) Get 2D-3D correspondences
    xys, Xws = get_2d3d_correspondences_for_image(img, points3D)
    if xys is None:
        return None

    depth_rel_i = depth_rel_i.to(device)  # (1,H,W)
    depth_resized = resize_depth_to_image(depth_rel_i, H, W)[0]
    
    # 2) Compute true depths from geometry: X_c = R X_w + t, depth = z
    R, t = get_image_rt(img)
    R = R.to(device)
    t = t.to(device)

    Xws_t = torch.from_numpy(Xws).to(device)  # (K,3)
    Xcs = (R @ Xws_t.T).T + t                # (K,3)
    d_true = Xcs[:, 2]                       # (K,)
    valid_geom = d_true > 1e-6               # in front of camera

    if valid_geom.sum().item() < min_corr:
        return None

    d_true = d_true[valid_geom]              # (K_valid,)
    xys_t = torch.from_numpy(xys).to(device)[valid_geom]  # (K_valid,2)

    # 3) Sample predicted depth at these pixel coordinates
    # xys are (u,v) in pixel coordinates [0,W) x [0,H)
    u = xys_t[:, 0].clamp(0, W - 1)
    v = xys_t[:, 1].clamp(0, H - 1)

    u_idx = u.round().long()
    v_idx = v.round().long()

    # depth_rel_i: (1,H,W) -> (H,W)
    depth_map = depth_rel_i.squeeze()               # (H,W)
    d_pred = depth_map[v_idx, u_idx]         # (K_valid,)

    # 4) Filter valid depth pairs
    mask_valid = (d_pred > 1e-6) & torch.isfinite(d_pred) & torch.isfinite(d_true)
    if mask_valid.sum().item() < min_corr:
        return None

    d_true = d_true[mask_valid]
    d_pred = d_pred[mask_valid]

    # 5) Robust scale estimate: median(d_true / d_pred)
    ratios = d_true / d_pred
    scale = torch.median(ratios).item()
    return scale


def main():

    root_dir = "/home/mtoso/Documents/Code/SemanticMapsFromSfM/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colmap_model_path = root_dir + "/Data/colmap_model"
    images_dir = root_dir + "Data/images" 
    output_dir = root_dir + "Data/depth_maps"
    cameras, images, points3D = read_model(colmap_model_path)

    visualize = False

    processor, model = load_da_v2()
    model = model.to(device)
    model.eval()

    depth_metric_maps = {}  # image_id -> torch.Tensor(H,W)
    scales = {}             # image_id -> float

    for image_id, img in images.items():
        img_path = os.path.join(images_dir, img.name)
        rgb = Image.open(img_path).convert("RGB")

        # Run depth prediction (same size as RGB)
        depth_rel_i =  predict_depth(rgb, processor, model)  # (1,H,W)

        h_i, w_i = rgb.size[1], rgb.size[0] 

        # Compute per-image scale from COLMAP geometry
        s_i = compute_image_depth_scale(depth_rel_i, img, points3D, h_i, w_i, device=device)
        if s_i is None:
            print(f"[WARN] Image {img.name}: not enough valid correspondences for scale.")
            continue

        scales[image_id] = s_i

        # Metric depth = scale * relative depth
        depth_metric_i = depth_rel_i[0] * s_i  # (H,W)

        if visualize:
            plt.figure(figsize=(15,5))
            plt.subplot(1,2,1)
            plt.imshow(rgb)
            plt.axis('off')
            plt.title('Original Image')
            plt.subplot(1,2,2)
            plt.imshow(depth_rel_i.squeeze().cpu().numpy(), cmap='plasma')
            plt.axis('off')
            plt.title('Depth Map')
            plt.subplot(1,3,2)
            plt.imshow(depth_metric_i.squeeze().cpu().numpy(), cmap='plasma')
            plt.axis('off')
            plt.title('Depth Map Scaled')

            plt.show()

            plt.close()

        plt.imshow(depth_metric_i.squeeze().cpu().numpy(), cmap='plasma')
        plt.axis('off')
        plt.savefig('{}/{}'.format(output_dir, img.name))


        depth_metric_maps[image_id] = depth_metric_i.cpu()

        print(f"Image {img.name}: scale = {s_i:.4f}, depth_metric shape = {depth_metric_i.shape}")
    # np.savez('dept_maps', depth_metric_maps) # It is a pain to read the dict, better use the pickle
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(depth_metric_maps, f)


if __name__ == "__main__":
    # check_depth_estimator()
    main()
