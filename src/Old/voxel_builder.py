import torch
import numpy as np
from colmap_utils import qvec2rotmat   # your function
from colmap_utils import read_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pickle


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


def compute_bounds_from_points(points3d_dict, margin=0.5):
    """
    points3d_dict: dict[point3D_id -> Point3D]
    Returns:
        xyz_min, xyz_max: (3,) torch.float32
    """
    xyz = np.stack([p.xyz for p in points3d_dict.values()], axis=0)  # (M,3)
    xyz_min = torch.from_numpy(xyz.min(axis=0)).float() - margin
    xyz_max = torch.from_numpy(xyz.max(axis=0)).float() + margin
    return xyz_min, xyz_max


def create_voxel_grid(xyz_min, xyz_max, voxel_size=0.05, device="cpu"):
    """
    voxel_size: scalar (m) for isotropic voxels
    Returns:
        origin: (3,) lower corner of grid (world coords)
        dims:   (3,) int (nx, ny, nz)
        voxel_size: float
    """
    vs = float(voxel_size)
    extents = xyz_max - xyz_min  # (3,)
    nx = int(torch.ceil(extents[0] / vs).item())
    ny = int(torch.ceil(extents[1] / vs).item())
    nz = int(torch.ceil(extents[2] / vs).item())
    origin = xyz_min.clone().to(device)
    dims = torch.tensor([nx, ny, nz], dtype=torch.long)
    return origin, dims, vs


def init_occupancy(dims, device="cpu"):
    nx, ny, nz = dims.tolist()
    hits  = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device)
    frees = torch.zeros((nx, ny, nz), dtype=torch.float32, device=device)
    return hits, frees


def world_to_voxel(x_w, origin, voxel_size, dims):
    """
    x_w: (...,3) world coords
    origin: (3,) world coords of voxel (0,0,0) corner
    voxel_size: float
    dims: (3,) [nx,ny,nz]
    Returns:
        ijk: (...,3) long; -1 for out-of-bounds
    """
    rel = (x_w - origin) / voxel_size   # (...,3)
    idx = torch.floor(rel).long()
    nx, ny, nz = dims.tolist()
    valid = (idx[..., 0] >= 0) & (idx[..., 0] < nx) & \
            (idx[..., 1] >= 0) & (idx[..., 1] < ny) & \
            (idx[..., 2] >= 0) & (idx[..., 2] < nz)
    idx[~valid] = -1
    return idx, valid


def bresenham3d(start_idx, end_idx):
    """
    Integer 3D Bresenham between two voxel indices.
    start_idx, end_idx: (3,) long
    Returns:
        list[ (i,j,k) ] including start, excluding end (so end is the surface voxel).
    """
    x1, y1, z1 = start_idx.tolist()
    x2, y2, z2 = end_idx.tolist()

    voxels = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    # Driving axis is X
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            voxels.append((x1, y1, z1))
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

    # Driving axis is Y
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            voxels.append((x1, y1, z1))
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz

    # Driving axis is Z
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            voxels.append((x1, y1, z1))
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    return voxels


def integrate_depth_frame(
    depth,         # (H,W) metric depth, torch
    img,           # COLMAP Image
    cam,           # COLMAP Camera
    origin, dims, voxel_size,
    hits, frees,
    device="cuda",
    depth_min=0.1, depth_max=20.0
):
    """
    Update hits and frees in-place using one depth frame.
    """
    depth = depth.to(device)  # (H,W)
    R_wc, C_w = get_world_from_cam(img)
    R_wc = R_wc.to(device)
    C_w  = C_w.to(device)

    K = get_intrinsics_matrix(cam).to(device)
    K_inv = torch.inverse(K)

    H, W = depth.shape
    # pixel grid
    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")   # (W,H) if indexing="xy"
    # uu = uu.T  # to (H,W)
    # vv = vv.T

    # valid depth mask
    D = depth
    valid = (D > depth_min) & (D < depth_max) & torch.isfinite(D)
    if valid.sum().item() == 0:
        return

    # get pixel coordinates for valid depths
    u_valid = uu[valid].float()
    v_valid = vv[valid].float()
    d_valid = D[valid].float()

    # backproject to camera frame
    ones = torch.ones_like(u_valid)
    pix = torch.stack([u_valid, v_valid, ones], dim=-1)  # (N,3)
    rays_c = (K_inv @ pix.T).T                           # (N,3) directions
    X_c = rays_c * d_valid.unsqueeze(-1)                 # (N,3)

    # to world frame
    X_w = (R_wc @ X_c.T).T + C_w.unsqueeze(0)            # (N,3)

    # camera center voxel index
    C_w_batch = C_w.unsqueeze(0)  # (1,3)
    cam_vox_idx, cam_valid = world_to_voxel(C_w_batch, origin, voxel_size, dims)
    if not cam_valid.item():
        # camera is outside grid -> skip frame or enlarge grid in practice
        return
    cam_vox = cam_vox_idx[0]  # (3,)

    nx, ny, nz = dims.tolist()

    # update occupancy per ray
    for i in range(X_w.shape[0]):
        x = X_w[i]
        # voxel of surface hit
        hit_idx, v_ok = world_to_voxel(x.unsqueeze(0), origin, voxel_size, dims)
        if not v_ok.item():
            continue
        hit_vox = hit_idx[0]  # (3,)

        # if identical voxel as camera (very close), just mark hit
        if torch.all(hit_vox == cam_vox):
            ix, iy, iz = hit_vox.tolist()
            hits[ix, iy, iz] += 1.0
            continue

        # free voxels along ray
        free_voxels = bresenham3d(cam_vox, hit_vox)
        for (ix, iy, iz) in free_voxels:
            if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                frees[ix, iy, iz] += 1.0

        # occupied surface voxel
        ix, iy, iz = hit_vox.tolist()
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            hits[ix, iy, iz] += 1.0


def integrate_all_frames(
    depth_metric_maps, cameras, images, points3D,
    voxel_size=0.05, device="cpu"
):
    # 1) Bounds from COLMAP sparse points
    xyz_min, xyz_max = compute_bounds_from_points(points3D)
    xyz_min, xyz_max = xyz_min.to(device), xyz_max.to(device)

    # 2) Grid
    origin, dims, vs = create_voxel_grid(xyz_min, xyz_max, voxel_size, device=device)
    hits, frees = init_occupancy(dims, device=device)

    # 3) Integrate each frame
    for image_id, img in images.items():
        if image_id not in depth_metric_maps:
            continue
        depth = depth_metric_maps[image_id]  # (H,W) torch
        cam = cameras[img.camera_id]
        integrate_depth_frame(depth, img, cam, origin, dims, vs, hits, frees, device=device)

    return origin, dims, vs, hits, frees


def occupancy_from_counts(hits, frees, occ_thresh=0.5):
    eps = 1e-6
    p_occ = hits / (hits + frees + eps)
    occ = torch.full_like(p_occ, fill_value=-1.0)  # -1 unknown
    # known cells
    known = (hits + frees) > 0
    occ[known] = (p_occ[known] > occ_thresh).float()
    return p_occ, occ  # p_occ in [0,1], occ in {-1,0,1} -> unknown/free/occ


def build_occupancy_voxels(
    colmap_model_path: str,
    depth_metric_maps: dict,
    voxel_size: float = 0.05,
    device: str = "cuda"
    ):
    """
    Build a 3D voxel occupancy grid from:
      - COLMAP cameras, images, 3D points
      - per-image metric depth maps

    Args:
        colmap_model_path: path to the COLMAP model folder (with cameras/images/points3D).
        depth_metric_maps: dict[image_id -> torch.Tensor(H,W)] of metric depths (meters).
                           image_id must match keys in the COLMAP `images` dict.
        voxel_size: size of each voxel in meters (isotropic).
        device: "cuda" or "cpu"

    Returns:
        origin:   (3,) torch.float32, world coords of voxel (0,0,0) corner
        dims:     (3,) long, number of voxels in x,y,z -> (nx,ny,nz)
        voxel_sz: float, same as voxel_size
        hits:     (nx,ny,nz) float32, # of surface hits per voxel
        frees:    (nx,ny,nz) float32, # of free-space observations per voxel
        p_occ:    (nx,ny,nz) float32, occupancy probability per voxel in [0,1]
        occ:      (nx,ny,nz) float32, -1 = unknown, 0 = free, 1 = occupied
    """
    device = torch.device(device)

    # 1) Read COLMAP model
    cameras, images, points3D = read_model(colmap_model_path)

    # 2) Derive bounds from COLMAP sparse 3D points
    xyz_min, xyz_max = compute_bounds_from_points(points3D)
    xyz_min, xyz_max = xyz_min.to(device), xyz_max.to(device)

    # 3) Create voxel grid
    origin, dims, vs = create_voxel_grid(xyz_min, xyz_max, voxel_size, device=device)
    hits, frees = init_occupancy(dims, device=device)

    # 4) Integrate each depth frame
    for image_id, img in images.items():
        if image_id not in depth_metric_maps:
            continue

        depth = depth_metric_maps[image_id]  # (H,W) torch, metric
        cam = cameras[img.camera_id]

        integrate_depth_frame(
            depth=depth,
            img=img,
            cam=cam,
            origin=origin,
            dims=dims,
            voxel_size=vs,
            hits=hits,
            frees=frees,
            device=device,
        )

    # 5) Convert hits/frees into occupancy probabilities
    p_occ, occ = occupancy_from_counts(hits, frees, occ_thresh=0.5)



    return origin, dims, vs, hits, frees, p_occ, occ



if __name__ == "__main__":

    root_dir = "/home/mtoso/Documents/Code/SemanticMapsFromSfM/"
    depth_path = 'saved_dictionary.pkl'
    
    with open(depth_path, 'rb') as f:
        depth = pickle.load(f)

    origin, dims, vs, hits, frees, p_occ, occ = build_occupancy_voxels(
        root_dir + '/Data/colmap_model',
        depth,
        voxel_size=0.05,
        device=device
    )

    print('pippo')