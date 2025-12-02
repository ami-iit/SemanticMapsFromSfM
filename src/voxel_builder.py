import torch
import numpy as np
from colmap_utils import qvec2rotmat   # your function
from colmap_utils import read_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    xyz_mean = xyz.mean(axis=0)
    xyz_std = xyz.std(axis=0)

    mask = np.abs(xyz - xyz_mean) <= 2.0 * xyz_std
    mask = np.all(mask, axis=1)

    xyz_min = torch.from_numpy(xyz[mask].min(axis=0)).float() - margin
    xyz_max = torch.from_numpy(xyz[mask].max(axis=0)).float() + margin


    show_occ_scatter(xyz[mask])

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
    voxel_size=0.05, device="cuda"
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

    np.save('origin', origin)
    np.save('dims', dims)
    np.save('hits', hits)
    np.save('frees', frees)

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
    output_path: str,
    voxel_size: float = 0.25,
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
        output_path: path where to store the voxelized occupancy grid (as numpy arrays).
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

    print('Working towards registering {} images...'.format(len(images.items())))

    for image_id, img in images.items():
        print('wip...')
        if image_id not in depth_metric_maps:
            continue
        print('registering image {}'.format(image_id))
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
    _, occ = occupancy_from_counts(hits, frees, occ_thresh=0.5)

    np.save(output_path + '/origin', origin)
    np.save(output_path + '/dims', dims)
    np.save(output_path + '/vs', vs)
    np.save(output_path + '/occ', occ)

    return origin, dims, vs, occ


def occ_to_points(
    origin: np.ndarray,
    dims: np.ndarray,
    vs: float,
    occ: np.ndarray,
    max_points = None,
    ) -> np.ndarray:
    """
    Convert a 3D occupancy grid into world-space point cloud.

    Args:
        origin: (3,) world coords of voxel (0,0,0)
        dims: (3,) number of voxels in each dimension [Nx, Ny, Nz]
        vs: voxel size
        occ: (Nx, Ny, Nz) occupancy values: 1 = occupied, -1 = free
        max_points: optional random subsampling budget

    Returns:
        points_w: (M,3) world coordinates of voxel centers
    """
    occ = np.asarray(occ)
    origin = np.asarray(origin, dtype=np.float32)
    dims = np.asarray(dims, dtype=np.int32)
    assert occ.shape == tuple(dims), f"occ.shape {occ.shape} != dims {tuple(dims)}"
    idxs = np.argwhere(occ == 1)  # (M,3) indices (ix, iy, iz)
    if idxs.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if max_points is not None and idxs.shape[0] > max_points:
        perm = np.random.permutation(idxs.shape[0])
        idxs = idxs[perm[:max_points]]

    # Voxel center: origin + (idx + 0.5) * vs
    centers = (idxs.astype(np.float32) + 0.5) * vs
    points_w = origin[None, :] + centers
    return points_w


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


if __name__ == "__main__":

    import pickle
    root_dir = '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/'

    depth_path = root_dir + 'saved_dictionary.pkl'
    output_path = root_dir + '/Data/output'

    with open(depth_path, 'rb') as f:
        depth = pickle.load(f)

    # depth_path = '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/dept_maps.npz'
    # detph = np.load(depth_path, allow_pickle=True)

    origin, dims, vs, occ = build_occupancy_voxels(
        '/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Data/colmap_model',
        depth,
        output_path,
        voxel_size=0.05,
        device=device
    )
    origin = np.load(output_path + '/origin.npy')
    dims = np.load(output_path + '/dims.npy')
    vs = np.load(output_path + '/vs.npy')
    occ = np.load(output_path + '/occ.npy')

    show_occ_scatter(origin, dims, vs, occ)

    points = occ_to_points(origin, dims, vs, occ)

    print('pippo')
