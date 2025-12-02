import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from sh_utils import SH2RGB
import json
from argparse import ArgumentParser
from sh_utils import SH2RGB
# from open_vocab_segmentor.encoders.openclip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
import open3d as o3d
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
import numpy as np
from sklearn.cluster import HDBSCAN
import scipy.sparse as sp 
from types import SimpleNamespace
from plyfile import PlyData


def plot3D(
    pts: np.ndarray,
    downsample = None,
    point_size: float = 1.0,
):
    """
    Quick 3D scatter of a 3D pointcloud using matplotlib.
    """
    import matplotlib.pyplot as plt

    if downsample is not None:
        keys = np.random.randint(0, len(pts), downsample)
        pts = pts[keys]

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


def get_csr_matrix(similarity, neighbor_indices_tensor):
    similarity = similarity.cpu().numpy()
    neighbor_indices_tensor = neighbor_indices_tensor.cpu().numpy()
    
    indices = []
    data = []

    n = similarity.shape[0]

    for i in range(n):
        w_per_neighbor = similarity[i]
        neighbor_indices = neighbor_indices_tensor[i]
        for j, w in enumerate(w_per_neighbor):
            k = neighbor_indices[j]
            if i < k:
                if w < 0.1:
                    indices.append([i, k])
                    data.append(w)
    
    return np.asarray(indices), np.asarray(data)


def graph(points, k=11):
    n_p = points.shape[0]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(points, points)  # Compute pairwise distances
    data, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    indices, values = get_csr_matrix(data, neighbor_indices_tensor)
    rows = indices[:, 0]
    cols = indices[:, 1]

    graph = sp.csr_matrix((values, (rows, cols)), shape=(n_p, n_p))

    num_components, segmentation = sp.csgraph.connected_components(graph, directed=False)

    return segmentation


def seg_3D(features, feature_prompt):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    logits = cos(features, feature_prompt[..., None])
    segmask = logits >= 0.85
    return segmask, logits


def cluster_by_position(points):
    hdb = HDBSCAN(min_cluster_size=30, store_centers='centroid', n_jobs=-1)
    hdb.fit(points)

    return np.asarray(hdb.labels_)


def pc_to_bbox(points):
    xyz_min = torch.min(points, dim=0)[0].squeeze()
    xyz_max = torch.max(points, dim=0)[0].squeeze()

    bbox = torch.zeros((8, 3))
        
    bbox[:4, 0] = xyz_min[0].item()
    bbox[4:, 0] = xyz_max[0].item()
    bbox[:2, 1] = xyz_min[1].item()
    bbox[4:6, 1] = xyz_min[1].item()
    bbox[2:4, 1] = xyz_max[1].item()
    bbox[6:, 1] = xyz_max[1].item()
    bbox[0, 2] = xyz_min[2].item()
    bbox[2, 2] = xyz_min[2].item()
    bbox[4, 2] = xyz_min[2].item()
    bbox[6, 2] = xyz_min[2].item()
    bbox[1, 2] = xyz_max[2].item()
    bbox[3, 2] = xyz_max[2].item()
    bbox[5, 2] = xyz_max[2].item()
    bbox[7, 2] = xyz_max[2].item()

    return bbox


def extract_data_from_gaussians(model_path, device, seg_features_dim=512):
    plydata = PlyData.read(model_path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

    lang_features = np.zeros((xyz.shape[0], seg_features_dim, 1))
    for idx in range(seg_features_dim):
        lang_features[:, idx, 0] = np.asarray(plydata.elements[0]["f_lang_"+str(idx)])
    lang_features = torch.tensor(lang_features, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(False)
    classes = lang_features[:, 0, :]
    classes /= (classes.norm(dim=1, keepdim=True) + 1e-6)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    colors = torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(False)

    colors_ = features_dc[:, 0, :]
    colors_ = SH2RGB(colors_)

    return xyz, classes, colors


if __name__ == "__main__":
    model_path = 'Data/2DGS Model/cris_lab_full.ply'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    xyz, classes, colors = extract_data_from_gaussians(model_path, device, seg_features_dim=512)
    
    xyz_o3d = o3d.geometry.PointCloud()
    xyz_o3d.points = o3d.utility.Vector3dVector(xyz)

    cl, ind = xyz_o3d.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
    
    o3d.visualization.draw_geometries([cl.voxel_down_sample(voxel_size=0.02)])
    
    xyz_final = np.asarray(xyz_o3d.points)
    
    plot3D(xyz, 15000)
    print('pippo')