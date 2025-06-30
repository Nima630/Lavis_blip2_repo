import torch
import matplotlib.pyplot as plt

def create_bev_grid(H=90, W=90, x_range=(-51.2, 51.2), y_range=(-51.2, 51.2)):
    xs = torch.linspace(x_range[0], x_range[1], W)
    ys = torch.linspace(y_range[0], y_range[1], H)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    zeros = torch.zeros_like(xx)
    bev_xyz = torch.stack([xx, yy, zeros, torch.ones_like(xx)], dim=-1)  # [H, W, 4]
    return bev_xyz.view(-1, 4)  # [H*W, 4]

def project_points_with_lidar2img(bev_points, lidar2img):
    proj = (lidar2img @ bev_points.T).T  # [N, 4]
    eps = 1e-5
    proj[:, :2] = proj[:, :2] / (proj[:, 2:3] + eps)  # Normalize by Z
    return proj[:, :2], proj[:, 2]

def generate_bev_feature_map(cam_feats, lidar2img_matrices, image_size):
    N_cam, _, C, Hf, Wf = cam_feats.shape
    bev_grid = create_bev_grid(H=90, W=90)
    bev_features = torch.zeros(bev_grid.shape[0], C)
    counts = torch.zeros(bev_grid.shape[0], 1)

    for i in range(N_cam):
        img_feat = cam_feats[i, 0]  # [C, Hf, Wf]
        proj_pts, depth = project_points_with_lidar2img(bev_grid, lidar2img_matrices[i])

        valid = (depth > 0) & \
                (proj_pts[:, 0] >= 0) & (proj_pts[:, 0] < image_size[1]) & \
                (proj_pts[:, 1] >= 0) & (proj_pts[:, 1] < image_size[0])

        px = proj_pts[valid].long()
        feat_vals = img_feat[:, px[:, 1], px[:, 0]].T  # [N, C]
        bev_features[valid] += feat_vals
        counts[valid] += 1

    counts[counts == 0] = 1
    bev_features /= counts
    return bev_features.view(90, 90, C)

def main():
    # Load from your dataset
    sample = torch.load("path/to/sample.pt")  # Replace with actual sample path

    cam_feats = sample["images"]           # [6, 1, 2048, 29, 50]
    lidar2img = sample["lidar2img"]        # [6, 4, 4]
    img_shape = sample["img_shapes"][0]    # [(928, 1600, 3)], we only need [0:2]

    bev_map = generate_bev_feature_map(cam_feats, lidar2img, image_size=(29, 50))
    bev_map_np = bev_map.detach().cpu().numpy()

    # Visualize one channel
    plt.imshow(bev_map_np[:, :, 0], cmap="viridis")
    plt.title("BEV Map (Channel 0)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
