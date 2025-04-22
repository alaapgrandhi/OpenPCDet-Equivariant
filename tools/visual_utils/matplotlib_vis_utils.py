import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

def draw_bev_features(img_features, lidar_features, fused_features):
    C1, H, W = img_features.shape
    C2, H, W = lidar_features.shape
    if isinstance(img_features, torch.Tensor):
        img_features = img_features.reshape(C1, H*W).transpose(0, 1).cpu().numpy()
    if isinstance(lidar_features, torch.Tensor):
        lidar_features = lidar_features.reshape(C2, H*W).transpose(0, 1).cpu().numpy()
    if isinstance(fused_features, torch.Tensor):
        fused_features = fused_features.reshape(C2, H*W).transpose(0, 1).cpu().numpy()

    scaler_img = StandardScaler()
    scaler_lidar = StandardScaler()
    scaler_bev = StandardScaler()

    img_features = scaler_img.fit_transform(img_features)
    lidar_features = scaler_lidar.fit_transform(lidar_features)
    fused_features = scaler_bev.fit_transform(fused_features)

    pca_img = PCA(n_components=3)
    pca_lidar = PCA(n_components=3)
    pca_bev = PCA(n_components=3)

    img_rgb_feats = pca_img.fit_transform(img_features)
    img_min = np.min(img_rgb_feats, axis=0)
    img_max = np.max(img_rgb_feats, axis=0)
    img_rgb_feats = (img_rgb_feats-img_min)/img_max
    img_rgb_feats = img_rgb_feats.reshape(H, W, 3)

    lidar_rgb_feats = pca_lidar.fit_transform(lidar_features)
    lidar_min = np.min(lidar_rgb_feats, axis=0)
    lidar_max = np.max(lidar_rgb_feats, axis=0)
    lidar_rgb_feats = (lidar_rgb_feats - lidar_min) / lidar_max
    lidar_rgb_feats = lidar_rgb_feats.reshape(H, W, 3)

    bev_rgb_feats = pca_bev.fit_transform(fused_features)
    bev_min = np.min(bev_rgb_feats, axis=0)
    bev_max = np.max(bev_rgb_feats, axis=0)
    bev_rgb_feats = (bev_rgb_feats - bev_min) / bev_max
    bev_rgb_feats = bev_rgb_feats.reshape(H, W, 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex="all", sharey="all")

    axes[0].imshow(img_rgb_feats)
    axes[0].set_title("Image BEV Features after PCA")

    axes[1].imshow(lidar_rgb_feats)
    axes[1].set_title("Lidar BEV Features after PCA")

    axes[2].imshow(bev_rgb_feats)
    axes[2].set_title("Fused BEV Features after PCA")

    plt.tight_layout()
    plt.show()

def draw_bev_features_tsne(img_features, lidar_features, fused_features):
    C1, H, W = img_features.shape
    C2, H, W = lidar_features.shape
    if isinstance(img_features, torch.Tensor):
        img_features = img_features.reshape(C1, H*W).transpose(0, 1).cpu().numpy()
    if isinstance(lidar_features, torch.Tensor):
        lidar_features = lidar_features.reshape(C2, H*W).transpose(0, 1).cpu().numpy()
    if isinstance(fused_features, torch.Tensor):
        fused_features = fused_features.reshape(C2, H*W).transpose(0, 1).cpu().numpy()

    scaler_img = StandardScaler()
    scaler_lidar = StandardScaler()
    scaler_bev = StandardScaler()

    img_features = scaler_img.fit_transform(img_features)
    lidar_features = scaler_lidar.fit_transform(lidar_features)
    fused_features = scaler_bev.fit_transform(fused_features)

    tsne_img = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='random', random_state=42)
    tsne_lidar = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='random', random_state=42)
    tsne_bev = TSNE(n_components=3, perplexity=30, learning_rate='auto', init='random', random_state=42)

    img_rgb_feats = tsne_img.fit_transform(img_features)
    img_min = np.min(img_rgb_feats, axis=0)
    img_max = np.max(img_rgb_feats, axis=0)
    img_rgb_feats = (img_rgb_feats-img_min)/img_max
    img_rgb_feats = img_rgb_feats.reshape(H, W, 3)

    lidar_rgb_feats = tsne_lidar.fit_transform(lidar_features)
    lidar_min = np.min(lidar_rgb_feats, axis=0)
    lidar_max = np.max(lidar_rgb_feats, axis=0)
    lidar_rgb_feats = (lidar_rgb_feats - lidar_min) / lidar_max
    lidar_rgb_feats = lidar_rgb_feats.reshape(H, W, 3)

    bev_rgb_feats = tsne_bev.fit_transform(fused_features)
    bev_min = np.min(bev_rgb_feats, axis=0)
    bev_max = np.max(bev_rgb_feats, axis=0)
    bev_rgb_feats = (bev_rgb_feats - bev_min) / bev_max
    bev_rgb_feats = bev_rgb_feats.reshape(H, W, 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex="all", sharey="all")

    axes[0].imshow(img_rgb_feats)
    axes[0].set_title("Image BEV Features after t-SNE")

    axes[1].imshow(lidar_rgb_feats)
    axes[1].set_title("Lidar BEV Features after t-SNE")

    axes[2].imshow(bev_rgb_feats)
    axes[2].set_title("Fused BEV Features after t-SNE")

    plt.tight_layout()
    plt.show()


def draw_bev_features_umap(img_features, lidar_features, fused_features):
    C1, H, W = img_features.shape
    C2, H, W = lidar_features.shape
    if isinstance(img_features, torch.Tensor):
        img_features = img_features.reshape(C1, H*W).transpose(0, 1).cpu().numpy()
    if isinstance(lidar_features, torch.Tensor):
        lidar_features = lidar_features.reshape(C2, H*W).transpose(0, 1).cpu().numpy()
    if isinstance(fused_features, torch.Tensor):
        fused_features = fused_features.reshape(C2, H*W).transpose(0, 1).cpu().numpy()

    scaler_img = StandardScaler()
    scaler_lidar = StandardScaler()
    scaler_bev = StandardScaler()

    img_features = scaler_img.fit_transform(img_features)
    lidar_features = scaler_lidar.fit_transform(lidar_features)
    fused_features = scaler_bev.fit_transform(fused_features)

    umap_img = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_lidar = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_bev = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)

    img_rgb_feats = umap_img.fit_transform(img_features)
    img_min = np.min(img_rgb_feats, axis=0)
    img_max = np.max(img_rgb_feats, axis=0)
    img_rgb_feats = (img_rgb_feats-img_min)/img_max
    img_rgb_feats = img_rgb_feats.reshape(H, W, 3)

    lidar_rgb_feats = umap_lidar.fit_transform(lidar_features)
    lidar_min = np.min(lidar_rgb_feats, axis=0)
    lidar_max = np.max(lidar_rgb_feats, axis=0)
    lidar_rgb_feats = (lidar_rgb_feats - lidar_min) / lidar_max
    lidar_rgb_feats = lidar_rgb_feats.reshape(H, W, 3)

    bev_rgb_feats = umap_bev.fit_transform(fused_features)
    bev_min = np.min(bev_rgb_feats, axis=0)
    bev_max = np.max(bev_rgb_feats, axis=0)
    bev_rgb_feats = (bev_rgb_feats - bev_min) / bev_max
    bev_rgb_feats = bev_rgb_feats.reshape(H, W, 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex="all", sharey="all")

    axes[0].imshow(img_rgb_feats)
    axes[0].set_title("Image BEV Features after UMAP")

    axes[1].imshow(lidar_rgb_feats)
    axes[1].set_title("Lidar BEV Features after UMAP")

    axes[2].imshow(bev_rgb_feats)
    axes[2].set_title("Fused BEV Features after UMAP")

    plt.tight_layout()
    plt.show()
