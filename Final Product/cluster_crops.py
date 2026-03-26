#!/usr/bin/env python3
"""
cluster_crops.py

Terminal-friendly clustering pipeline for insect crops.

Pipeline:
  1) Load crops from --image_dir
  2) Extract DINOv2 embeddings (timm) with caching
  3) PCA -> (optional) silhouette scan to pick k
  4) KMeans clustering
  5) Select 1 anchor per cluster (most representative via mean cosine similarity)
  6) Save:
      - out_dir/cluster_manifest.csv (paths, cluster, coords, anchor, sims)
      - out_dir/anchors.csv
      - out_dir/clusters/cluster_XXX/ (anchor + ranked images)
      - (optional) out_dir/umap_dashboard.html (interactive)

Example:
  python cluster_crops.py --image_dir ../crops/kept --out_dir ../clusters_output --k 20 --export_html

Notes:
- If you want HTML export, install: umap-learn plotly
- By default, only .jpg/.jpeg/.png/.webp/.bmp/.tif/.tiff are processed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


SEED_DEFAULT = 42
IMG_EXTS_DEFAULT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(image_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    exts_lower = tuple(e.lower() for e in exts)
    files = []
    for p in sorted(image_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts_lower:
            files.append(p)
    return files


def load_dinov2_timm(model_name: str, device: str) -> tuple[torch.nn.Module, object]:
    model = create_model(model_name, pretrained=True).to(device)
    model.eval()
    config = resolve_data_config({}, model=model)
    preprocess = create_transform(**config)
    return model, preprocess


@torch.no_grad()
def compute_embeddings_dinov2(
    paths: List[Path],
    model: torch.nn.Module,
    preprocess,
    device: str,
    batch_size: int,
) -> np.ndarray:
    embs = []
    kept_paths = 0

    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting embeddings"):
        batch_imgs = []
        for p in paths[i : i + batch_size]:
            try:
                img = Image.open(p).convert("RGB")
                img_t = preprocess(img).unsqueeze(0)
                batch_imgs.append(img_t)
                kept_paths += 1
            except Exception as e:
                print(f"[warn] Skipping unreadable image: {p} ({e})")

        if not batch_imgs:
            continue

        batch = torch.cat(batch_imgs).to(device)

        # autocast helps speed on GPU; on CPU it's disabled.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=torch.cuda.is_available() and device.startswith("cuda"),
        ):
            feats = model.forward_features(batch)

        # global mean pooling -> (B, D)
        pooled = feats.mean(dim=1)
        pooled = nn.functional.normalize(pooled, dim=-1)

        embs.append(pooled.detach().cpu().numpy())

    if not embs:
        return np.zeros((0, 0), dtype=np.float32)

    X = np.concatenate(embs, axis=0)
    print(f"✅ Embeddings extracted: {X.shape} (images processed: {kept_paths}/{len(paths)})")
    return X


def pca_reduce(X: np.ndarray, n_components: int, whiten: bool, seed: int) -> tuple[np.ndarray, PCA]:
    n_components = min(n_components, X.shape[1])
    pca = PCA(n_components=n_components, random_state=seed, whiten=whiten)
    Xr = pca.fit_transform(X)
    return Xr, pca


def pick_best_k_silhouette(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    seed: int,
    sample_size: Optional[int] = None,
) -> tuple[int, dict]:
    """
    k in [k_min, k_max] inclusive.
    sample_size: silhouette sampling for speed (None = full).
    """
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=sample_size, random_state=seed)
        scores[k] = float(sil)
        print(f"k={k:02d} -> silhouette={sil:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"✅ Best k={best_k} (silhouette={scores[best_k]:.4f})")
    return best_k, scores


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def select_anchors_mean_cosine(X: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """
    Anchor per cluster = point with highest mean cosine similarity to others in cluster.
    Efficient: use dot products within each cluster after L2-normalizing.
    Returns: {cluster_id: global_index}
    """
    Xn = l2_normalize(X)
    anchors: dict[int, int] = {}
    for cid in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            anchors[cid] = int(idx[0])
            continue
        sims = Xn[idx] @ Xn[idx].T  # (m, m)
        mean_sims = sims.mean(axis=1)
        anchors[cid] = int(idx[int(np.argmax(mean_sims))])
    return anchors


def rank_within_cluster_by_anchor_sim(X: np.ndarray, labels: np.ndarray, anchors: dict[int, int]) -> dict[int, list[tuple[int, float]]]:
    """
    For each cluster: returns list of (global_index, cosine_sim_to_anchor) sorted desc.
    """
    Xn = l2_normalize(X)
    out: dict[int, list[tuple[int, float]]] = {}
    for cid in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue
        a = anchors[cid]
        sims = (Xn[idx] @ Xn[a]).astype(np.float32)  # (m,)
        order = np.argsort(-sims)
        out[cid] = [(int(idx[j]), float(sims[j])) for j in order]
    return out


def safe_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        except Exception:
            # fallback to copy
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_cache_if_valid(cache_prefix: Path, paths: List[Path]) -> Optional[np.ndarray]:
    """
    Cache is:
      - <prefix>.npy
      - <prefix>.paths.json  (list of absolute paths)
    """
    npy = cache_prefix.with_suffix(".npy")
    meta = cache_prefix.with_suffix(".paths.json")

    if not npy.exists() or not meta.exists():
        return None

    try:
        cached_paths = json.loads(meta.read_text(encoding="utf-8"))
        cur_paths = [str(p.resolve()) for p in paths]
        if cached_paths != cur_paths:
            return None
        X = np.load(npy)
        return X
    except Exception:
        return None


def save_cache(cache_prefix: Path, paths: List[Path], X: np.ndarray) -> None:
    npy = cache_prefix.with_suffix(".npy")
    meta = cache_prefix.with_suffix(".paths.json")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, X)
    write_json(meta, [str(p.resolve()) for p in paths])


def export_umap_html(
    out_path: Path,
    X_for_umap: np.ndarray,
    labels: np.ndarray,
    anchors: dict[int, int],
    paths: List[Path],
    dim: int = 2,
    n_neighbors: int = 20,
    min_dist: float = 0.1,
    metric: str = "cosine",
    max_points: int = 15000,
    seed: int = 42,
) -> None:
    """
    Optional dependency: umap-learn + plotly.
    Creates an interactive UMAP scatter with dropdown toggles per cluster + anchors.
    Subsamples points if too many (keeps all anchors).
    """
    try:
        import umap  # type: ignore
        import plotly.graph_objects as go  # type: ignore
    except Exception as e:
        raise RuntimeError("HTML export requires `umap-learn` and `plotly`. Install them and retry.") from e

    if dim not in (2, 3):
        raise ValueError("--umap_dim must be 2 or 3")

    n = X_for_umap.shape[0]
    anchor_idx = set(anchors.values())

    # Subsample if needed (keep all anchors)
    if n > max_points:
        non_anchor = [i for i in range(n) if i not in anchor_idx]
        keep_non_anchor = max_points - len(anchor_idx)
        keep_non_anchor = max(0, keep_non_anchor)
        chosen = set(anchor_idx)
        if keep_non_anchor > 0:
            chosen.update(random.Random(seed).sample(non_anchor, k=min(keep_non_anchor, len(non_anchor))))
        chosen = sorted(chosen)
        X_use = X_for_umap[chosen]
        labels_use = labels[chosen]
        map_idx = {old: new for new, old in enumerate(chosen)}
        anchors_use = {cid: map_idx[i] for cid, i in anchors.items() if i in map_idx}
        paths_use = [paths[i] for i in chosen]
    else:
        X_use = X_for_umap
        labels_use = labels
        anchors_use = anchors
        paths_use = paths

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=dim,
        metric=metric,
        random_state=seed,
    )
    emb = reducer.fit_transform(X_use)

    fig = go.Figure()
    unique_clusters = sorted(np.unique(labels_use).tolist())

    # One trace per cluster (toggleable)
    for cid in unique_clusters:
        mask = labels_use == cid
        if dim == 2:
            fig.add_trace(go.Scattergl(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                name=f"Cluster {cid}",
                marker=dict(size=4, opacity=0.45),
                text=[paths_use[i].name for i, m in enumerate(mask) if m],
                hovertemplate="File: %{text}<br>Cluster: " + str(cid) + "<extra></extra>",
                visible=True
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=emb[mask, 0],
                y=emb[mask, 1],
                z=emb[mask, 2],
                mode="markers",
                name=f"Cluster {cid}",
                marker=dict(size=3, opacity=0.45),
                text=[paths_use[i].name for i, m in enumerate(mask) if m],
                hovertemplate="File: %{text}<br>Cluster: " + str(cid) + "<extra></extra>",
                visible=True
            ))

    # Anchors trace
    anchor_points = np.array([anchors_use[cid] for cid in sorted(anchors_use.keys())], dtype=int)
    anchor_text = [f"Anchor C{cid}: {paths_use[anchors_use[cid]].name}" for cid in sorted(anchors_use.keys())]

    if dim == 2:
        fig.add_trace(go.Scattergl(
            x=emb[anchor_points, 0],
            y=emb[anchor_points, 1],
            mode="markers+text",
            name="Anchors",
            marker=dict(size=10, color="black", symbol="diamond"),
            text=[f"C{cid}" for cid in sorted(anchors_use.keys())],
            textposition="top center",
            hovertext=anchor_text,
            hovertemplate="%{hovertext}<extra></extra>",
            visible=True
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=emb[anchor_points, 0],
            y=emb[anchor_points, 1],
            z=emb[anchor_points, 2],
            mode="markers+text",
            name="Anchors",
            marker=dict(size=8, color="black", symbol="diamond"),
            text=[f"C{cid}" for cid in sorted(anchors_use.keys())],
            textposition="top center",
            hovertext=anchor_text,
            hovertemplate="%{hovertext}<extra></extra>",
            visible=True
        ))

    n_traces = len(unique_clusters)
    buttons = [
        dict(label="Show All", method="update", args=[{"visible": [True] * (n_traces + 1)}]),
        dict(label="Anchors Only", method="update", args=[{"visible": [False] * n_traces + [True]}]),
    ]
    for i, cid in enumerate(unique_clusters):
        vis = [False] * (n_traces + 1)
        vis[i] = True
        vis[-1] = True  # keep anchors visible
        buttons.append(dict(label=f"Cluster {cid} + Anchors", method="update", args=[{"visible": vis}]))

    fig.update_layout(
        title=f"UMAP ({dim}D) — Clusters + Anchors",
        updatemenus=[dict(buttons=buttons, direction="down", x=1.02, y=0.95, showactive=True)],
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=True,
    )
    if dim == 2:
        fig.update_xaxes(title_text="UMAP-1")
        fig.update_yaxes(title_text="UMAP-2")
    else:
        fig.update_layout(scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"✅ Wrote HTML dashboard: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster insect crops (DINOv2 -> PCA -> KMeans) with optional HTML export.")
    p.add_argument("--image_dir", type=str, required=True, help="Directory containing crop images.")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    p.add_argument("--model", type=str, default="vit_base_patch14_dinov2", help="timm DINOv2 model name.")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu. Default: auto.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding extraction.")
    p.add_argument("--max_images", type=int, default=None, help="Process only first N images (for quick tests).")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)

    # PCA / clustering
    p.add_argument("--sil_pca", type=int, default=50, help="PCA dims used for silhouette k-scan.")
    p.add_argument("--final_pca", type=int, default=128, help="PCA dims used for final clustering & similarity.")
    p.add_argument("--pca_whiten", action="store_true", help="Enable PCA whitening.")
    p.add_argument("--k", type=int, default=None, help="Number of clusters. If omitted, chooses best by silhouette.")
    p.add_argument("--k_min", type=int, default=2, help="Min k for silhouette scan (inclusive).")
    p.add_argument("--k_max", type=int, default=30, help="Max k for silhouette scan (inclusive).")
    p.add_argument("--sil_sample", type=int, default=None, help="Silhouette sample size for speed (optional).")

    # Caching
    p.add_argument("--emb_cache_prefix", type=str, default=None,
                   help="Prefix for embedding cache (writes <prefix>.npy and <prefix>.paths.json). "
                        "If exists & matches current file list, loads instead of recomputing.")
    p.add_argument("--force_recompute", action="store_true", help="Ignore cache and recompute embeddings.")

    # Saving behavior
    p.add_argument("--copy_mode", choices=["copy", "symlink"], default="copy",
                   help="How to place images into cluster folders.")
    p.add_argument("--top_n_per_cluster", type=int, default=None,
                   help="If set, only save top-N most similar images per cluster (including anchor).")

    # Optional HTML
    p.add_argument("--export_html", action="store_true", help="Export an interactive UMAP HTML dashboard.")
    p.add_argument("--umap_dim", type=int, default=2, help="UMAP dimension for HTML (2 or 3).")
    p.add_argument("--umap_neighbors", type=int, default=20, help="UMAP n_neighbors.")
    p.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    p.add_argument("--html_max_points", type=int, default=15000, help="Max points in HTML (subsample if more).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    image_dir = Path(args.image_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    paths = list_images(image_dir, IMG_EXTS_DEFAULT)
    if args.max_images is not None:
        paths = paths[: args.max_images]

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {image_dir} (extensions: {IMG_EXTS_DEFAULT})")

    print(f"📂 Found {len(paths)} images in {image_dir}")

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Embedding cache
    cache_prefix = Path(args.emb_cache_prefix).expanduser().resolve() if args.emb_cache_prefix else None
    X = None
    if cache_prefix and not args.force_recompute:
        X = load_cache_if_valid(cache_prefix, paths)
        if X is not None:
            print(f"✅ Loaded cached embeddings: {cache_prefix.with_suffix('.npy')} -> {X.shape}")

    if X is None:
        print("🔎 Loading DINOv2 model...")
        model, preprocess = load_dinov2_timm(args.model, device)

        print("🧠 Extracting embeddings...")
        X = compute_embeddings_dinov2(paths, model, preprocess, device, batch_size=args.batch_size)

        if X.size == 0:
            raise RuntimeError("No embeddings extracted. Check images and dependencies.")

        if cache_prefix:
            save_cache(cache_prefix, paths, X)
            print(f"💾 Saved embedding cache: {cache_prefix.with_suffix('.npy')}")

    # PCA for silhouette scan
    X_sil, pca_sil = pca_reduce(X, args.sil_pca, args.pca_whiten, args.seed)

    # Choose k
    if args.k is None:
        print("📏 Picking k by silhouette...")
        best_k, sil_scores = pick_best_k_silhouette(
            X_sil, args.k_min, args.k_max, seed=args.seed, sample_size=args.sil_sample
        )
        k = best_k
        write_json(out_dir / "silhouette_scores.json", sil_scores)
        print(f"💾 Saved silhouette_scores.json")
    else:
        k = args.k
    print(f"🎯 Using k={k}")

    # PCA for clustering/similarity
    X_final, pca_final = pca_reduce(X, args.final_pca, args.pca_whiten, args.seed)
    write_json(out_dir / "pca_explained_variance.json", {
        "sil_pca": int(args.sil_pca),
        "final_pca": int(args.final_pca),
        "sil_explained_variance_sum": float(np.sum(pca_sil.explained_variance_ratio_)),
        "final_explained_variance_sum": float(np.sum(pca_final.explained_variance_ratio_)),
    })

    print("🧩 Clustering with KMeans...")
    km = KMeans(n_clusters=k, random_state=args.seed, n_init="auto")
    labels = km.fit_predict(X_final)

    # Anchors + ranking
    print("⭐ Selecting anchors...")
    anchors = select_anchors_mean_cosine(X_final, labels)  # {cid: idx}

    print("📌 Ranking images within each cluster by anchor similarity...")
    ranking = rank_within_cluster_by_anchor_sim(X_final, labels, anchors)  # {cid: [(idx, sim), ...]}

    # Save outputs
    clusters_root = out_dir / "clusters"
    clusters_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for cid, ranked in ranking.items():
        cluster_dir = clusters_root / f"cluster_{cid:03d}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Optionally truncate
        if args.top_n_per_cluster is not None:
            ranked = ranked[: args.top_n_per_cluster]

        for rank_pos, (idx, sim) in enumerate(ranked, start=1):
            src = paths[idx]
            is_anchor = (idx == anchors[cid])

            # Name: anchor_... or sim_... for others
            sim_str = f"{sim:.4f}"
            stem = src.stem
            ext = src.suffix.lower()

            if is_anchor:
                dst_name = f"anchor_{cid:03d}_{rank_pos:03d}_{stem}{ext}"
            else:
                dst_name = f"sim_{sim_str}_c{cid:03d}_{rank_pos:03d}_{stem}{ext}"

            dst = cluster_dir / dst_name
            safe_copy(src, dst, args.copy_mode)

            manifest_rows.append({
                "index": idx,
                "file": str(src),
                "file_name": src.name,
                "cluster": cid,
                "rank_in_cluster": rank_pos,
                "cosine_sim_to_anchor": sim,
                "is_anchor": bool(is_anchor),
                "saved_as": str(dst),
            })

    df_manifest = pd.DataFrame(manifest_rows).sort_values(["cluster", "rank_in_cluster"]).reset_index(drop=True)

    # Add PCA coords (handy for quick debug plots)
    df_manifest["pca1"] = X_final[df_manifest["index"].values, 0]
    df_manifest["pca2"] = X_final[df_manifest["index"].values, 1]
    if X_final.shape[1] >= 3:
        df_manifest["pca3"] = X_final[df_manifest["index"].values, 2]

    manifest_csv = out_dir / "cluster_manifest.csv"
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"💾 Wrote: {manifest_csv}")

    # Anchors CSV
    anchors_rows = []
    for cid in sorted(anchors.keys()):
        idx = anchors[cid]
        anchors_rows.append({
            "cluster": cid,
            "anchor_index": idx,
            "anchor_file": str(paths[idx]),
            "anchor_file_name": paths[idx].name,
        })
    df_anchors = pd.DataFrame(anchors_rows)
    anchors_csv = out_dir / "anchors.csv"
    df_anchors.to_csv(anchors_csv, index=False)
    print(f"💾 Wrote: {anchors_csv}")

    # Optional parquet (only if pyarrow is installed)
    try:
        parquet_path = out_dir / "cluster_manifest.parquet"
        df_manifest.to_parquet(parquet_path, index=False)
        print(f"💾 Wrote: {parquet_path}")
    except Exception:
        pass

    # Optional HTML export
    if args.export_html:
        html_path = out_dir / "umap_dashboard.html"
        try:
            export_umap_html(
                out_path=html_path,
                X_for_umap=X_final,
                labels=labels,
                anchors=anchors,
                paths=paths,
                dim=int(args.umap_dim),
                n_neighbors=int(args.umap_neighbors),
                min_dist=float(args.umap_min_dist),
                max_points=int(args.html_max_points),
                seed=int(args.seed),
            )
        except Exception as e:
            print(f"[warn] HTML export failed: {e}")

    print("✅ Done!")
    print(f"Clusters saved to: {clusters_root}")
    print(f"Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
