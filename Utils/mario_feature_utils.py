from pathlib import Path

import dgl
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FUSION_CHOICES = ("concat", "mean", "sum", "text", "image")


def resolve_repo_path(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_graph_num_nodes(graph_path):
    graph = dgl.load_graphs(str(resolve_repo_path(graph_path)))[0][0]
    return graph.num_nodes()


def normalize_feature_array(array, feature_name):
    feature = np.asarray(array, dtype=np.float32)
    if feature.ndim == 2:
        return feature
    if feature.ndim == 3:
        reshaped = feature.reshape(-1, feature.shape[-1])
        print(f"{feature_name}: reshaped {tuple(feature.shape)} -> {tuple(reshaped.shape)}")
        return reshaped
    raise ValueError(
        f"{feature_name} must be a 2D or 3D numpy array, but got shape {tuple(feature.shape)}."
    )


def load_feature_matrix(feature_path, feature_name):
    resolved_path = resolve_repo_path(feature_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"{feature_name} not found: {resolved_path}")
    feature = np.load(resolved_path, mmap_mode="r")
    return normalize_feature_array(feature, feature_name), resolved_path


def fuse_features(text_feature, image_feature, fusion):
    if fusion == "text":
        return text_feature
    if fusion == "image":
        return image_feature
    if text_feature.shape != image_feature.shape:
        raise ValueError(
            "Text and image features must have the same shape for this fusion mode, "
            f"but got {text_feature.shape} and {image_feature.shape}."
        )
    if fusion == "concat":
        return np.concatenate([text_feature, image_feature], axis=1)
    if fusion == "mean":
        return (text_feature + image_feature) / 2.0
    if fusion == "sum":
        return text_feature + image_feature
    raise ValueError(f"Unsupported fusion mode: {fusion}")


def infer_dataset_name(graph_path):
    return resolve_repo_path(graph_path).parent.name


def infer_prepared_feature_path(graph_path, fusion):
    dataset_name = infer_dataset_name(graph_path)
    return resolve_repo_path(
        Path("Data") / dataset_name / "MMFeature" / f"{dataset_name}_MARIO_{fusion}.npy"
    )


def prepare_feature_matrix(
    graph_path,
    fusion="concat",
    feature_path=None,
    text_feature_path=None,
    image_feature_path=None,
    prepared_feature_path=None,
    save_prepared=True,
):
    if fusion not in FUSION_CHOICES:
        raise ValueError(f"fusion must be one of {FUSION_CHOICES}, but got {fusion}.")

    num_nodes = load_graph_num_nodes(graph_path)

    if feature_path is not None:
        feature, source_path = load_feature_matrix(feature_path, "feature_path")
        output_path = resolve_repo_path(prepared_feature_path) if prepared_feature_path else source_path
    else:
        text_feature = image_feature = None
        if fusion != "image":
            if text_feature_path is None:
                raise ValueError("text_feature_path is required unless fusion='image' or feature_path is given.")
            text_feature, _ = load_feature_matrix(text_feature_path, "text_feature_path")
        if fusion != "text":
            if image_feature_path is None:
                raise ValueError("image_feature_path is required unless fusion='text' or feature_path is given.")
            image_feature, _ = load_feature_matrix(image_feature_path, "image_feature_path")
        feature = fuse_features(text_feature, image_feature, fusion)
        output_path = (
            resolve_repo_path(prepared_feature_path)
            if prepared_feature_path
            else infer_prepared_feature_path(graph_path, fusion)
        )

    if feature.shape[0] != num_nodes:
        raise ValueError(
            f"Feature rows ({feature.shape[0]}) do not match graph nodes ({num_nodes}). "
            "If FeatureExtractor.py used batching, make sure no samples were dropped."
        )

    if save_prepared:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, feature.astype(np.float32))
        print(f"Prepared feature saved to: {output_path}")

    return feature.astype(np.float32), output_path
