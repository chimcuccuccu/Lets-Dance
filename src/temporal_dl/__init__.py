"""
Person 2 — Temporal DL (nhịp/timing lệch so với reference).

API deliverable Tuần 5:

    from src.temporal_dl import load_temporal_checkpoint, temporal_model
    model, ckpt = load_temporal_checkpoint(
        "experiments/temporal_dl/temporal_khopnhip_dtw_official.pth")
    embedding, dtw_features = temporal_model(seq, model=model, dtw_row=row)
    # embedding: (64,) = mean ⊕ std trên embedding từng cửa sổ
    # dtw_features["vector"]: (4,) [total, seg_mean, seg_std, seg_max] ← fusion Tuần 6
"""
from src.temporal_dl.dataset import (
    DTW_INPUT_DIM,
    NODTW_INPUT_DIM,
    TemporalSampleMeta,
    TemporalSeqDataset,
    build_temporal_sequence,
    get_temporal_loaders,
    get_temporal_loaders_grouped,
    sorted_seg_columns,
    split_video_indices,
)
from src.temporal_dl.infer import (
    aggregate_windows,
    encode_video,
    encode_windows,
    export_temporal_embeddings,
    load_temporal_checkpoint,
    predict_video_score,
)
from src.temporal_dl.lstm import (
    AISTPPPretrainModel,
    TemporalBiLSTM,
    TemporalRegressionModel,
    load_pretrained_backbone,
    temporal_model,
)

__all__ = [
    # dataset
    "TemporalSeqDataset",
    "TemporalSampleMeta",
    "build_temporal_sequence",
    "sorted_seg_columns",
    "split_video_indices",
    "get_temporal_loaders",
    "get_temporal_loaders_grouped",
    "DTW_INPUT_DIM",
    "NODTW_INPUT_DIM",
    # model
    "TemporalBiLSTM",
    "TemporalRegressionModel",
    "AISTPPPretrainModel",
    "load_pretrained_backbone",
    "temporal_model",
    # infer
    "load_temporal_checkpoint",
    "encode_video",
    "encode_windows",
    "aggregate_windows",
    "predict_video_score",
    "export_temporal_embeddings",
]
