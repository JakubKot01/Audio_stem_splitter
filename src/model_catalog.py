from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    backend: str
    architecture: str
    purpose: str


MODEL_SPECS = (
    ModelSpec("htdemucs", "demucs", "Demucs", "General 4-stem separation"),
    ModelSpec("htdemucs_ft", "demucs", "Demucs", "Fine-tuned 4-stem separation"),
    ModelSpec("htdemucs_6s", "demucs", "Demucs", "6 stems including guitar and piano"),
    ModelSpec(
        "vocals_mel_band_roformer.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Vocal specialist; highest listed vocal SDR",
    ),
    ModelSpec(
        "melband_roformer_big_beta4.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Large fine-tuned vocal specialist",
    ),
    ModelSpec(
        "mel_band_roformer_kim_ft_unwa.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Fine-tuned general vocal specialist",
    ),
    ModelSpec(
        "melband_roformer_big_beta5e.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Large fine-tuned vocal specialist",
    ),
    ModelSpec(
        "MelBandRoformerBigSYHFTV1.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Alternative large vocal fine-tune",
    ),
    ModelSpec(
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "audio-separator",
        "MDXC / BS-Roformer",
        "Balanced 2-stem model with stronger vocals",
    ),
    ModelSpec(
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "audio-separator",
        "MDXC / BS-Roformer",
        "Balanced 2-stem model with stronger instrumental",
    ),
    ModelSpec(
        "melband_roformer_instvoc_duality_v1.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Balanced vocal and instrumental separation",
    ),
    ModelSpec(
        "melband_roformer_instvox_duality_v2.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Revised balanced vocal and instrumental separation",
    ),
    ModelSpec(
        "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        "audio-separator",
        "MDXC / MDX23C",
        "High-quality balanced vocal and instrumental separation",
    ),
    ModelSpec(
        "MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
        "audio-separator",
        "MDXC / MDX23C",
        "Alternative high-quality balanced checkpoint",
    ),
    ModelSpec(
        "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "General vocal and instrumental separation",
    ),
    ModelSpec(
        "melband_roformer_inst_v2.ckpt",
        "audio-separator",
        "MDXC / MelBand Roformer",
        "Instrumental specialist for vocal removal",
    ),
    ModelSpec(
        "UVR-MDX-NET_Main_406.onnx",
        "audio-separator",
        "MDX-Net",
        "Classic MDX vocal specialist and lightweight baseline",
    ),
    ModelSpec(
        "UVR_MDXNET_Main.onnx",
        "audio-separator",
        "MDX-Net",
        "Classic general MDX baseline",
    ),
    ModelSpec(
        "UVR-MDX-NET-Voc_FT.onnx",
        "audio-separator",
        "MDX-Net",
        "Classic vocal-fine-tuned MDX model",
    ),
    ModelSpec(
        "Kim_Vocal_2.onnx",
        "audio-separator",
        "MDX-Net",
        "Classic Kim vocal specialist",
    ),
    ModelSpec(
        "UVR-MDX-NET_Main_340.onnx",
        "audio-separator",
        "MDX-Net",
        "Classic MDX checkpoint variant",
    ),
    ModelSpec(
        "UVR-MDX-NET_Main_427.onnx",
        "audio-separator",
        "MDX-Net",
        "Classic MDX checkpoint variant",
    ),
)

MODEL_BY_FILENAME = {model.filename: model for model in MODEL_SPECS}
MODEL_FILENAMES = tuple(model.filename for model in MODEL_SPECS)


def get_model_spec(filename: str) -> ModelSpec:
    try:
        return MODEL_BY_FILENAME[filename]
    except KeyError as error:
        raise ValueError(f"Unsupported model: {filename}") from error
