from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm.auto import tqdm

import torch
import segmentation_models_pytorch as smp


DEFAULT_CFG = {
    "architecture": "Unet",
    "encoder_name": "resnet34",
    "band_ids": [1, 2, 3, 4, 5, 6],
    "band_names": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    "spectral_indices": ["MNDWI"],
    "input_features": ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "MNDWI"],
    "input_channels": 7,
    "patch_size": 256,
    "stride": 256,
    "threshold": 0.5,
}


def resolve_default_model_path():
    experiment_models = sorted(Path("experiments").glob("run_*/best_model.pth"))
    if experiment_models:
        return str(experiment_models[-1])
    return "weights/best_model_water.pth"


def resolve_input_image_path(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return input_path

    if input_path.is_dir():
        part5_matches = sorted(input_path.glob("*_img.tif"))
        s2_matches = [p for p in part5_matches if "_s2_" in p.name]
        if len(s2_matches) == 1:
            return s2_matches[0]
        if len(part5_matches) == 1:
            return part5_matches[0]

        tif_matches = sorted(input_path.glob("*.tif"))
        if len(tif_matches) == 1:
            return tif_matches[0]

        raise FileNotFoundError(
            f"No pude resolver una imagen unica dentro de la carpeta: {input_path}"
        )

    raise FileNotFoundError(f"No existe la ruta de entrada: {input_path}")


def infer_input_channels_from_checkpoint(state_dict):
    for key, value in state_dict.items():
        if key.endswith("encoder.conv1.weight"):
            return int(value.shape[1])
    raise KeyError("No se pudo inferir input_channels desde el checkpoint.")


def resolve_truth_mask_path(img_path):
    img_path = Path(img_path)

    if img_path.name.endswith("_img.tif"):
        mask_path = img_path.with_name(img_path.name.replace("_img.tif", "_msk.tif"))
        if mask_path.exists():
            return mask_path

    parent_name = img_path.parent.name
    if parent_name in {"tra_scene", "val_scene"}:
        mask_dir = img_path.parent.parent / parent_name.replace("_scene", "_truth")
        mask_name = img_path.name.replace("_6Bands_", "_").replace(".tif", "_Truth.tif")
        mask_path = mask_dir / mask_name
        if mask_path.exists():
            return mask_path

    return None


def build_positions(size, patch_size, stride):
    if size <= patch_size:
        return [0]

    pos = list(range(0, size - patch_size + 1, stride))
    last = size - patch_size
    if pos[-1] != last:
        pos.append(last)
    return pos


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = DEFAULT_CFG.copy()
    model_cfg.update(ckpt.get("cfg", {}))
    model_cfg["input_channels"] = infer_input_channels_from_checkpoint(ckpt["model_state_dict"])

    if "band_ids" not in model_cfg or not model_cfg["band_ids"]:
        if model_cfg["input_channels"] == 5:
            model_cfg["band_ids"] = [1, 2, 3, 4]
        elif model_cfg["input_channels"] == 7:
            model_cfg["band_ids"] = [1, 2, 3, 4, 5, 6]

    if "input_features" not in model_cfg or not model_cfg["input_features"]:
        if model_cfg["input_channels"] == 5:
            model_cfg["input_features"] = ["Blue", "Green", "Red", "NIR", "NDWI"]
            model_cfg["spectral_indices"] = ["NDWI"]
            model_cfg["band_names"] = ["Blue", "Green", "Red", "NIR"]
        elif model_cfg["input_channels"] == 7:
            model_cfg["input_features"] = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "MNDWI"]
            model_cfg["spectral_indices"] = ["MNDWI"]
            model_cfg["band_names"] = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

    model = smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=None,
        in_channels=model_cfg["input_channels"],
        classes=1,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_cfg


def preprocess_patch(img_patch, model_cfg):
    # Sentinel-2 suele venir en reflectancia escalada por 10000.
    img_patch = img_patch.astype(np.float32) / 10000.0

    input_channels = model_cfg["input_channels"]

    if input_channels == 5:
        blue = img_patch[0]
        green = img_patch[1]
        red = img_patch[2]
        nir = img_patch[3]

        # Compatibilidad con checkpoints viejos de 5 canales.
        ndwi = (green - nir) / (green + nir + 1e-6)
        return np.stack([blue, green, red, nir, ndwi], axis=0)

    if input_channels == 7:
        blue = img_patch[0]
        green = img_patch[1]
        red = img_patch[2]
        nir = img_patch[3]
        swir1 = img_patch[4]
        swir2 = img_patch[5]

        # MNDWI resalta mejor agua frente a suelo urbano y ruido costero.
        mndwi = (green - swir1) / (green + swir1 + 1e-6)
        return np.stack([blue, green, red, nir, swir1, swir2, mndwi], axis=0)

    if input_channels == img_patch.shape[0]:
        return img_patch

    raise ValueError(
        f"Configuración no soportada: input_channels={input_channels}, "
        f"bandas leídas={img_patch.shape[0]}."
    )


def predict_geotiff(
    img_path,
    model,
    device,
    model_cfg,
    patch_size=256,
    stride=256,
    threshold=0.5,
):
    img_path = Path(img_path)
    band_ids = model_cfg["band_ids"]

    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        rows = build_positions(height, patch_size, stride)
        cols = build_positions(width, patch_size, stride)

        prob_sum = np.zeros((height, width), dtype=np.float32)
        count_sum = np.zeros((height, width), dtype=np.float32)
        meta = src.meta.copy()

        if src.count < max(band_ids):
            raise ValueError(
                f"La imagen {img_path.name} solo tiene {src.count} bandas, "
                f"pero el modelo espera al menos {max(band_ids)}."
            )

        with torch.no_grad():
            for row in tqdm(rows, desc=f"infer {img_path.name}", leave=False):
                for col in cols:
                    window = Window(col_off=col, row_off=row, width=patch_size, height=patch_size)
                    img_patch = src.read(band_ids, window=window)
                    img_patch = preprocess_patch(img_patch, model_cfg)

                    x = torch.from_numpy(img_patch).unsqueeze(0).to(device)
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        logits = model(x)
                        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()

                    row_end = row + patch_size
                    col_end = col + patch_size
                    prob_sum[row:row_end, col:col_end] += probs
                    count_sum[row:row_end, col:col_end] += 1.0

    prob_map = prob_sum / np.clip(count_sum, a_min=1e-6, a_max=None)
    pred_mask = (prob_map >= threshold).astype(np.uint8)
    return prob_map, pred_mask, meta


def save_outputs(prob_map, pred_mask, meta, mask_out_path, prob_out_path=None):
    mask_out_path = Path(mask_out_path)
    mask_out_path.parent.mkdir(parents=True, exist_ok=True)

    mask_meta = meta.copy()
    mask_meta.update(count=1, dtype="uint8")
    with rasterio.open(mask_out_path, "w", **mask_meta) as dst:
        dst.write(pred_mask, 1)

    if prob_out_path is not None:
        prob_out_path = Path(prob_out_path)
        prob_out_path.parent.mkdir(parents=True, exist_ok=True)

        prob_meta = meta.copy()
        prob_meta.update(count=1, dtype="float32")
        with rasterio.open(prob_out_path, "w", **prob_meta) as dst:
            dst.write(prob_map.astype(np.float32), 1)


def percentile_stretch(rgb):
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)

    for i in range(rgb.shape[2]):
        band = rgb[:, :, i]
        p2, p98 = np.percentile(band, [2, 98])
        if p98 <= p2:
            out[:, :, i] = np.clip(band, 0.0, 1.0)
        else:
            out[:, :, i] = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)

    return out


def read_rgb_preview(img_path):
    with rasterio.open(img_path) as src:
        if src.count < 3:
            raise ValueError(f"No fue posible construir una vista RGB desde {img_path}")

        red = src.read(3).astype(np.float32) / 10000.0
        green = src.read(2).astype(np.float32) / 10000.0
        blue = src.read(1).astype(np.float32) / 10000.0
        rgb = np.stack([red, green, blue], axis=-1)
        return percentile_stretch(rgb)


def build_error_overlay(rgb, pred_mask, true_mask):
    overlay = rgb.copy()
    fp = (pred_mask == 1) & (true_mask == 0)
    fn = (pred_mask == 0) & (true_mask == 1)

    # Rojo para falsos positivos y azul para falsos negativos.
    overlay[fp] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    overlay[fn] = np.array([0.0, 0.3, 1.0], dtype=np.float32)
    return overlay, fp, fn


def save_preview_figure(img_path, prob_map, pred_mask, preview_out_path, true_mask_path=None, show=False):
    rgb = read_rgb_preview(img_path)
    true_mask = None

    if true_mask_path is not None and Path(true_mask_path).exists():
        with rasterio.open(true_mask_path) as src:
            true_mask = (src.read(1) > 0).astype(np.uint8)
        if true_mask.shape != pred_mask.shape:
            true_mask = None

    if true_mask is None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[1].imshow(prob_map, cmap="Blues")
        axes[1].set_title("Probabilidad agua")
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Prediccion")
        axes[3].imshow(rgb)
        axes[3].imshow(pred_mask, cmap="autumn", alpha=0.45)
        axes[3].set_title("Prediccion sobre RGB")
    else:
        error_overlay, fp, fn = build_error_overlay(rgb, pred_mask, true_mask)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[1].imshow(true_mask, cmap="gray")
        axes[1].set_title("Mascara real")
        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Prediccion")
        axes[3].imshow(prob_map, cmap="Blues")
        axes[3].set_title("Probabilidad agua")
        axes[4].imshow(error_overlay)
        axes[4].set_title("Errores: FP rojo / FN azul")
        axes[5].imshow(rgb)
        axes[5].imshow(fp, cmap="Reds", alpha=0.55)
        axes[5].imshow(fn, cmap="cool", alpha=0.40)
        axes[5].set_title("Overlay de errores")

    for ax in np.array(axes).ravel():
        ax.axis("off")

    fig.suptitle(Path(img_path).name, fontsize=14)
    fig.tight_layout()

    preview_out_path = Path(preview_out_path)
    preview_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_out_path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_run_info(info_path, img_path, model_path, cfg, device, patch_size, stride, threshold):
    info = {
        "input_image": str(Path(img_path).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "device": str(device),
        "input_format": {
            "source_type": "GeoTIFF",
            "raw_bands_expected": cfg.get("band_names", []),
            "raw_band_order": cfg["band_ids"],
            "derived_index": ", ".join(cfg.get("spectral_indices", [])),
            "model_features": cfg.get("input_features", []),
            "normalization": "reflectance / 10000.0",
            "patch_size": patch_size,
            "stride": stride,
        },
        "output_format": {
            "mask_type": "GeoTIFF uint8",
            "mask_values": {"0": "no water", "1": "water"},
            "probability_type": "GeoTIFF float32",
            "threshold": threshold,
        },
        "model": {
            "architecture": cfg["architecture"],
            "encoder_name": cfg["encoder_name"],
            "encoder_weights": None,
            "input_channels": cfg["input_channels"],
        },
    }

    info_path = Path(info_path)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia para segmentación de agua en Sentinel-2. "
            "Entrada esperada: GeoTIFF con bandas Blue, Green, Red, NIR, SWIR1 y SWIR2. "
            "El script calcula MNDWI internamente y reconstruye la máscara por ventanas 256x256."
        )
    )
    parser.add_argument("--input", default=None, help="Ruta del GeoTIFF de entrada.")
    parser.add_argument("--model", default=resolve_default_model_path(), help="Ruta del checkpoint .pth.")
    parser.add_argument("--mask", default=None, help="Ruta opcional de la mascara real para comparar.")
    parser.add_argument("--mask-out", default=None, help="Ruta de salida para la máscara binaria.")
    parser.add_argument("--prob-out", default=None, help="Ruta opcional de salida para el mapa de probabilidad.")
    parser.add_argument("--info-out", default=None, help="Ruta opcional para guardar un JSON con el formato de entrada/salida.")
    parser.add_argument("--preview-out", default=None, help="Ruta opcional para guardar un panel PNG con la visualizacion.")
    parser.add_argument("--patch-size", type=int, default=None, help="Tamaño del parche. Por defecto usa el del checkpoint o 256.")
    parser.add_argument("--stride", type=int, default=None, help="Stride de inferencia. Por defecto usa el del checkpoint o 256.")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral para binarizar la máscara.")
    parser.add_argument("--show", action="store_true", help="Muestra la figura al terminar la inferencia.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.input is None:
        user_input = input("Ruta del GeoTIFF de entrada: ").strip()
        if not user_input:
            raise ValueError("Debes indicar una ruta de entrada para continuar.")
        args.input = user_input

    img_path = resolve_input_image_path(args.input)
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint del modelo: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(model_path, device)

    patch_size = args.patch_size or cfg.get("patch_size", DEFAULT_CFG["patch_size"])
    stride = args.stride or cfg.get("stride", cfg.get("val_stride", DEFAULT_CFG["stride"]))
    threshold = args.threshold if args.threshold is not None else cfg.get("threshold", DEFAULT_CFG["threshold"])

    default_dir = model_path.parent / "inference_outputs"
    default_stem = img_path.stem

    mask_out = Path(args.mask_out) if args.mask_out else default_dir / f"{default_stem}_pred_mask.tif"
    prob_out = Path(args.prob_out) if args.prob_out else default_dir / f"{default_stem}_prob_map.tif"
    info_out = Path(args.info_out) if args.info_out else default_dir / f"{default_stem}_inference_info.json"
    preview_out = Path(args.preview_out) if args.preview_out else default_dir / f"{default_stem}_preview.png"

    prob_map, pred_mask, meta = predict_geotiff(
        img_path=img_path,
        model=model,
        device=device,
        model_cfg=cfg,
        patch_size=patch_size,
        stride=stride,
        threshold=threshold,
    )

    save_outputs(prob_map, pred_mask, meta, mask_out, prob_out)
    save_run_info(info_out, img_path, model_path, cfg, device, patch_size, stride, threshold)

    true_mask_path = Path(args.mask) if args.mask else resolve_truth_mask_path(img_path)
    save_preview_figure(
        img_path=img_path,
        prob_map=prob_map,
        pred_mask=pred_mask,
        preview_out_path=preview_out,
        true_mask_path=true_mask_path,
        show=args.show,
    )

    print(f"device: {device}")
    print(f"mask_out: {mask_out}")
    print(f"prob_out: {prob_out}")
    print(f"info_out: {info_out}")
    print(f"preview_out: {preview_out}")
    if true_mask_path is not None:
        print(f"true_mask: {true_mask_path}")


if __name__ == "__main__":
    main()
