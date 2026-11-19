"""ComfyUI-LongCatInt8Glue

Lets Kijai's ComfyUI-WanVideoWrapper load Comfy-Org's LongCat-Video-Avatar 1.6 DiT
(`*_int8_convrot.safetensors`, int8 tensorwise + ConvRot Hadamard rotation) directly.

The wrapper has no ConvRot support. This pack wraps `comfy.utils.load_torch_file`: when the file
being loaded holds ConvRot int8 weights, each quantized tensor is dequantized with comfy-kitchen and
re-quantized to the fp8 e4m3fn layout the wrapper already understands (weight F8_E4M3 + weight_scale F32[1]
+ comfy_quant JSON), entirely in memory. No converted file is written or hosted. Any other file passes
through untouched. It has no nodes of its own; ComfyUI only needs to import it BEFORE
ComfyUI-WanVideoWrapper (custom nodes load alphabetically, and "ComfyUI-LongCat..." < "ComfyUI-WanVideo...").
"""
import json
import struct
import sys
import time

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_QMETA = json.dumps({"format": "float8_e4m3fn", "full_precision_matrix_mult": True}).encode()
_SKIP = (".weight_scale", ".comfy_quant", ".comfy_attention.config")


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def is_convrot_int8(path):
    """True when the safetensors file stores int8 ConvRot weights (cheap: header + one tiny tensor)."""
    try:
        h = _header(path)
        if not any(v.get("dtype") == "I8" for k, v in h.items() if k != "__metadata__"):
            return False
        from safetensors import safe_open
        with safe_open(path, "pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".comfy_quant"):
                    cfg = json.loads(bytes(f.get_tensor(k).tolist()))
                    return bool(cfg.get("convrot")) and cfg.get("format") == "int8_tensorwise"
    except Exception:
        return False
    return False


def convert_to_fp8_sd(path, device=None, compute_device=None, key_filter=None):
    """int8 ConvRot safetensors -> Kijai-style fp8 state dict (same math as tools/convert_longcat16_int8_to_fp8.py)."""
    import torch
    from safetensors import safe_open
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight

    out_dev = torch.device(device) if device is not None else torch.device("cpu")
    cdev = torch.device(compute_device) if compute_device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    qmeta = torch.tensor(list(_QMETA), dtype=torch.uint8)
    t0 = time.time()
    sd = {}
    nq = 0
    with safe_open(path, "pt", device="cpu") as f:
        keys = list(f.keys())
        kset = set(keys)
        for k in keys:
            if k.endswith(_SKIP) or (key_filter and not key_filter(k)):
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.int8 and (k + "_scale") in kset:
                base = k[:-len(".weight")]
                cfg = json.loads(bytes(f.get_tensor(base + ".comfy_quant").tolist()))
                assert cfg.get("convrot") and cfg["format"] == "int8_tensorwise", (k, cfg)
                scale = f.get_tensor(k + "_scale").float().to(cdev)
                w = dequantize_int8_convrot_weight(t.to(cdev), scale, cfg["convrot_groupsize"])
                if "modulation.1" in k:
                    sd[k] = w.to(torch.bfloat16).to(out_dev).contiguous()
                else:
                    s = (w.abs().max() / fp8_max).clamp(min=1e-12)
                    sd[k] = (w / s).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).to(out_dev).contiguous()
                    sd[base + ".weight_scale"] = s.reshape(1).float().to(out_dev)
                    sd[base + ".comfy_quant"] = qmeta.clone().to(out_dev)
                    nq += 1
                del w
            else:
                t = t.float() if k.startswith("patch_embedding") else t
                sd[k] = t.to(out_dev)
    print(f"[LongCatInt8Glue] converted {nq} ConvRot int8 tensors to fp8 in {time.time() - t0:.1f}s from {path}", flush=True)
    return sd


def _install():
    try:
        import comfy.utils as cu
    except Exception as e:  # not inside ComfyUI (unit tests)
        print(f"[LongCatInt8Glue] comfy.utils unavailable, not patching: {e}", flush=True)
        return
    orig = cu.load_torch_file

    def load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
        p = str(ckpt)
        if p.endswith(".safetensors") and is_convrot_int8(p):
            sd = convert_to_fp8_sd(p, device=device)
            meta = {"model_type": "LongCat-Avatar-single", "format": "pt"}
            return (sd, meta) if return_metadata else sd
        return orig(ckpt, safe_load=safe_load, device=device, return_metadata=return_metadata)

    cu.load_torch_file = load_torch_file
    patched = ["comfy.utils"]
    # If the wrapper was imported first, it already holds the original name: patch it too.
    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, "load_torch_file", None) is orig:
            mod.load_torch_file = load_torch_file
            patched.append(name)
    print(f"[LongCatInt8Glue] ConvRot int8 loader hook installed ({', '.join(patched)})", flush=True)


_install()
