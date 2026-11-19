# ComfyUI-LongCatInt8Glue

Lets Kijai's [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) load Comfy-Org's
[LongCat-Video-Avatar 1.6](https://huggingface.co/Comfy-Org/LongCat-Video-Avatar) `*_int8_convrot.safetensors` directly.

It wraps `comfy.utils.load_torch_file`: ConvRot int8 weights are dequantized with comfy-kitchen and re-quantized to fp8 e4m3fn
(the layout the wrapper reads) in memory. Nothing is written to disk, other files pass through untouched.
Stopgap until native LongCat Avatar support lands in ComfyUI (Comfy-Org/ComfyUI#17831). No nodes; just install it.
Needs comfy-kitchen with `dequantize_int8_convrot_weight` (>= 0.2.41) and must import before the wrapper (alphabetical order does this).
