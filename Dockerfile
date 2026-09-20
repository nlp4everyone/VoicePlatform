FROM vllm/vllm-omni:v0.26.0

# VoxCPM2 talker in vllm-omni imports `voxcpm.core.VoxCPM`, which the base
# image does not ship (see vllm_omni/.../voxcpm2_import_utils.py).
RUN pip install --no-cache-dir "voxcpm>=2.0" \
    && python -c "from voxcpm.core import VoxCPM"
