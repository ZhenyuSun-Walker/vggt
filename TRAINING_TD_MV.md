# TD/MV VGGT training

The 8-GPU training configuration is [`training/config/td_fusion_8a100.yaml`](training/config/td_fusion_8a100.yaml).
It uses four perspective frames plus one top-down conditioning image per sample.
The first sorted perspective frame is always the reference frame; the TD image is appended last.

Expected paths, relative to this repository:

```text
data/
  TD_HM3D/
    00000-scene_name/
      rgb/00000000_1.jpeg
      depth/00000000_1_depth.exr
      camera/00000000_1_camera_params.json
      rgb_td/rgb_td_mirror.png   # rgb_td.png also supported
  FP_Stru3D/
    stru3d_00000/
      rgb/0.png
      depth/0.png                # 16-bit millimetres
      pose/0.txt
      rgb_td/rgb_td_mirror.png   # rgb_td.png also supported
```

`VGGT` is cloned from the existing `Pi3` conda environment.  To reproduce the
environment on a new machine, run:

```bash
bash scripts/create_vggt_env.sh
```

All pip installs in that script use `-i https://pypi.org/simple/`.  Start the
prepared 8-A100 job with:

```bash
bash scripts/train_td_fusion_8a100.sh
```

The launcher activates `VGGT`, requires eight local processes, and rejects a
host that does not expose at least eight A100 GPUs.  To change ordinary Hydra
options, append them to the command, for example `max_epochs=10`.

The TD image carries dummy geometry with an all-false supervision mask.  It is
therefore available to global attention as context but contributes neither pose
nor depth supervision.  Its token suffix receives dedicated QKV and FFN
parameters in every global attention block; frame-attention parameters remain
shared.
