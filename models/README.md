# Local learned-perception weights

Downloaded from the official Ultralytics and Google model repositories. Exact URLs,
file sizes and SHA-256 hashes are recorded in `manifest.json`; the binary files are
excluded from Git. These are YOLO11 Nano segmentation and MediaPipe Pose Landmarker Lite.

Launch from the repository root:

```sh
source .venv/bin/activate
python -m robopod app --yolo-model models/yolo11n-seg.pt --pose-model models/pose_landmarker_lite.task
```

Click **Explore scene** to build a new map with both backends. The Studio settings
panel shows the selected paths; live status reports detections, poses, associations
and inference time. Simulator segmentation remains the fallback.

Run with the normal macOS desktop graphics session: MediaPipe's macOS native graph
initializes graphics services even when CPU inference is used. A restricted graphics
sandbox may abort initialization.


Verification on this machine: both backends processed all 14 retained MuJoCo camera
views without adapter errors. Warm combined perception took roughly 110–310 ms per
frame. YOLO sometimes confused the stylised plant with a sports ball; MediaPipe found
a pose in one view. These pretrained models target photographic imagery, so fallback
segmentation remains important for this demo. Raw results are saved in
`outputs/learned-perception-check/inference_report.json`; tested package versions are
recorded in `installed-versions.json`.
