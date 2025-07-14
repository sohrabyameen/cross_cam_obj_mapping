Player Re-Identification Across Multi-Camera Football Videos

This project focuses on tracking football players across two different camera angles using a fine-tuned object detection model and a deep ReID embedding-based similarity matching system. It assigns consistent player IDs across both videos, even when players appear from different perspectives.

To assign each football player a consistent and unique ID across two separate match videos recorded from different camera angles. This ID should remain the same even if the player:

Moves in and out of the frame,
Appears from different perspectives,
Changes apparent size due to zoom or camera position.

Methodology

1. Object Detection & Tracking
A fine-tuned YOLOv11 model is used to detect and track players in both videos.

Tracking is done using BoT-SORT, which generates temporary local IDs within each video.

2. Feature Extraction with ReID
A deep re-identification model (osnet_x1_0 from Torchreid) extracts robust visual embeddings for each detected player.

For Video A, multiple crops per player are averaged to create stable identity embeddings.

3. Cross-Video ID Matching (Video B)
For each player in Video B, the model compares its embedding to players in Video A using cosine similarity.

A global ID is assigned if similarity > 0.75; otherwise, a new ID is created.

Output
After running the script:

Two new folders output_A/ and output_B/ will be created

Each contains an annotated .avi video with bounding boxes and global player IDs

How to Run
Place both videos (tacticam.mp4, broadcast.mp4) inside the videos/ directory.

Make sure the pretrained YOLO model "best.pt" is in the main directory.

Install dependencies from requirements.txt

Dependencies

Python ≥ 3.12
Ultralytics YOLOv8
TorchReID
OpenCV
PyTorch
Scikit-learn

You may need a CUDA-compatible GPU for faster inference (optional but recommended)


This uses cosine similarity only, no histogram or jersey color features.

Designed for low-resolution (720p) video with barely visible facial features.

Works best when the player movements are visible for at least a few frames.

