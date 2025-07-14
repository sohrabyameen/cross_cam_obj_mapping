from ultralytics import YOLO
import torch
import os
import torchreid
from torchvision import transforms

from utils import first_vid, second_vid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = YOLO("best.pt")


first_video_path = "videos/tacticam.mp4"
results_1 = base_model.track(source=first_video_path, show=False, save=False, persist=True, tracker="botsort.yaml")

out_dir = "output_A"
os.makedirs(out_dir, exist_ok=True)

first_vid_obj = first_vid()
first_vid_obj.player_detection(results_1)


# Build the model
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,  # irrelevant here, we won’t use classifier
    pretrained=True
)

# Remove classifier (optional, but model won't use it if you skip softmax)
model.eval()
model.to(device)

# Image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),  # required shape for OsNet
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

player_gallery = first_vid_obj.embedding_extraction(model, transform)


second_video_path = "videos/broadcast.mp4"
results_2 = base_model.track(source=second_video_path, show=False, save=False, persist=True, tracker="botsort.yaml")

out_dir = "output_B"
os.makedirs(out_dir, exist_ok=True)

second_vid_obj = second_vid()
second_vid_obj.player_mapping(results_2,player_gallery, model, transform)




