import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image


class SimpleAnomalyCLIPStyle:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model.eval().to(self.device)

        # 更 object-agnostic 的提示词
        self.normal_prompts = [
            "a normal object surface",
            "a clean surface",
            "an intact surface",
            "a flawless surface",
        ]
        self.abnormal_prompts = [
            "an abnormal region",
            "a defect",
            "a damaged region",
            "a stain",
            "a scratch",
            "a dirty spot",
        ]

    @torch.no_grad()
    def encode_text(self, prompts):
        tokens = self.tokenizer(prompts).to(self.device)
        feat = self.model.encode_text(tokens)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def classify_image(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        img = self.preprocess(pil).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(img)
        img_feat = F.normalize(img_feat, dim=-1)

        normal_feat = self.encode_text(self.normal_prompts).mean(dim=0, keepdim=True)
        abnormal_feat = self.encode_text(self.abnormal_prompts).mean(dim=0, keepdim=True)

        s_normal = (img_feat @ normal_feat.T).item()
        s_abnormal = (img_feat @ abnormal_feat.T).item()

        return {
            "normal_score": s_normal,
            "abnormal_score": s_abnormal,
            "anomaly_score": s_abnormal - s_normal,
        }