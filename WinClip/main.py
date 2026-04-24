import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from torchvision import transforms


class SimpleWinCLIP:
    def __init__(self, device=None, input_size=448):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        # 可换成你本地可用的 CLIP backbone
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model.eval().to(self.device)

        self.normal_prompts = [
            "a photo of a normal coin",
            "a clean coin",
            "an undamaged coin surface",
            "a flawless metallic coin",
        ]
        self.abnormal_prompts = [
            "a defective coin",
            "a dirty coin",
            "a stained coin",
            "a scratched coin",
            "a damaged coin surface",
        ]

    @torch.no_grad()
    def encode_texts(self, prompts):
        tokens = self.tokenizer(prompts).to(self.device)
        text_feat = self.model.encode_text(tokens)
        text_feat = F.normalize(text_feat, dim=-1)
        return text_feat

    @torch.no_grad()
    def encode_image_global(self, pil_img):
        img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img)
        feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def sliding_window_score_map(self, bgr_img, win=96, stride=32):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        H, W, _ = rgb.shape

        normal_text = self.encode_texts(self.normal_prompts).mean(dim=0, keepdim=True)
        abnormal_text = self.encode_texts(self.abnormal_prompts).mean(dim=0, keepdim=True)

        score_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        for y in range(0, H - win + 1, stride):
            for x in range(0, W - win + 1, stride):
                crop = rgb[y:y + win, x:x + win]
                pil = Image.fromarray(crop)
                img_feat = self.encode_image_global(pil)

                s_normal = (img_feat @ normal_text.T).item()
                s_abnormal = (img_feat @ abnormal_text.T).item()

                # abnormal 相对更高时，分数更大
                score = s_abnormal - s_normal

                score_map[y:y + win, x:x + win] += score
                count_map[y:y + win, x:x + win] += 1.0

        score_map /= (count_map + 1e-8)
        score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)
        return score_map

    def threshold(self, score_map, k=2.0):
        thr = score_map.mean() + k * score_map.std()
        mask = (score_map >= thr).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, float(thr)