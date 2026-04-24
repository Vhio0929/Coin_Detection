"""
检测硬币并裁切
模板图和待测图做几何对齐
输入 DINOv2，提 patch token / feature map
对应位置做 L2 / cosine distance
上采样成 anomaly map
阈值化得到缺陷区域
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os


class DinoFeatureDiffDetector:
    def __init__(
        self,
        model_name="dinov2_vits14",
        device=None,
        input_size=518,
        safe_inner_ratio=0.82,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.safe_inner_ratio = safe_inner_ratio

        # 官方常见加载方式
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)

        self.tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def build_safe_mask(self, size):
        center = size // 2
        yy, xx = np.ogrid[:size, :size]
        r = int(size * 0.5 * self.safe_inner_ratio)
        mask = (((xx - center) ** 2 + (yy - center) ** 2) <= r ** 2).astype(np.uint8)
        return mask

    def preprocess_pil(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.tf(pil).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def extract_patch_features(self, bgr_img):
        """
        对 DINOv2，不同版本接口可能略有不同。
        常用思路是取 patch tokens，再 reshape 成 feature map。
        """
        x = self.preprocess_pil(bgr_img)

        # 一种常见写法
        feats = self.model.forward_features(x)

        # 不同实现里 key 名可能不同，常见有 x_norm_patchtokens
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            patch_tokens = feats["x_norm_patchtokens"]  # [1, N, C]
        else:
            raise RuntimeError("Please inspect your DINOv2 forward_features output keys.")

        B, N, C = patch_tokens.shape
        side = int(np.sqrt(N))
        fmap = patch_tokens.reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()  # [1,C,H,W]
        return fmap

    def compute_feature_distance_map(self, template_bgr, test_bgr):
        ft_t = self.extract_patch_features(template_bgr)   # [1,C,h,w]
        ft_x = self.extract_patch_features(test_bgr)

        # L2 distance
        dist = torch.norm(ft_t - ft_x, dim=1, keepdim=True)   # [1,1,h,w]

        # 上采样回输入尺寸
        dist_up = F.interpolate(
            dist,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        score = dist_up.detach().cpu().numpy()
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)
        return score

    def threshold_map(self, score_map, safe_mask, k=2.2):
        valid = score_map[safe_mask > 0]
        thr = valid.mean() + k * valid.std()
        mask = (score_map >= thr).astype(np.uint8) * 255

        # 后处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask[safe_mask == 0] = 0
        return mask, float(thr)

    def overlay(self, gray_img, score_map, defect_mask):
        base = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        heat = (score_map * 255).clip(0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        out = cv2.addWeighted(base, 0.6, heat, 0.4, 0)

        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
        return out

    def scoremap_to_mask(self, score_map, k=2.0, min_thr=0.35):
        """
        score_map: float32, shape [H, W], 建议范围 0~1
        k: 均值+ k*std 的阈值系数
        min_thr: 给一个下限，避免阈值过低
        """
        score_map = score_map.astype(np.float32)

        mean_v = float(score_map.mean())
        std_v = float(score_map.std())
        thr = max(mean_v + k * std_v, min_thr)

        mask = (score_map >= thr).astype(np.uint8) * 255

        # 去噪 + 连通
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

        return mask, thr

    def extract_bboxes_from_mask(self, mask, min_area=40, max_area=20000, max_aspect_ratio=6.0):
        """
        从二值mask中提取候选缺陷框
        返回: boxes = [(x, y, w, h), ...]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > max_aspect_ratio:
                continue

            boxes.append((x, y, w, h))

        return boxes

    def draw_bboxes(self, image_bgr, boxes, color=(0, 0, 255), thickness=2):
        out = image_bgr.copy()
        for i, (x, y, w, h) in enumerate(boxes, 1):
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                out,
                f"defect_{i}",
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return out
    
    def detect_and_box_defects(self, test_img, score_map, k=5.0, min_thr=0.85,
                           min_area=150, max_area=20000, max_aspect_ratio=3.0, safe_mask=None, min_mean_score=1.0):
        mask, thr = self.scoremap_to_mask(score_map, k=k, min_thr=min_thr)
        
        # 应用安全掩码过滤边缘区域
        if safe_mask is not None:
            mask = mask * safe_mask
        
        # 提取边界框并过滤低分数区域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if w <= 0 or h <= 0:
                continue
            
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio > max_aspect_ratio:
                continue
            
            # 计算轮廓内的平均分数
            mask_roi = np.zeros_like(score_map, dtype=np.uint8)
            cv2.drawContours(mask_roi, [cnt], -1, 1, -1)
            mean_score = np.mean(score_map[mask_roi > 0])
            
            # 只保留平均分数高于阈值的区域
            if mean_score >= min_mean_score:
                boxes.append((x, y, w, h))
        
        vis = self.draw_bboxes(test_img, boxes)

        return {
            "mask": mask,
            "boxes": boxes,
            "defect_vis": vis,
            "threshold": thr,
        }
    
    def scoremap_to_mask_v2(self, score_map, k=2.3, min_thr=0.45, top_percent=0.02):
        """
        更严格的初筛：
        1. 全局阈值 mean + k*std
        2. 同时要求进入 top_percent 高分区域
        """
        score_map = score_map.astype(np.float32)

        mean_v = float(score_map.mean())
        std_v = float(score_map.std())
        thr1 = max(mean_v + k * std_v, min_thr)

        flat = score_map.reshape(-1)
        thr2 = np.quantile(flat, 1.0 - top_percent)

        thr = max(thr1, thr2)

        mask = (score_map >= thr).astype(np.uint8) * 255

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

        return mask, float(thr)


    def make_ring_mask(self, shape, center, r_inner, r_outer):
        h, w = shape
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        mask = ((dist2 >= r_inner ** 2) & (dist2 <= r_outer ** 2)).astype(np.uint8)
        return mask


    def component_stats(self, cnt):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        fill_ratio = area / (w * h + 1e-6)
        aspect = max(w / max(h, 1), h / max(w, 1))
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
        return {
            "area": area,
            "bbox": (x, y, w, h),
            "solidity": solidity,
            "fill_ratio": fill_ratio,
            "aspect": aspect,
            "circularity": circularity,
        }


    def local_contrast_score(self, score_map, bbox, pad=10):
        """
        看候选框相对周围环带是否真的突出
        """
        h, w = score_map.shape
        x, y, bw, bh = bbox

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)

        roi = score_map[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        outer_x1 = max(0, x - pad)
        outer_y1 = max(0, y - pad)
        outer_x2 = min(w, x + bw + pad)
        outer_y2 = min(h, y + bh + pad)

        outer = score_map[outer_y1:outer_y2, outer_x1:outer_x2]

        ring_mask = np.ones_like(outer, dtype=np.uint8)
        inner_x1 = x1 - outer_x1
        inner_y1 = y1 - outer_y1
        inner_x2 = x2 - outer_x1
        inner_y2 = y2 - outer_y1
        ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

        outer_vals = outer[ring_mask > 0]
        if outer_vals.size == 0:
            return 0.0

        return float(roi.mean() - outer_vals.mean())


    def extract_bboxes_from_mask_v2(
        self,
        score_map,
        mask,
        min_area=80,
        max_area=12000,
        max_aspect_ratio=3.5,
        min_solidity=0.55,
        min_fill_ratio=0.18,
        min_local_contrast=0.08,
    ):
        """
        比第一版更严格：
        1. 面积筛选
        2. 形状筛选
        3. 局部对比度筛选
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            st = self.component_stats(cnt)

            area = st["area"]
            x, y, w, h = st["bbox"]

            if area < min_area or area > max_area:
                continue
            if st["aspect"] > max_aspect_ratio:
                continue
            if st["solidity"] < min_solidity:
                continue
            if st["fill_ratio"] < min_fill_ratio:
                continue

            contrast = self.local_contrast_score(score_map, (x, y, w, h), pad=12)
            if contrast < min_local_contrast:
                continue

            boxes.append({
                "bbox": (x, y, w, h),
                "area": area,
                "solidity": st["solidity"],
                "fill_ratio": st["fill_ratio"],
                "contrast": contrast,
                "score_mean": float(score_map[y:y+h, x:x+w].mean()),
            })

        # 按局部突出度排序
        boxes = sorted(boxes, key=lambda z: (z["contrast"], z["score_mean"]), reverse=True)
        return boxes


    def draw_bboxes_v2(self, image_bgr, box_infos, color=(0, 0, 255), thickness=2):
        out = image_bgr.copy()
        for i, info in enumerate(box_infos, 1):
            x, y, w, h = info["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                out,
                f"defect_{i}",
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )
        return out


    def detect_and_box_defects_v2(
        self,
        test_img,
        score_map,
        k=2.5,
        min_thr=0.50,
        top_percent=0.01,
        min_area=120,
        max_area=10000,
        max_aspect_ratio=3.2,
        min_solidity=0.60,
        min_fill_ratio=0.22,
        min_local_contrast=0.10,
        safe_mask=None,
    ):
        mask, thr = self.scoremap_to_mask_v2(
            score_map,
            k=k,
            min_thr=min_thr,
            top_percent=top_percent,
        )

        if safe_mask is not None:
            mask = mask * safe_mask

        box_infos = self.extract_bboxes_from_mask_v2(
            score_map,
            mask,
            min_area=min_area,
            max_area=max_area,
            max_aspect_ratio=max_aspect_ratio,
            min_solidity=min_solidity,
            min_fill_ratio=min_fill_ratio,
            min_local_contrast=min_local_contrast,
        )

        vis = self.draw_bboxes_v2(test_img, box_infos)

        return {
            "mask": mask,
            "box_infos": box_infos,
            "defect_vis": vis,
            "threshold": thr,
        }
    
    def run(self, template_bgr, test_bgr):
        """
        这里默认 template_bgr 和 test_bgr 已经:
        1) 找到硬币
        2) 裁切成同尺度
        3) 做过旋转/平移配准
        你可以把前面的传统几何配准模块直接接进来。
        """
        template_bgr = cv2.resize(template_bgr, (self.input_size, self.input_size))
        test_bgr = cv2.resize(test_bgr, (self.input_size, self.input_size))

        safe_mask = self.build_safe_mask(self.input_size)

        score_map = self.compute_feature_distance_map(template_bgr, test_bgr)
        defect_mask, thr = self.threshold_map(score_map, safe_mask, k=2.2)

        gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
        vis = self.overlay(gray, score_map, defect_mask)
        results = self.detect_and_box_defects_v2(test_bgr, score_map, safe_mask=safe_mask)

        results = {
            **results,
            "gray": gray,
            'score_map': score_map,
            "vis": vis,
        }
        return results

        # return {
        #     "score_map": score_map,
        #     "defect_mask": defect_mask,
        #     "threshold": thr,
        #     "vis": vis,
        # }
    
if __name__ == "__main__":
    detector = DinoFeatureDiffDetector()
    template_path = r"D:\Code\gb\code\results\ori.jpg"   # 正常模板图
    test_path = r"D:\Code\gb\code\results\aligned_test.jpg"           # 待测图
    save_dir = "outputs_0423"
    os.makedirs(save_dir, exist_ok=True)

    template_bgr = cv2.imread(template_path)
    test_bgr = cv2.imread(test_path)

    results = detector.run(template_bgr, test_bgr)
    
    print(results.keys())
    defect_vis = results["defect_vis"]
    cv2.imwrite(os.path.join(save_dir, "defect_vis.jpg"), defect_vis)
    
    score_map = results["score_map"]
    cv2.imwrite(os.path.join(save_dir, "score_map.jpg"), score_map * 255)
    # defect_mask = results["defect_mask"]
    # cv2.imwrite(os.path.join(save_dir, "defect_mask.jpg"), defect_mask * 255)
    # thr = results["threshold"]
    # print(f"threshold: {thr}")
    
    vis = results["vis"]
    cv2.imwrite(os.path.join(save_dir, "vis.jpg"), vis)

    #print(results)
   
