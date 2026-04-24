import os
import cv2
import numpy as np


class CoinAnomalyDetectorV3:
    """
    更偏“黑点/污渍”检测的版本
    改进重点：
    1. 只检测比模板更暗的残差（dark residual）
    2. 用模板边缘图抑制结构性误报
    3. 用 black-hat 强化小黑点/污渍
    4. 连通域按形状过滤，去掉长条边缘误报
    """

    def __init__(
        self,
        output_size=512,
        crop_margin=1.15,
        coarse_angle_step=10,
        use_clahe=True,
        binary_threshold=40,
        outer_mask_ratio=0.85,
        safe_inner_ratio=0.80,
        highlight_thresh=240,
    ):
        self.output_size = output_size
        self.crop_margin = crop_margin
        self.coarse_angle_step = coarse_angle_step
        self.use_clahe = use_clahe
        self.binary_threshold = binary_threshold
        self.outer_mask_ratio = outer_mask_ratio
        self.safe_inner_ratio = safe_inner_ratio
        self.highlight_thresh = highlight_thresh

    @staticmethod
    def read_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    @staticmethod
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rotate_image(image, angle_deg):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        return cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

    @staticmethod
    def normalize_to_uint8(x):
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - mn) / (mx - mn)
        return (y * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def draw_circle(img, circle, color=(0, 255, 0), thickness=2):
        x, y, r = circle
        out = img.copy()
        cv2.circle(out, (int(x), int(y)), int(r), color, thickness)
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
        return out

    def preprocess_gray(self, gray):
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        return gray

    def illumination_normalize(self, gray):
        gray_f = gray.astype(np.float32) + 1.0
        bg = cv2.GaussianBlur(gray_f, (0, 0), 25)
        norm = gray_f / (bg + 1e-6)
        norm = norm / (norm.max() + 1e-6)
        return (norm * 255).clip(0, 255).astype(np.uint8)

    def extract_circular(self, img):

        # 创建保存目录
        #save_dir = os.path.dirname(save_path)
        

        if img is None:
            print("传入图片不存在")
            return None, None

        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        #cv2.imwrite(os.path.splitext(save_path)[0] + "_edges.jpg", edges)
        print("边缘检测完成")

        # 检测轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 找到最大的轮廓（最外圈）
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            # 创建掩码
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)

            # 裁剪圆形区域
            result = cv2.bitwise_and(img, img, mask=mask)
            #cv2.imwrite(save_path, result)
            print("圆形区域裁剪完成")
            return result, (center[0], center[1], radius)
        else:
            print("未检测到轮廓")
            return None, None

    def detect_coin_circle(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("No contour found for coin.")

        h, w = gray.shape
        image_area = h * w
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * image_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            (x, y), r = cv2.minEnclosingCircle(cnt)
            if circularity > 0.7 and 50 < r < min(h, w) * 0.45:
                candidates.append((cnt, area, circularity, x, y, r))

        if not candidates:
            cnt = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(cnt)
            return int(x), int(y), int(r)

        cnt, area, circularity, x, y, r = max(candidates, key=lambda z: z[1])
        return int(x), int(y), int(r)

    def build_coin_masks(self, size, outer_ratio=0.85, inner_ratio=0.80):
        center = size // 2
        yy, xx = np.ogrid[:size, :size]
        outer_r = int(size * 0.5 * outer_ratio)
        inner_r = int(outer_r * inner_ratio)

        coin_mask = (((xx - center) ** 2 + (yy - center) ** 2) <= outer_r ** 2).astype(np.uint8) * 255
        safe_mask = (((xx - center) ** 2 + (yy - center) ** 2) <= inner_r ** 2).astype(np.uint8) * 255
        return coin_mask, safe_mask

    def crop_and_normalize_coin(self, image_bgr, circle):
        x, y, r = circle
        margin_r = int(r * self.crop_margin)

        x1 = max(0, x - margin_r)
        y1 = max(0, y - margin_r)
        x2 = min(image_bgr.shape[1], x + margin_r)
        y2 = min(image_bgr.shape[0], y + margin_r)

        crop = image_bgr[y1:y2, x1:x2].copy()
        crop = cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

        coin_mask, safe_mask = self.build_coin_masks(
            self.output_size,
            outer_ratio=self.outer_mask_ratio,
            inner_ratio=self.safe_inner_ratio
        )
        return crop, coin_mask, safe_mask

    def coarse_rotation_search(self, template_gray, test_gray, mask):
        valid = mask > 0
        best_angle = 0
        best_score = float("inf")
        best_rot = test_gray.copy()

        for angle in range(0, 360, self.coarse_angle_step):
            rot = self.rotate_image(test_gray, angle)
            diff = (template_gray.astype(np.float32) - rot.astype(np.float32))[valid]
            score = np.mean(diff ** 2)
            if score < best_score:
                best_score = score
                best_angle = angle
                best_rot = rot

        return best_angle, best_rot

    def ecc_align(self, template_gray, input_gray, mask):
        template_f = template_gray.astype(np.float32) / 255.0
        input_f = input_gray.astype(np.float32) / 255.0
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                template_f,
                input_f,
                warp_matrix,
                motionType=cv2.MOTION_EUCLIDEAN,
                criteria=criteria,
                inputMask=mask,
                gaussFiltSize=5
            )
            aligned = cv2.warpAffine(
                input_gray,
                warp_matrix,
                (template_gray.shape[1], template_gray.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            return aligned, warp_matrix, cc
        except cv2.error:
            return input_gray.copy(), np.eye(2, 3, dtype=np.float32), None

    def build_highlight_mask(self, gray, thresh=240):
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[gray >= thresh] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.dilate(mask, k, iterations=1)

    def build_structure_mask(self, template_gray, coin_mask):
        """
        用模板自身提取“原有结构边缘”，这些地方天然容易误报
        """
        edges = cv2.Canny(template_gray, 60, 140)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.dilate(edges, k, iterations=2)
        edges[coin_mask == 0] = 0
        return edges

    def build_dark_blob_map(self, template_gray, aligned_gray, coin_mask, safe_mask):
        """
        只关注“比模板更暗”的局部污点
        """
        t = template_gray.astype(np.float32)
        a = aligned_gray.astype(np.float32)

        # 1) 只取 darker residual
        dark_residual = np.maximum(t - a, 0)

        # 2) black-hat 强化小黑污点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        blackhat = cv2.morphologyEx(aligned_gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

        # 3) LoG 风格的 blob 响应
        blur1 = cv2.GaussianBlur(aligned_gray, (0, 0), 1.2).astype(np.float32)
        blur2 = cv2.GaussianBlur(aligned_gray, (0, 0), 3.0).astype(np.float32)
        dog = np.maximum(blur2 - blur1, 0)

        # 归一化
        dark_n = dark_residual / (dark_residual.max() + 1e-6)
        blackhat_n = blackhat / (blackhat.max() + 1e-6)
        dog_n = dog / (dog.max() + 1e-6)

        # 以 dark/blob 为主，不强调边缘
        score = 0.50 * dark_n + 0.35 * blackhat_n + 0.15 * dog_n

        # 安全区外降权
        ring_mask = ((coin_mask > 0) & (safe_mask == 0))
        score[ring_mask] *= 0.15

        # 高光区降权
        highlight_mask = self.build_highlight_mask(aligned_gray, self.highlight_thresh)
        score[highlight_mask > 0] *= 0.2

        # 模板强结构边缘降权
        structure_mask = self.build_structure_mask(template_gray, coin_mask)
        score[structure_mask > 0] *= 0.15

        score[coin_mask == 0] = 0
        score = cv2.GaussianBlur(score, (5, 5), 0)

        return score, dark_residual, blackhat, dog, structure_mask, highlight_mask

    def filter_components_by_shape(self, binary_mask):
        """
        只保留更像“污点/斑块”的连通域
        去掉长条边缘、弧线、细碎噪声
        """
        out = np.zeros_like(binary_mask)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            if area > 1200:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w == 0 or h == 0:
                continue

            aspect = max(w, h) / max(1, min(w, h))
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            fill_ratio = area / (w * h + 1e-6)

            # 去掉细长边缘型
            if aspect > 4.5 and area < 250:
                continue

            # 太空、太散的不要
            if solidity < 0.45:
                continue

            # 面积较小的时候要求更像 blob
            if area < 80:
                if circularity < 0.08 and fill_ratio < 0.18:
                    continue

            cv2.drawContours(out, [cnt], -1, 255, -1)

        return out

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

    def segment_defect(self, score_map, coin_mask, safe_mask):
        score_u8 = self.normalize_to_uint8(score_map)

        valid_vals = score_u8[safe_mask > 0]
        if len(valid_vals) == 0:
            raise RuntimeError("Empty safe mask.")

        thr = float(valid_vals.mean() + 2.2 * valid_vals.std())
        thr = min(max(thr, 25), 230)

        _, binary = cv2.threshold(score_u8, int(thr), 255, cv2.THRESH_BINARY)
        binary[coin_mask == 0] = 0

        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k2)

        binary = self.filter_components_by_shape(binary)
        return binary, int(thr)

    def overlay_heatmap(self, image_gray, score_map, coin_mask, alpha=0.45):
        base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        heat = self.normalize_to_uint8(score_map)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat[coin_mask == 0] = 0
        return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)

    def overlay_defects(self, image_bgr, defect_mask, color=(0, 0, 255)):
        out = image_bgr.copy()
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 绘制轮廓
        cv2.drawContours(out, contours, -1, color, 2)
        # 绘制边界框
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        return out

    def overlay_masks(self, image_bgr, coin_mask, safe_mask):
        out = image_bgr.copy()
        contours1, _ = cv2.findContours(coin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours1, -1, (0, 255, 0), 2)
        cv2.drawContours(out, contours2, -1, (255, 0, 0), 2)
        return out

    def run(self, template_path, test_path, save_dir="outputs_v3"):
        self.ensure_dir(save_dir)

        template_bgr = self.read_image(template_path)
        test_bgr = self.read_image(test_path)
        tpl_crop, tpl_orig_circle = self.extract_circular(template_bgr)
        tst_crop, tst_orig_circle = self.extract_circular(test_bgr)
        
        tpl_circle = self.detect_coin_circle(tpl_crop)
        tst_circle = self.detect_coin_circle(tst_crop)

        tpl_circle_vis = self.draw_circle(tpl_crop, tpl_circle)
        tst_circle_vis = self.draw_circle(tst_crop, tst_circle)

        tpl_crop, tpl_coin_mask, tpl_safe_mask = self.crop_and_normalize_coin(tpl_crop, tpl_circle)
        tst_crop, tst_coin_mask, tst_safe_mask = self.crop_and_normalize_coin(tst_crop, tst_circle)

        coin_mask = cv2.bitwise_and(tpl_coin_mask, tst_coin_mask)
        safe_mask = cv2.bitwise_and(tpl_safe_mask, tst_safe_mask)

        tpl_gray = cv2.cvtColor(tpl_crop, cv2.COLOR_BGR2GRAY)
        tst_gray = cv2.cvtColor(tst_crop, cv2.COLOR_BGR2GRAY)

        tpl_gray = self.illumination_normalize(self.preprocess_gray(tpl_gray))
        tst_gray = self.illumination_normalize(self.preprocess_gray(tst_gray))

        tpl_gray[coin_mask == 0] = 0
        tst_gray[coin_mask == 0] = 0

        best_angle, tst_rot = self.coarse_rotation_search(tpl_gray, tst_gray, safe_mask)
        tst_aligned, warp_matrix, ecc_score = self.ecc_align(tpl_gray, tst_rot, safe_mask)
        tst_aligned[coin_mask == 0] = 0

        score_map, dark_residual, blackhat, dog, structure_mask, highlight_mask = self.build_dark_blob_map(
            tpl_gray, tst_aligned, coin_mask, safe_mask
        )

        # 调试：查看score_map的统计信息
        score_mean = float(score_map.mean())
        score_std = float(score_map.std())
        score_max = float(score_map.max())
        score_min = float(score_map.min())
        print(f"Score map stats: mean={score_mean:.4f}, std={score_std:.4f}, max={score_max:.4f}, min={score_min:.4f}")

        # 使用DINOv2的筛选逻辑
        detection_result = self.detect_and_box_defects_v2(
            tst_aligned,
            score_map,
            k=2.0,  # 适当的k值
            min_thr=0.05,  # 降低到更适合实际score map范围的阈值
            top_percent=0.05,  # 保持较高以包含更多区域
            min_area=30,  # 进一步降低最小面积
            max_area=10000,
            max_aspect_ratio=3.5,  # 稍微放宽长宽比
            min_solidity=0.4,  # 进一步降低最小实心度
            min_fill_ratio=0.1,  # 进一步降低最小填充比
            min_local_contrast=0.01,  # 进一步降低最小局部对比度
            safe_mask=safe_mask,
        )

        mask = detection_result["mask"]
        box_infos = detection_result["box_infos"]
        defect_vis = detection_result["defect_vis"]

        defect_mask, threshold = self.segment_defect(score_map, coin_mask, safe_mask)

        score_overlay = self.overlay_heatmap(tpl_gray, score_map, coin_mask, alpha=0.45)
        aligned_bgr = cv2.cvtColor(tst_aligned, cv2.COLOR_GRAY2BGR)
        defect_overlay = self.overlay_defects(aligned_bgr, defect_mask)
        mask_overlay = self.overlay_masks(tpl_crop, coin_mask, safe_mask)

        cv2.imwrite(os.path.join(save_dir, "01_template_circle.jpg"), tpl_circle_vis)
        cv2.imwrite(os.path.join(save_dir, "02_test_circle.jpg"), tst_circle_vis)
        cv2.imwrite(os.path.join(save_dir, "03_template_crop.jpg"), tpl_crop)
        cv2.imwrite(os.path.join(save_dir, "04_test_crop.jpg"), tst_crop)
        cv2.imwrite(os.path.join(save_dir, "05_masks_overlay.jpg"), mask_overlay)
        cv2.imwrite(os.path.join(save_dir, "06_template_gray_norm.jpg"), tpl_gray)
        cv2.imwrite(os.path.join(save_dir, "07_test_gray_norm.jpg"), tst_gray)
        cv2.imwrite(os.path.join(save_dir, "08_test_rotated.jpg"), tst_rot)
        cv2.imwrite(os.path.join(save_dir, "09_test_aligned.jpg"), tst_aligned)
        cv2.imwrite(os.path.join(save_dir, "10_dark_residual.jpg"), self.normalize_to_uint8(dark_residual))
        cv2.imwrite(os.path.join(save_dir, "11_blackhat.jpg"), self.normalize_to_uint8(blackhat))
        cv2.imwrite(os.path.join(save_dir, "12_dog.jpg"), self.normalize_to_uint8(dog))
        cv2.imwrite(os.path.join(save_dir, "13_structure_mask.jpg"), structure_mask)
        cv2.imwrite(os.path.join(save_dir, "14_highlight_mask.jpg"), highlight_mask)
        cv2.imwrite(os.path.join(save_dir, "15_score_map.jpg"), self.normalize_to_uint8(score_map))
        cv2.imwrite(os.path.join(save_dir, "16_score_overlay.jpg"), score_overlay)
        cv2.imwrite(os.path.join(save_dir, "17_defect_mask.jpg"), defect_mask)
        cv2.imwrite(os.path.join(save_dir, "18_defect_overlay.jpg"), defect_overlay)
        cv2.imwrite(os.path.join(save_dir, "19_dino_style_detections.jpg"), defect_vis)

        # 使用过滤逻辑在原始测试图像上绘制检测结果
        original_test_with_defects = test_bgr.copy()

        norm_size = self.output_size
        tst_crop_h, tst_crop_w = tst_crop.shape[:2]
        orig_h, orig_w = test_bgr.shape[:2]

        tst_x, tst_y, tst_r = tst_circle
        margin_r = int(tst_r * self.crop_margin)
        orig_x1 = max(0, tst_x - margin_r)
        orig_y1 = max(0, tst_y - margin_r)

        scaled_box_infos = []
        for info in box_infos:
            x, y, w, h = info["bbox"]

            scale_x = tst_crop_w / norm_size
            scale_y = tst_crop_h / norm_size
            crop_x = int(x * scale_x)
            crop_y = int(y * scale_y)
            crop_w = int(w * scale_x)
            crop_h = int(h * scale_y)

            final_x = int(orig_x1 + crop_x)
            final_y = int(orig_y1 + crop_y)
            final_w = int(crop_w)
            final_h = int(crop_h)

            final_x = max(0, final_x)
            final_y = max(0, final_y)
            final_x2 = min(orig_w, final_x + final_w)
            final_y2 = min(orig_h, final_y + final_h)

            scaled_box_infos.append({
                "bbox": (final_x, final_y, final_x2 - final_x, final_y2 - final_y),
                "area": info["area"],
                "solidity": info["solidity"],
                "fill_ratio": info["fill_ratio"],
                "contrast": info["contrast"],
                "score_mean": info["score_mean"],
            })

        for i, info in enumerate(scaled_box_infos, 1):
            x, y, w, h = info["bbox"]
            cv2.rectangle(original_test_with_defects, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                original_test_with_defects,
                f"defect_{i}",
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # 保存带有检测结果的原始图像
        cv2.imwrite(os.path.join(save_dir, "20_original_test_with_defects.jpg"), original_test_with_defects)

        results = {
            "template_circle": tpl_circle,
            "test_circle": tst_circle,
            "best_coarse_angle": best_angle,
            "ecc_score": ecc_score,
            "threshold": threshold,
            "defect_pixels": int((defect_mask > 0).sum()),
            "is_defective": bool((defect_mask > 0).sum() > 0),
            "save_dir": save_dir,
            "warp_matrix": warp_matrix,
            "box_infos": scaled_box_infos,
            "detection_result": detection_result,
        }
        return results


def main():
    template_path = r"D:\Code\gb\code\results\ori.jpg"   # 正常模板图
    test_path = r"D:\Code\gb\code\results\aligned_test.jpg"           # 待测图
    save_dir = "outputs_v"

    detector = CoinAnomalyDetectorV3(
        output_size=512,
        crop_margin=1.15,
        coarse_angle_step=10,
        use_clahe=True,
        binary_threshold=40,
        outer_mask_ratio=0.85,
        safe_inner_ratio=0.80,
        highlight_thresh=240,
    )

    results = detector.run(template_path, test_path, save_dir=save_dir)

    print("===== Detection Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()