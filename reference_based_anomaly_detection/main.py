import os
import cv2
import numpy as np


class CoinAnomalyDetector:
    """
    基于参考模板的硬币瑕疵检测 baseline
    流程：
    1) 检测硬币圆
    2) 裁切并缩放到固定尺寸
    3) 粗旋转搜索
    4) ECC精配准
    5) 多特征差分生成 anomaly map
    6) 阈值分割 + 形态学后处理
    """

    def __init__(
        self,
        output_size=512,
        crop_margin=1.15,
        canny1=80,
        canny2=160,
        hough_dp=1.2,
        hough_min_dist=100,
        hough_param1=120,
        hough_param2=30,
        min_radius_ratio=0.20,
        max_radius_ratio=0.48,
        coarse_angle_step=10,
        use_clahe=True,
    ):
        self.output_size = output_size
        self.crop_margin = crop_margin
        self.canny1 = canny1
        self.canny2 = canny2
        self.hough_dp = hough_dp
        self.hough_min_dist = hough_min_dist
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.min_radius_ratio = min_radius_ratio
        self.max_radius_ratio = max_radius_ratio
        self.coarse_angle_step = coarse_angle_step
        self.use_clahe = use_clahe

    # =========================
    # 基础工具
    # =========================
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
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return rotated

    @staticmethod
    def normalize_to_uint8(x):
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - mn) / (mx - mn)
        return (y * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def apply_mask(image, mask):
        if len(image.shape) == 2:
            out = image.copy()
            out[mask == 0] = 0
            return out
        out = image.copy()
        out[mask == 0] = 0
        return out

    @staticmethod
    def draw_circle(img, circle, color=(0, 255, 0), thickness=2):
        x, y, r = circle
        out = img.copy()
        cv2.circle(out, (int(x), int(y)), int(r), color, thickness)
        cv2.circle(out, (int(x), int(y)), 2, (0, 0, 255), 3)
        return out

    # =========================
    # 预处理
    # =========================
    def preprocess_gray(self, gray):
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        return gray

    def detect_coin_circle(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu 也可以试，但你这个黑背景图固定阈值通常更稳
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

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
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            (x, y), r = cv2.minEnclosingCircle(cnt)

            # 圆度别太差，半径也别离谱
            if circularity > 0.7 and 50 < r < min(h, w) * 0.45:
                candidates.append((cnt, area, circularity, x, y, r))

        if not candidates:
            # 没有合格候选时，退化为最大轮廓
            cnt = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(cnt)
            return int(x), int(y), int(r)

        # 选面积最大的候选
        cnt, area, circularity, x, y, r = max(candidates, key=lambda z: z[1])

        return int(x), int(y), int(r)

    def crop_and_normalize_coin(self, image_bgr, circle):
        x, y, r = circle
        margin_r = int(r * self.crop_margin)

        x1 = max(0, x - margin_r)
        y1 = max(0, y - margin_r)
        x2 = min(image_bgr.shape[1], x + margin_r)
        y2 = min(image_bgr.shape[0], y + margin_r)

        crop = image_bgr[y1:y2, x1:x2].copy()
        crop = cv2.resize(crop, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

        # 在归一化图里构造圆形mask
        center = self.output_size // 2
        norm_r = int(self.output_size / 2 / self.crop_margin * 0.98)

        yy, xx = np.ogrid[:self.output_size, :self.output_size]
        mask = ((xx - center) ** 2 + (yy - center) ** 2 <= norm_r ** 2).astype(np.uint8) * 255

        return crop, mask

    # =========================
    # 对齐
    # =========================
    def coarse_rotation_search(self, template_gray, test_gray, mask):
        """
        在较粗的角度网格上搜索，让 test 旋转后与 template 最接近
        这里用 masked MSE 作为粗搜索指标
        """
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
        """
        用ECC做细配准。这里使用欧式变换（旋转+平移），避免过度形变。
        """
        template_f = template_gray.astype(np.float32) / 255.0
        input_f = input_gray.astype(np.float32) / 255.0

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-5
        )

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
            # 如果ECC失败，就直接返回粗对齐结果
            return input_gray.copy(), np.eye(2, 3, dtype=np.float32), None

    # =========================
    # 异常图
    # =========================
    def build_anomaly_map(self, template_gray, aligned_gray, mask):
        """
        多种差分融合：
        1) 强度差分
        2) 边缘差分
        3) 局部高频差分
        """
        t = template_gray
        a = aligned_gray

        # 1) 强度差
        abs_diff = cv2.absdiff(t, a).astype(np.float32)

        # 2) Sobel边缘差
        tx = cv2.Sobel(t, cv2.CV_32F, 1, 0, ksize=3)
        ty = cv2.Sobel(t, cv2.CV_32F, 0, 1, ksize=3)
        ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)

        grad_t = cv2.magnitude(tx, ty)
        grad_a = cv2.magnitude(ax, ay)
        grad_diff = np.abs(grad_t - grad_a)

        # 3) 高频差（去掉缓慢光照影响）
        blur_t = cv2.GaussianBlur(t, (0, 0), 5)
        blur_a = cv2.GaussianBlur(a, (0, 0), 5)
        high_t = cv2.subtract(t, blur_t).astype(np.float32)
        high_a = cv2.subtract(a, blur_a).astype(np.float32)
        high_diff = np.abs(high_t - high_a)

        # 标准化
        abs_n = abs_diff / (abs_diff.max() + 1e-6)
        grad_n = grad_diff / (grad_diff.max() + 1e-6)
        high_n = high_diff / (high_diff.max() + 1e-6)

        # 融合权重可调
        anomaly = 0.45 * abs_n + 0.35 * grad_n + 0.20 * high_n

        # 圆外归零
        anomaly[mask == 0] = 0

        # 平滑一下
        anomaly = cv2.GaussianBlur(anomaly, (5, 5), 0)

        return anomaly

    def segment_defect(self, anomaly_map, coin_mask):
        """
        依据异常图生成二值mask
        """
        anomaly_u8 = self.normalize_to_uint8(anomaly_map)

        # 只在硬币区域统计阈值
        valid_vals = anomaly_u8[coin_mask > 0]
        if len(valid_vals) == 0:
            raise RuntimeError("Empty coin mask.")

        # 自适应阈值：均值 + 1.5 * std
        thr = float(valid_vals.mean() + 1.5 * valid_vals.std())
        thr = min(max(thr, 20), 220)

        _, defect_mask = cv2.threshold(anomaly_u8, int(thr), 255, cv2.THRESH_BINARY)

        # 只保留硬币区域
        defect_mask[coin_mask == 0] = 0

        # 形态学去噪
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, k1)
        defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, k2)

        # 去小连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
        clean = np.zeros_like(defect_mask)
        min_area = max(10, int(0.0002 * defect_mask.shape[0] * defect_mask.shape[1]))

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean[labels == i] = 255

        return clean, int(thr)

    # =========================
    # 可视化
    # =========================
    def overlay_heatmap(self, image_gray, anomaly_map, coin_mask, alpha=0.5):
        base = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        heat = self.normalize_to_uint8(anomaly_map)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat[coin_mask == 0] = 0
        out = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
        return out

    def overlay_defects(self, image_bgr, defect_mask, color=(0, 0, 255)):
        out = image_bgr.copy()
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        return out

    def build_coin_masks(self, size, radius_ratio=0.85):
        """
        构造两个mask:
        1) coin_mask: 整枚硬币区域
        2) safe_mask: 去掉最外圈后的稳定检测区域
        """
        center = size // 2
        yy, xx = np.ogrid[:size, :size]

        outer_r = int(size * 0.5 * 0.85)
        inner_r = int(outer_r * radius_ratio)

        coin_mask = (((xx - center) ** 2 + (yy - center) ** 2) <= outer_r ** 2).astype(np.uint8) * 255
        safe_mask = (((xx - center) ** 2 + (yy - center) ** 2) <= inner_r ** 2).astype(np.uint8) * 255

        return coin_mask, safe_mask
    # =========================
    # 主流程
    # =========================
    def run(self, template_path, test_path, save_dir="outputs"):
        self.ensure_dir(save_dir)

        template_bgr = self.read_image(template_path)
        test_bgr = self.read_image(test_path)

        # 1) 检测圆
        tpl_circle = self.detect_coin_circle(template_bgr)
        tst_circle = self.detect_coin_circle(test_bgr)

        tpl_circle_vis = self.draw_circle(template_bgr, tpl_circle)
        tst_circle_vis = self.draw_circle(test_bgr, tst_circle)

        # 2) 裁切归一化
        tpl_crop, tpl_mask = self.crop_and_normalize_coin(template_bgr, tpl_circle)
        tst_crop, tst_mask = self.crop_and_normalize_coin(test_bgr, tst_circle)

        # 使用两者共同mask
        coin_mask = cv2.bitwise_and(tpl_mask, tst_mask)

        tpl_gray = cv2.cvtColor(tpl_crop, cv2.COLOR_BGR2GRAY)
        tst_gray = cv2.cvtColor(tst_crop, cv2.COLOR_BGR2GRAY)
        tpl_gray = self.preprocess_gray(tpl_gray)
        tst_gray = self.preprocess_gray(tst_gray)

        tpl_gray[coin_mask == 0] = 0
        tst_gray[coin_mask == 0] = 0

        # 3) 粗旋转
        best_angle, tst_rot = self.coarse_rotation_search(tpl_gray, tst_gray, coin_mask)

        # 4) ECC细配准
        tst_aligned, warp_matrix, ecc_score = self.ecc_align(tpl_gray, tst_rot, coin_mask)
        tst_aligned[coin_mask == 0] = 0

        # 5) 异常图
        anomaly_map = self.build_anomaly_map(tpl_gray, tst_aligned, coin_mask)

        # 6) 缺陷分割
        defect_mask, threshold = self.segment_defect(anomaly_map, coin_mask)

        # 7) 可视化
        heatmap_overlay = self.overlay_heatmap(tpl_gray, anomaly_map, coin_mask, alpha=0.45)
        aligned_bgr = cv2.cvtColor(tst_aligned, cv2.COLOR_GRAY2BGR)
        defect_overlay = self.overlay_defects(aligned_bgr, defect_mask)

        # 保存
        cv2.imwrite(os.path.join(save_dir, "01_template_circle.jpg"), tpl_circle_vis)
        cv2.imwrite(os.path.join(save_dir, "02_test_circle.jpg"), tst_circle_vis)
        cv2.imwrite(os.path.join(save_dir, "03_template_crop.jpg"), tpl_crop)
        cv2.imwrite(os.path.join(save_dir, "04_test_crop.jpg"), tst_crop)
        cv2.imwrite(os.path.join(save_dir, "05_template_gray.jpg"), tpl_gray)
        cv2.imwrite(os.path.join(save_dir, "06_test_rotated.jpg"), tst_rot)
        cv2.imwrite(os.path.join(save_dir, "07_test_aligned.jpg"), tst_aligned)
        cv2.imwrite(os.path.join(save_dir, "08_anomaly_map.jpg"), self.normalize_to_uint8(anomaly_map))
        cv2.imwrite(os.path.join(save_dir, "09_heatmap_overlay.jpg"), heatmap_overlay)
        cv2.imwrite(os.path.join(save_dir, "10_defect_mask.jpg"), defect_mask)
        cv2.imwrite(os.path.join(save_dir, "11_defect_overlay.jpg"), defect_overlay)

        results = {
            "template_circle": tpl_circle,
            "test_circle": tst_circle,
            "best_coarse_angle": best_angle,
            "ecc_score": ecc_score,
            "threshold": threshold,
            "warp_matrix": warp_matrix,
            "save_dir": save_dir,
            "defect_pixels": int((defect_mask > 0).sum()),
            "is_defective": bool((defect_mask > 0).sum() > 0),
        }
        return results


def main():
    """
    用法示例：
    python coin_anomaly_baseline.py
    然后在下面填你自己的路径
    """
    template_path = r"D:\Code\gb\code\results\ori.jpg"   # 正常模板图
    test_path = r"D:\Code\gb\code\results\aligned_test.jpg"           # 待测图
    save_dir = "outputs"

    detector = CoinAnomalyDetector(
        output_size=512,
        crop_margin=1.15,
        coarse_angle_step=10,   # 可以改成 5 更细，但更慢
        use_clahe=True
    )

    results = detector.run(template_path, test_path, save_dir=save_dir)

    print("===== Detection Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()