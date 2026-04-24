import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

# 提取圆形硬币区域
def extract_circular(img):
    if img is None:
        print("输入图像为空")
        return 
        
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("./results/gray.jpg", gray)

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    #cv2.imwrite("./results/blurred.jpg", blurred)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    #cv2.imwrite("./results/edges10.jpg", edges)

    # 检测轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 寻找最大轮廓
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
        print("圆形区域裁剪完成")
    else:
        print("未检测到轮廓")
        result = img
        center = None
        radius = None
    
    return result, (center, radius)

def get_circle_center(img):
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 检测轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        return (int(x), int(y)), int(radius)
    else:
        return None, None

def crop_circle(img, center, radius):
    # 换为整数坐标
    center = (int(center[0]), int(center[1]))
    radius = int(radius)
    
    # 裁剪圆形区域
    x, y = center
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(img.shape[1], x + radius)
    y2 = min(img.shape[0], y + radius)
    return img[y1:y2, x1:x2]

# 平移旋转-特征对齐
def align_images(img1, img1_center, img2, img2_center, auto_rotate=True):
    # 获取两张图片的圆形中心
    center1 = img1_center
    center2 = img2_center
    
    if center1 is not None and center2 is not None:
        # print(f"图片1圆心: {center1}, 半径: {radius1}")
        # print(f"图片2圆心: {center2}, 半径: {radius2}")
        
        h, w = img1.shape[:2]
        
        # 1. 基于圆心的位置对齐
        print("执行基于圆心的位置对齐...")
        M_position = np.eye(3)
        M_position[0, 2] = center1[0] - center2[0]
        M_position[1, 2] = center1[1] - center2[1]
        position_aligned = cv2.warpPerspective(img2, M_position, (w, h))
        
        # 2. 基于特征点的精确对齐
        print("执行基于特征点的精确对齐...")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(position_aligned, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        print(f"找到 {len(good_matches)} 个好的匹配点")
        
        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M_feature, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            aligned_img2 = cv2.warpPerspective(position_aligned, M_feature, (w, h))
            print("特征对齐完成")
        else:
            print("匹配点不足，仅使用位置对齐")
            aligned_img2 = position_aligned
        
        #cv2.imwrite("./results/position_aligned.jpg", position_aligned)
        #cv2.imwrite("./results/feature_aligned.jpg", aligned_img2)
    else:
        print("无法检测到圆形轮廓")
        aligned_img2 = img2
    
    return aligned_img2

class  DinoFeatureDiffDetector:
    def __init__(self, model_name='dinov2_vits14', device=None, input_size=518, safe_inner_ratio=0.82):

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.safe_inner_ratio = safe_inner_ratio

        # 加载模型
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
        x = self.preprocess_pil(bgr_img)

        # 提取特征
        feats = self.model.forward_features(x)

        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            patch_tokens = feats["x_norm_patchnorms"]
        else:
            raise RuntimeError("Please inspect your DINOv2 forward_features output keys")
        
        B, N, C = patch_tokens.shape
        side = int(np.sqrt(N))

        fmap = patch_tokens.reshape(B, side, side, C).permute(0, 3, 1, 2).contiguous()
        return fmap


    def compute_feature_distance_map(self, template_bgr, test_bgr):
        ft_t = self.extract_patch_features(template_bgr)
        ft_x = self.extract_patch_features(test_bgr)

        dist = torch.norm(ft_t - ft_x, dim=1, keepdim=True)

        dist_up = F.interpolate(
            dist,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False,
        )[0, 0]

        score = dist_up.detach().cpu().numpy()
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)
        return score

        


    def run (self, template_bgr, test_bgr):

        template_bgr = cv2.resize(template_bgr, (self.input_size, self.input_size))
        test_bgr = cv2.resize(test_bgr, (self.input_size, self.input_size))

        safe_mask = self.build_safe_mask(self.input_size)

        score_map = self.compute_feature_distance_map(template_bgr, test_bgr)



    
    

# 用DINOV2提取特征
def extract_features(img):
    pass

# 利用特征图进行差分
def diff_features(img1, img2):
    pass

def main():

    # 加载图像
    ori_path = r"D:\Code\gb\code\images\ori.jpg"
    test_path = r"D:\Code\gb\code\images\1.jpg"
    ori_img = cv2.imread(ori_path)
    test_img = cv2.imread(test_path)

    # 裁剪圆形区域
    print("开始裁剪圆形区域...")
    ori_circle_area, (ori_center, ori_radius) = extract_circular(ori_img)
    test_circle_area, (test_center, test_radius) = extract_circular(test_img)

    # 对齐图像
    print("开始对齐原始图像...")
    aligned_test_img = align_images(ori_circle_area, ori_center,test_circle_area, test_center, auto_rotate=True)
    #cv2.imwrite("./results/aligned_test.jpg", aligned_test_img)

    # 裁剪圆
    aligned_test_img = crop_circle(aligned_test_img, ori_center, ori_radius)
    ori_circle_area = crop_circle(ori_circle_area, ori_center, ori_radius)
    cv2.imwrite("./results/aligned_test.jpg", aligned_test_img)
    cv2.imwrite("./results/ori.jpg", ori_circle_area)



if __name__ == "__main__":
    main()