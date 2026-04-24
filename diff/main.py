import cv2
import numpy as np
import os


def extract_circular(img, save_path="./results/circular_crop.jpg"):

    # 创建保存目录
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")

    if img is None:
        print("传入图片不存在")
        return

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imwrite(os.path.splitext(save_path)[0] + "_edges.jpg", edges)
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
        cv2.imwrite(save_path, result)
        print("圆形区域裁剪完成")
    else:
        print("未检测到轮廓")
    return result

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

def align_images(img1, img2, auto_rotate=True):
    # 获取两张图片的圆形中心
    center1, radius1 = get_circle_center(img1)
    center2, radius2 = get_circle_center(img2)
    
    if center1 is not None and center2 is not None:
        print(f"图片1圆心: {center1}, 半径: {radius1}")
        print(f"图片2圆心: {center2}, 半径: {radius2}")
        
        h, w = img1.shape[:2]
        
        # 1. 基于圆心的位置对齐
        print("执行基于圆心的位置对齐...")
        # 构建平移变换矩阵
        M_position = np.eye(3)
        # 计算平移量
        M_position[0, 2] = center1[0] - center2[0]
        M_position[1, 2] = center1[1] - center2[1]
        # 应用位置对齐
        position_aligned = cv2.warpPerspective(img2, M_position, (w, h))
        
        # 2. 基于特征点的精确对齐
        print("执行基于特征点的精确对齐...")
        # 转换为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(position_aligned, cv2.COLOR_BGR2GRAY)
        
        # 使用SIFT特征检测器
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # 筛选好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        print(f"找到 {len(good_matches)} 个好的匹配点")
        
        # 使用RANSAC进行透视变换
        if len(good_matches) >= 4:
            # 提取匹配点坐标
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 计算透视变换矩阵
            M_feature, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # 应用透视变换
            aligned_img2 = cv2.warpPerspective(position_aligned, M_feature, (w, h))
            print("特征对齐完成")
        else:
            print("匹配点不足，仅使用位置对齐")
            aligned_img2 = position_aligned
        
        # 保存对齐结果
        cv2.imwrite("./results/position_aligned.jpg", position_aligned)
        cv2.imwrite("./results/feature_aligned.jpg", aligned_img2)
        
        # 3. 瑕疵检测
        print("执行瑕疵检测...")
        # 计算差值图像
        gray_aligned = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray_aligned)
        
        # 阈值处理
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 标记瑕疵
        defect_img = aligned_img2.copy()
        defect_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # 过滤小面积噪声
                defect_count += 1
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                cv2.rectangle(defect_img, (x, y), (x + w_contour, y + h_contour), (0, 0, 255), 2)
        
        print(f"检测到 {defect_count} 个瑕疵")
        
        # 保存瑕疵检测结果
        cv2.imwrite("./results/diff.jpg", diff)
        cv2.imwrite("./results/thresh.jpg", thresh)
        cv2.imwrite("./results/defects.jpg", defect_img)
        
        # 保存匹配结果
        if len(good_matches) > 0:
            match_img = cv2.drawMatches(img1, keypoints1, position_aligned, keypoints2, good_matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite("./results/match.jpg", match_img)
            print("已保存match.jpg")
    else:
        print("无法检测到圆形轮廓")
        aligned_img2 = img2
    
    return aligned_img2

if __name__ == "__main__":
    # 读取原始图像
    ori_path = r"D:\Code\gb\code\images\ori.jpg"
    ori_img = cv2.imread(ori_path)
    
    q1_path = r"D:\Code\gb\code\images\1.jpg"
    q1_img = cv2.imread(q1_path)

    # 然后裁剪圆形区域
    print("开始裁剪圆形区域...")
    save_path = "./results/ori_circle.jpg"
    ori_img = extract_circular(ori_img, save_path)
    
    save_path = "./results/1_circle.jpg"
    q1_img = extract_circular(q1_img, save_path)
    
    # 先对齐原始图像 - 使用自动旋转功能
    print("开始对齐原始图像...")
    aligned_q1_img = align_images(ori_img, q1_img, auto_rotate=True)
    
    
    
