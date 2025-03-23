import cv2
import numpy as np
import argparse
import os
import json
from matplotlib import pyplot as plt
from PIL import Image

def detect_green_screen(image, min_area=500, debug=False):
    """
    检测图像中的绿色屏幕区域，并返回其四个角点坐标
    
    参数:
    image: 输入图像
    min_area: 最小区域面积，用于过滤小物体
    debug: 是否显示调试图像
    
    返回:
    corners: 检测到的四个角点坐标 [左上, 右上, 右下, 左下]
    """
    # 转换为HSV颜色空间，便于分离绿色区域
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 扩大绿色的HSV范围以适应更多绿色变体
    lower_green = np.array([30, 50, 50])    # 修改：扩大绿色下限
    upper_green = np.array([100, 255, 255]) # 修改：扩大绿色上限
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 调整形态学操作，使用更小的核心
    kernel = np.ones((3, 3), np.uint8)  # 修改：减小核心大小
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        plt.figure(figsize=(15, 8))
        plt.subplot(141), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('原始图像'), plt.axis('off')
        plt.subplot(142), plt.imshow(mask, cmap='gray')
        plt.title('绿色掩码'), plt.axis('off')
        
        # 添加原始HSV图像显示以便调试
        plt.subplot(143), plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        plt.title('HSV图像'), plt.axis('off')
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 根据面积排序，取最大的轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 显示找到的所有轮廓（调试用）
    if debug and contours:
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 255), 2)
        plt.subplot(144), plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title(f'所有轮廓 ({len(contours)}个)'), plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # 打印所有轮廓的面积，帮助调整min_area
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            print(f"轮廓 {i+1} 面积: {area}")
    
    screen_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤掉太小的区域
        if area < min_area:
            continue
        
        # 多边形近似，获取角点
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 只保留近似为四边形的轮廓
        if len(approx) == 4:
            screen_contours.append(approx)
    
    if not screen_contours:
        print("未检测到合适的屏幕区域")
        return None
    
    # 获取角点坐标
    corners_list = []
    for screen in screen_contours:
        # 提取四个角点
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 重新排列角点：左上、右上、右下、左下
        # 先计算每个点到原点的距离
        points = np.squeeze(screen)
        
        # 计算质心
        center = np.mean(points, axis=0)
        
        # 计算角点相对质心的角度
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        
        # 根据角度排序
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        # 找到左上角（通常是角度最小的点）
        top_left = sorted_points[0]
        
        # 现在按照顺时针顺序排列其他点
        # 计算每个点相对于左上角的角度
        relative_angles = np.arctan2(sorted_points[:, 1] - top_left[1], sorted_points[:, 0] - top_left[0])
        
        # 调整角度，使得所有角度都是正的
        relative_angles = np.mod(relative_angles, 2 * np.pi)
        
        # 再次排序
        sorted_indices = np.argsort(relative_angles)
        sorted_points = sorted_points[sorted_indices]
        
        # 正确的顺序应该是: 左上, 右上, 右下, 左下
        # 但由于arctan2的特性，我们需要手动调整
        
        # 首先找到左上角点（x和y坐标之和最小的点）
        s = points.sum(axis=1)
        top_left_idx = np.argmin(s)
        
        # 然后找到右下角点（x和y坐标之和最大的点）
        bottom_right_idx = np.argmax(s)
        
        # 剩下两个点中，x坐标较大的是右上角
        remaining_idx = [i for i in range(4) if i != top_left_idx and i != bottom_right_idx]
        if points[remaining_idx[0]][0] > points[remaining_idx[1]][0]:
            top_right_idx, bottom_left_idx = remaining_idx
        else:
            bottom_left_idx, top_right_idx = remaining_idx
        
        # 按照左上、右上、右下、左下的顺序排列
        rect[0] = points[top_left_idx]
        rect[1] = points[top_right_idx]
        rect[2] = points[bottom_right_idx]
        rect[3] = points[bottom_left_idx]
        
        corners_list.append(rect.tolist())
        
        if debug:
            # 画出检测结果
            result_img = image.copy()
            cv2.drawContours(result_img, [screen], 0, (0, 255, 0), 2)
            
            # 标记四个角点
            for i, point in enumerate(rect):
                cv2.circle(result_img, tuple(map(int, point)), 10, (0, 0, 255), -1)
                cv2.putText(result_img, f"{i}", tuple(map(int, point)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            plt.subplot(133), plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title('检测结果'), plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    return corners_list

def detect_screen_by_edge(image, min_area=500, aspect_ratio_range=(0.3, 0.9), debug=False):
    """
    使用边缘检测和轮廓分析来识别屏幕区域，不依赖颜色信息
    
    参数:
    image: 输入图像
    min_area: 最小区域面积
    aspect_ratio_range: 屏幕宽高比范围 (min_ratio, max_ratio)
    debug: 是否显示调试信息
    
    返回:
    corners: 检测到的四个角点坐标 [左上, 右上, 右下, 左下]
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用高斯模糊减少噪点
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 获取图像亮度的直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # 使用直方图信息自适应设置阈值
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # 修改：调整边缘检测参数，使用较低的阈值以检测更多边缘
    edges = cv2.Canny(blurred, 30, 100)
    
    # 修改：增加膨胀迭代次数，确保边缘连接
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    if debug:
        plt.figure(figsize=(15, 8))
        plt.subplot(141), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('原始图像'), plt.axis('off')
        plt.subplot(142), plt.imshow(binary, cmap='gray')
        plt.title('二值图'), plt.axis('off')
        plt.subplot(143), plt.imshow(edges, cmap='gray')
        plt.title('边缘检测'), plt.axis('off')
        plt.subplot(144), plt.imshow(dilated, cmap='gray')
        plt.title('膨胀边缘'), plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # 查找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 根据面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if debug and contours:
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours[:10], -1, (0, 255, 255), 2)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title(f'前10个最大轮廓'), plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        # 打印前10个轮廓的面积
        for i, cnt in enumerate(contours[:10]):
            area = cv2.contourArea(cnt)
            print(f"轮廓 {i+1} 面积: {area}")
    
    # 创建额外的方法来检测明亮区域
    bright_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bright_contours = sorted(bright_contours, key=cv2.contourArea, reverse=True)
    
    if debug and bright_contours:
        bright_img = image.copy()
        cv2.drawContours(bright_img, bright_contours[:5], -1, (255, 0, 0), 2)
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(bright_img, cv2.COLOR_BGR2RGB))
        plt.title('亮区域检测'), plt.axis('off')
        plt.show()
    
    # 筛选可能的屏幕轮廓
    screen_contours = []
    
    # 尝试从边缘检测结果中找到屏幕
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤掉太小的区域
        if area < min_area:
            continue
        
        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 允许更多的顶点变化
        if 4 <= len(approx) <= 10:  # 修改：允许更多的顶点
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # 修改：使用int32替代int0
            
            # 计算矩形的宽高比
            width = rect[1][0]
            height = rect[1][1]
            
            # 确保宽度大于高度
            if width < height:
                width, height = height, width
                
            aspect_ratio = height / width
            
            # 放宽宽高比条件
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                screen_contours.append(box)
                
                if debug:
                    print(f"找到可能的屏幕，宽高比: {aspect_ratio:.2f}, 面积: {area}")
    
    # 如果边缘检测没找到，尝试使用亮度检测
    if not screen_contours and bright_contours:
        for contour in bright_contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
                
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # 修改：使用int32替代int0
            
            # 计算矩形的宽高比
            width = rect[1][0]
            height = rect[1][1]
            
            if width < height:
                width, height = height, width
                
            aspect_ratio = height / width
            
            # 放宽宽高比检查
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                screen_contours.append(box)
                
                if debug:
                    print(f"从亮区域检测到屏幕，宽高比: {aspect_ratio:.2f}, 面积: {area}")
    
    if not screen_contours:
        # 最后的尝试：直接使用最大的亮区域轮廓，不检查宽高比
        if bright_contours and cv2.contourArea(bright_contours[0]) > min_area:
            rect = cv2.minAreaRect(bright_contours[0])
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # 修改：使用int32替代int0
            screen_contours.append(box)
            
            if debug:
                print(f"使用最大亮区域，面积: {cv2.contourArea(bright_contours[0])}")
    
    if not screen_contours:
        print("未检测到合适的屏幕区域")
        return None
    
    # 获取角点坐标
    corners_list = []
    for screen in screen_contours:
        # 重新排列角点：左上、右上、右下、左下
        rect = order_points(screen)
        corners_list.append(rect.tolist())
        
        if debug:
            # 绘制结果
            result_img = image.copy()
            for i, point in enumerate(rect):
                cv2.circle(result_img, tuple(map(int, point)), 10, (0, 0, 255), -1)
                cv2.putText(result_img, f"{i}", tuple(map(int, point)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 绘制矩形轮廓
            pts = rect.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.title('检测结果'), plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    return corners_list

def order_points(pts):
    """
    按照左上、右上、右下、左下的顺序排列点
    """
    # 确保pts是numpy数组
    pts = np.array(pts)
    
    # 初始化结果数组
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # 计算左上角和右下角
    # 左上角坐标之和最小
    # 右下角坐标之和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 计算右上角和左下角
    # 通过差值区分：右上差最大（x大y小），左下差最小（x小y大）
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def save_corners(corners, output_file):
    """保存角点坐标到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corners, f, indent=4)
    print(f"角点坐标已保存到 {output_file}")

def debug_imshow(windows):
    """
    使用cv2.imshow显示多个图像窗口
    
    参数:
    windows: 包含(窗口名称, 图像)元组的列表
    """
    for window_name, img in windows:
        cv2.imshow(window_name, img)
    
    # 等待按键
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_bright_green_screen(image, min_area=1000, debug=False):
    """
    专门用于检测黑色背景中的绿色屏幕
    
    参数:
    image: 输入图像
    min_area: 最小区域面积
    debug: 是否显示调试图像
    
    返回:
    corners: 检测到的四个角点坐标 [左上, 右上, 右下, 左下]
    """
    # 分离BGR通道
    b, g, r = cv2.split(image)
    
    # 使用更直接的绿色检测方法（适用于饱和度高的绿色）
    bright_green_mask = cv2.bitwise_and(
        cv2.threshold(g, 100, 255, cv2.THRESH_BINARY)[1],
        cv2.bitwise_and(
            cv2.threshold(g - r, 50, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(g - b, 50, 255, cv2.THRESH_BINARY)[1]
        )
    )
    
    # 尝试多种方法结合
    green_mask_hsv = None
    if bright_green_mask.sum() < 1000:
        # 使用HSV空间作为备用方法
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask_hsv = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
        
        if debug:
            # 显示两种掩码对比
            debug_imshow([
                ("RGB绿色掩码", bright_green_mask),
                ("HSV绿色掩码", green_mask_hsv)
            ])
            
        # 组合两种方法
        green_mask = cv2.bitwise_or(bright_green_mask, green_mask_hsv)
    else:
        green_mask = bright_green_mask
    
    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        # 显示处理过程图像
        debug_imshow([
            ("原始图像", image),
            ("绿色通道", g),
            ("绿色掩码", green_mask)
        ])
    
    # 查找轮廓
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 根据面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if debug and contours:
        # 绘制轮廓
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours[:5], -1, (0, 255, 255), 2)
        
        debug_imshow([("绿色区域轮廓", contour_img)])
        
        # 打印前5个轮廓的面积
        for i, cnt in enumerate(contours[:5]):
            area = cv2.contourArea(cnt)
            print(f"轮廓 {i+1} 面积: {area}")
    
    # 筛选可能的屏幕轮廓 - 通常是最大的绿色区域
    screen_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤掉太小的区域
        if area < min_area:
            continue
        
        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 取最大的凸多边形
        hull = cv2.convexHull(contour)
        hull_approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # 检查形状是否近似为四边形 (允许一定的误差)
        if 4 <= len(approx) <= 8:
            # 使用最小面积矩形拟合
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            screen_contours.append(np.int32(box))
            
            if debug:
                print(f"找到可能的屏幕，面积: {area}, 顶点数: {len(approx)}")
                print(f"最小面积矩形：{rect}")
            
            # 只取最大的一个区域
            break
    
    if not screen_contours:
        # 如果没找到符合四边形的轮廓，使用最大的轮廓
        if contours and cv2.contourArea(contours[0]) >= min_area:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            screen_contours.append(np.int32(box))
            
            if debug:
                print(f"使用最大绿色区域，面积: {cv2.contourArea(contours[0])}")
    
    if not screen_contours:
        print("未检测到合适的绿色屏幕区域")
        return None
    
    # 获取角点坐标
    corners_list = []
    for screen in screen_contours:
        # 重新排列角点：左上、右上、右下、左下
        rect = order_points(screen)
        corners_list.append(rect.tolist())
        
        if debug:
            # 绘制结果
            result_img = image.copy()
            for i, point in enumerate(rect):
                cv2.circle(result_img, tuple(map(int, point)), 10, (0, 0, 255), -1)
                cv2.putText(result_img, f"{i}", tuple(map(int, point)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 绘制矩形轮廓
            pts = rect.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
            
            debug_imshow([("检测结果", result_img)])
    
    return corners_list

def load_image_with_alpha(image_path, debug=False):
    """
    专门用于加载并正确处理带有透明通道的图像
    
    参数:
    image_path: 图像文件路径
    debug: 是否显示调试信息
    
    返回:
    image: 处理后的BGR图像
    alpha: 透明通道 (0-255，0表示完全透明)
    """
    # 使用PIL读取可以保留透明通道
    from PIL import Image
    pil_image = Image.open(image_path)
    
    # 检查图像模式
    has_alpha = 'A' in pil_image.getbands()
    print(f"图像模式: {pil_image.mode}, 尺寸: {pil_image.size}")
    
    # 转换为numpy数组（保留所有通道）
    np_image = np.array(pil_image)
    
    # 处理透明通道
    if has_alpha:
        print("检测到图像包含透明通道")
        
        if np_image.shape[2] == 4:  # RGBA格式
            # 分离RGB和Alpha通道
            rgb = np_image[:, :, :3]
            alpha = np_image[:, :, 3]
            
            # 根据透明度创建掩码
            # 值为0的地方是完全透明的
            mask = alpha > 0
            
            # 转换为OpenCV的BGR格式
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            if debug:
                # 显示透明通道
                cv2.imshow("透明通道 (白色=不透明)", alpha)
                
                # 创建可视化用的图像
                vis_img = bgr.copy()
                # 将透明区域标记为红色（便于辨识）
                vis_img[~mask] = [0, 0, 255]  # 红色标记透明区域
                cv2.imshow("透明区域标记", vis_img)
                
                # 等待按键
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return bgr, alpha
        
    # 如果没有透明通道，正常转换为BGR
    if len(np_image.shape) == 3 and np_image.shape[2] == 3:
        bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    else:
        bgr = np_image  # 灰度图像或其他情况
        
    return bgr, None

def detect_green_screen_with_alpha(image, alpha, min_area=1000, debug=False):
    """
    使用透明通道辅助的绿色屏幕检测
    
    参数:
    image: BGR图像
    alpha: 透明通道 (0=完全透明, 255=完全不透明)
    min_area: 最小区域面积
    debug: 是否显示调试信息
    
    返回:
    corners_list: 检测到的角点列表
    """
    # 直接使用alpha通道作为掩码 - 255表示完全不透明
    # 二值化以清理部分透明区域
    _, alpha_mask = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    
    # 应用形态学操作清理掩码
    kernel = np.ones((5, 5), np.uint8)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, kernel)
    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        debug_imshow([
            ("原始图像", image),
            ("透明通道", alpha),
            ("二值化透明掩码", alpha_mask)
        ])
    
    # 查找轮廓
    contours, _ = cv2.findContours(alpha_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 根据面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if debug and contours:
        # 绘制所有找到的轮廓
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours[:5], -1, (0, 255, 255), 2)
        debug_imshow([("不透明区域轮廓", contour_img)])
        
        # 打印前几个轮廓的面积
        for i, cnt in enumerate(contours[:5]):
            area = cv2.contourArea(cnt)
            print(f"轮廓 {i+1} 面积: {area}")
    
    # 筛选合适的轮廓
    screen_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # 过滤小区域
        if area < min_area:
            continue
        
        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 使用最小面积矩形拟合，以获得规则的四边形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 保存这个轮廓
        screen_contours.append(box)
        
        if debug:
            print(f"找到合适的不透明区域，面积: {area}, 顶点数: {len(approx)}")
    
    if not screen_contours:
        print("未检测到合适的屏幕区域")
        return None
    
    # 获取角点坐标
    corners_list = []
    for screen in screen_contours:
        # 重新排列角点：左上、右上、右下、左下
        rect = order_points(screen)
        corners_list.append(rect.tolist())
        
        if debug:
            # 绘制结果
            result_img = image.copy()
            # 绘制背景以便于可视化
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [np.int32(rect)], (255, 255, 255))
            result_img = cv2.addWeighted(result_img, 0.7, mask, 0.3, 0)
            
            # 绘制角点和标签
            for i, point in enumerate(rect):
                cv2.circle(result_img, tuple(map(int, point)), 10, (0, 0, 255), -1)
                cv2.putText(result_img, f"{i}", tuple(map(int, point)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 绘制矩形轮廓
            pts = rect.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
            
            debug_imshow([("检测结果", result_img)])
    
    return corners_list

def find_white_phone_outline(image_path, min_area=1000, debug=False):
    """
    专门用于检测透明PNG中的白色手机轮廓
    
    参数:
    image_path: 图像文件路径
    min_area: 最小区域面积
    debug: 是否打印调试信息
    
    返回:
    corners: 检测到的四个角点坐标 [左上, 右上, 右下, 左下]
    """
    # 直接读取图像（不处理透明通道）
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # 假设图像被分为左右两部分，只关注右半部分
    right_half = image[:, width//2:, :]
    
    if debug:
        cv2.imshow("原始图像", image)
        cv2.imshow("右半部分", right_half)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 转换右侧图像为灰度图
    gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理（白色轮廓在黑色背景上）
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 形态学操作，确保轮廓连续
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        cv2.imshow("二值化图像", binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按面积排序并过滤小轮廓
    contours = [cnt for cnt in sorted(contours, key=cv2.contourArea, reverse=True) 
                if cv2.contourArea(cnt) >= min_area]
    
    if not contours:
        print("未检测到合适的轮廓")
        return None
    
    if debug:
        contour_img = right_half.copy()
        cv2.drawContours(contour_img, contours, 0, (0, 255, 0), 2)
        cv2.imshow("检测到的轮廓", contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"检测到 {len(contours)} 个轮廓")
        for i, cnt in enumerate(contours[:3]):
            print(f"轮廓 {i+1} 面积: {cv2.contourArea(cnt)}")
    
    # 获取最大轮廓的最小面积矩形
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    
    # 调整坐标，加上右半部分的偏移
    box[:, 0] += width//2
    
    # 将坐标点重排序为左上、右上、右下、左下
    box_sorted = np.array(sorted(box, key=lambda p: p[0] + p[1]))
    
    # 前两个点按x排序
    if box_sorted[0][0] > box_sorted[1][0]:
        box_sorted[[0, 1]] = box_sorted[[1, 0]]
    
    # 后两个点按x排序
    if box_sorted[2][0] > box_sorted[3][0]:
        box_sorted[[2, 3]] = box_sorted[[3, 2]]
    
    if debug:
        result_img = image.copy()
        # 绘制检测结果
        for i, point in enumerate(box_sorted):
            point_int = (int(point[0]), int(point[1]))
            cv2.circle(result_img, point_int, 10, (0, 0, 255), -1)
            cv2.putText(result_img, f"{i}", point_int, 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 绘制矩形
        pts = box_sorted.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
        
        cv2.imshow("最终结果", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return [box_sorted.tolist()]

def main():
    parser = argparse.ArgumentParser(description="自动检测图像中的平行四边形区域")
    parser.add_argument("image", help="输入图像路径")
    parser.add_argument("-o", "--output", default="detected_corners.json", help="输出JSON文件路径")
    parser.add_argument("-d", "--debug", action="store_true", help="显示调试信息和图像")
    parser.add_argument("-m", "--min-area", type=int, default=5000, help="最小区域面积")
    parser.add_argument("-e", "--edge", action="store_true", help="使用边缘检测方法而非颜色检测")
    parser.add_argument("-a", "--aspect-ratio", type=str, default="0.4,0.7", 
                        help="屏幕宽高比范围，格式为'min,max'")
    parser.add_argument("-g", "--green", action="store_true", 
                        help="使用专门的绿色屏幕检测方法（针对黑色背景中的绿色屏幕）")
    parser.add_argument("--alpha", action="store_true", 
                        help="特殊处理带有透明通道的图像")
    parser.add_argument("--special", action="store_true", 
                       help="使用特殊方法处理白色手机轮廓图像")
    
    args = parser.parse_args()
    
    # 检查图像文件路径 - 新增
    full_path = os.path.abspath(args.image)
    print(f"图像完整路径: {full_path}")
    
    if not os.path.exists(full_path):
        print(f"警告: 指定的图像文件不存在！")
        
        # 尝试列出当前目录下所有图像文件
        current_dir = os.path.dirname(full_path) if os.path.dirname(full_path) else '.'
        all_images = [f for f in os.listdir(current_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if all_images:
            print(f"当前目录下的图像文件:")
            for img in all_images:
                print(f"  - {img}")
        else:
            print(f"当前目录下没有找到图像文件")
            
        return
    
    # 解析宽高比范围
    aspect_ratio_range = tuple(map(float, args.aspect_ratio.split(',')))
    
    # 根据参数选择图像加载方式
    if args.special:
        print("使用特殊方法处理白色手机轮廓...")
        corners = find_white_phone_outline(args.image, min_area=args.min_area, debug=args.debug)
    elif args.alpha:
        print("使用支持透明通道的图像加载方式...")
        image, alpha_channel = load_image_with_alpha(args.image, debug=args.debug)
        
        if alpha_channel is not None:
            # 将保存的图像也包含透明通道的可视化
            alpha_vis = cv2.merge([alpha_channel, alpha_channel, alpha_channel])
            combined = np.hstack((image, alpha_vis))
            cv2.imwrite("debug_loaded_image_with_alpha.png", combined)
            print("已保存带透明通道的调试图像")
            
            # 对于绿色屏幕检测，可以使用透明通道作为额外的信息
            if args.green:
                # 创建一个使用透明度信息改进的绿色检测函数
                corners = detect_green_screen_with_alpha(image, alpha_channel, 
                                                      min_area=args.min_area, debug=args.debug)
            else:
                # 使用普通方法
                corners = detect_bright_green_screen(image, min_area=args.min_area, debug=args.debug)
        else:
            print("图像不包含透明通道，使用标准方法...")
            # 使用原有方法处理
            if args.green:
                corners = detect_bright_green_screen(image, min_area=args.min_area, debug=args.debug)
            elif args.edge:
                corners = detect_screen_by_edge(image, min_area=args.min_area, 
                                            aspect_ratio_range=aspect_ratio_range, debug=args.debug)
            else:
                corners = detect_green_screen(image, min_area=args.min_area, debug=args.debug)
    else:
        # 尝试多种方法读取图像
        image = cv2.imread(args.image)
        if image is None:
            print(f"尝试使用其他方法加载图像...")
            
            try:
                # 方法1: 使用PIL读取
                pil_image = Image.open(args.image)
                image = np.array(pil_image)
                # OpenCV使用BGR，PIL使用RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print("使用PIL成功读取图像")
            except Exception as e:
                print(f"PIL加载失败: {e}")
                
                try:
                    # 方法2: 使用二进制方式读取
                    with open(args.image, 'rb') as f:
                        img_data = f.read()
                    nparr = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    print("使用imdecode成功读取图像")
                except Exception as e:
                    print(f"二进制加载失败: {e}")
                    return
        
        # 显示正在处理的图像信息
        print(f"正在处理图像: {args.image}")
        print(f"图像尺寸: {image.shape}")
        
        # 验证图像内容 - 新增
        image_hash = hash(image.tobytes())
        print(f"图像哈希值: {image_hash}")
        
        # 保存读取的图像 - 新增
        cv2.imwrite("debug_loaded_image.png", image)
        print(f"已将读取到的图像保存为 debug_loaded_image.png，请检查该图像是否与预期一致")
        
        # 确认处理的是正确的图像
        if args.debug:
            cv2.imshow("输入图像", image)
            print("这是您想处理的图像，按任意键继续，或按ESC取消")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == 27:  # ESC键
                print("已取消操作。")
                return
        
        # 根据选择的方法检测角点
        if args.green:
            print("使用专门的绿色屏幕检测方法...")
            corners = detect_bright_green_screen(image, min_area=args.min_area, debug=args.debug)
        elif args.edge:
            print("使用边缘检测方法...")
            corners = detect_screen_by_edge(
                image, min_area=args.min_area, 
                aspect_ratio_range=aspect_ratio_range, debug=args.debug
            )
        else:
            print("使用标准绿色检测方法...")
            corners = detect_green_screen(image, min_area=args.min_area, debug=args.debug)
    
    if corners:
        # 保存角点坐标
        save_corners(corners, args.output)
        
        # 打印角点坐标
        for i, screen_corners in enumerate(corners):
            print(f"屏幕 {i+1} 的角点坐标:")
            print(f"  左上: {screen_corners[0]}")
            print(f"  右上: {screen_corners[1]}")
            print(f"  右下: {screen_corners[2]}")
            print(f"  左下: {screen_corners[3]}")
    else:
        print("未检测到任何屏幕区域")

if __name__ == "__main__":
    main() 