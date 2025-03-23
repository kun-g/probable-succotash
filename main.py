import cv2
import numpy as np
import argparse
import os
import json
from select_corners import interactive_select_corners, save_corners, load_corners

def overlay_screenshot(template, screenshot, corners):
    """
    将模板图片覆盖到App截图上，并将屏幕区域的绿色设为透明
    
    参数:
    template: 模板图片（已加载的图像）
    screenshot (str): App截图路径
    corners (list): 手机屏幕四个角的坐标 [左上, 右上, 右下, 左下]
    
    返回:
    处理后的图像
    """
    # 读取截图
    screenshot_img = cv2.imread(screenshot)
    
    if screenshot_img is None:
        raise ValueError(f"无法读取截图文件: {screenshot}")
    
    # 计算透视变换
    corners = np.array(corners, dtype=np.float32)
    screenshot_h, screenshot_w = screenshot_img.shape[:2]
    screenshot_corners = np.array([
        [0, 0],
        [screenshot_w, 0],
        [screenshot_w, screenshot_h],
        [0, screenshot_h]
    ], dtype=np.float32)
    
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(screenshot_corners, corners)
    
    # 应用透视变换，将截图变换到模板大小，使用INTER_LINEAR插值方法
    warped_screenshot = cv2.warpPerspective(
        screenshot_img, M, (template.shape[1], template.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    
    # 创建一个掩码，确定屏幕区域
    mask = np.zeros(template.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    
    # 应用高斯模糊平滑掩码边缘，减少锯齿
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 在模板中提取屏幕区域
    screen_area = cv2.bitwise_and(template, template, mask=mask)
    
    # 检测绿色区域 (在HSV颜色空间中更容易识别颜色)
    hsv_screen = cv2.cvtColor(screen_area, cv2.COLOR_BGR2HSV)
    
    # 设置绿色的HSV范围 - 可能需要根据实际绿屏调整这个值
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_screen, lower_green, upper_green)
    
    # 应用形态学操作平滑绿色掩码边缘
    kernel = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # 应用高斯模糊平滑绿色掩码边缘
    green_mask_blurred = cv2.GaussianBlur(green_mask, (3, 3), 0)
    
    # 创建最终掩码：屏幕区域中的非绿色部分
    final_mask = cv2.bitwise_and(mask_blurred, cv2.bitwise_not(green_mask_blurred))
    
    # 将掩码转换为浮点型，以便进行平滑过渡
    final_mask_float = final_mask.astype(float) / 255.0
    green_mask_float = green_mask_blurred.astype(float) / 255.0
    non_screen_mask = cv2.bitwise_not(mask_blurred)
    non_screen_mask_float = non_screen_mask.astype(float) / 255.0
    
    # 使用浮点掩码进行图像混合
    result = np.zeros_like(template, dtype=float)
    
    # 对每个颜色通道进行混合
    for c in range(3):
        # 非屏幕区域保持原样
        result[:,:,c] += template[:,:,c] * non_screen_mask_float
        
        # 屏幕区域中的非绿色部分使用模板
        result[:,:,c] += template[:,:,c] * final_mask_float
        
        # 屏幕区域中的绿色部分使用截图
        result[:,:,c] += warped_screenshot[:,:,c] * green_mask_float * (mask_blurred / 255.0)
    
    # 转换回uint8类型
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="将App截图覆盖到手机模板上")
    parser.add_argument("template", help="手机模板图片路径")
    parser.add_argument("screenshots", nargs='+', help="App截图路径（可多个）")
    parser.add_argument("-o", "--output", default="result.jpg", help="输出图片路径")
    parser.add_argument("-c", "--corners", help="包含角点坐标的JSON文件")
    parser.add_argument("-s", "--select", action="store_true", help="交互式选择屏幕角点并保存")
    parser.add_argument("-n", "--num-screens", type=int, default=1, help="模板中的手机屏幕数量")
    
    args = parser.parse_args()
    
    # 确保提供的截图数量与屏幕数量匹配
    if len(args.screenshots) != args.num_screens:
        print(f"错误: 提供了 {len(args.screenshots)} 张截图，但指定了 {args.num_screens} 个屏幕")
        return
    
    all_corners = None
    
    # 尝试加载角点坐标
    if args.corners and os.path.exists(args.corners):
        all_corners = load_corners(args.corners)
        print(f"已从 {args.corners} 加载角点坐标")
    
    # 如果需要交互式选择角点
    if args.select or all_corners is None:
        print("进入交互式角点选择模式...")
        all_corners = interactive_select_corners(args.template, args.num_screens)
        
        # 如果成功选择了所有角点，保存它们
        if len(all_corners) == args.num_screens:
            save_filename = args.corners if args.corners else f"{os.path.splitext(os.path.basename(args.template))[0]}_corners.json"
            save_corners(all_corners, save_filename)
    
    if all_corners and len(all_corners) == args.num_screens:
        # 读取模板图像
        template = cv2.imread(args.template)
        if template is None:
            print(f"错误: 无法读取模板图像 {args.template}")
            return
            
        result = template.copy()
        
        # 对每个屏幕应用相应的截图
        for i, (corners, screenshot_path) in enumerate(zip(all_corners, args.screenshots)):
            result = overlay_screenshot(result, screenshot_path, corners)
        
        # 保存最终结果
        cv2.imwrite(args.output, result)
        print(f"结果已保存到 {args.output}")
    else:
        print("错误: 无法获取有效的角点坐标")

if __name__ == "__main__":
    main()
