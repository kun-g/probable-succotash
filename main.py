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
    
    # 应用透视变换，将截图变换到模板大小
    warped_screenshot = cv2.warpPerspective(
        screenshot_img, M, (template.shape[1], template.shape[0])
    )
    
    # 创建一个掩码，确定屏幕区域
    mask = np.zeros(template.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    
    # 在模板中提取屏幕区域
    screen_area = cv2.bitwise_and(template, template, mask=mask)
    
    # 检测绿色区域 (在HSV颜色空间中更容易识别颜色)
    hsv_screen = cv2.cvtColor(screen_area, cv2.COLOR_BGR2HSV)
    # 设置绿色的HSV范围 - 可能需要根据实际绿屏调整这个值
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_screen, lower_green, upper_green)
    
    # 创建最终掩码：屏幕区域中的非绿色部分
    final_mask = cv2.bitwise_and(mask, cv2.bitwise_not(green_mask))
    
    # 使用最终掩码提取模板中的非绿色部分
    foreground = cv2.bitwise_and(template, template, mask=final_mask)
    
    # 使用green_mask提取屏幕区域，这部分将显示截图
    background_mask = cv2.bitwise_and(mask, green_mask)
    background = cv2.bitwise_and(warped_screenshot, warped_screenshot, mask=background_mask)
    
    # 模板的非屏幕区域
    non_screen_mask = cv2.bitwise_not(mask)
    non_screen_area = cv2.bitwise_and(template, template, mask=non_screen_mask)
    
    # 组合所有部分：非屏幕区域 + 屏幕区域中的非绿色部分 + 截图中对应绿色区域的部分
    result = cv2.add(non_screen_area, foreground)
    result = cv2.add(result, background)
    
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
