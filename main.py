import cv2
import numpy as np
import argparse
import os
import json
from select_corners import interactive_select_corners, save_corners, load_corners

def overlay_screenshot(template, screenshot, corners):
    """
    将App截图覆盖到模板图片的手机屏幕上
    
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
    
    # 计算目标四边形的宽高比
    width = np.sqrt(((corners[1][0] - corners[0][0]) ** 2) + ((corners[1][1] - corners[0][1]) ** 2))
    height = np.sqrt(((corners[3][0] - corners[0][0]) ** 2) + ((corners[3][1] - corners[0][1]) ** 2))
    target_ratio = width / height
    
    # 调整截图以匹配目标宽高比
    screenshot_h, screenshot_w = screenshot_img.shape[:2]
    current_ratio = screenshot_w / screenshot_h
    
    if current_ratio > target_ratio:
        # 过宽，需要裁剪宽度
        new_width = int(screenshot_h * target_ratio)
        start_x = (screenshot_w - new_width) // 2
        screenshot_img = screenshot_img[:, start_x:start_x+new_width]
    elif current_ratio < target_ratio:
        # 过高，需要裁剪高度
        new_height = int(screenshot_w / target_ratio)
        start_y = (screenshot_h - new_height) // 2
        screenshot_img = screenshot_img[start_y:start_y+new_height, :]
    
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
    print("变换矩阵:", M)
    
    # 应用透视变换
    warped_screenshot = cv2.warpPerspective(
        screenshot_img, M, (template.shape[1], template.shape[0])
    )
    
    # 创建一个掩码，确定应该被替换的区域
    mask = np.zeros(template.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    
    # 创建掩码的逆
    inv_mask = cv2.bitwise_not(mask)
    
    # 使用掩码分离原始模板的背景和前景
    background = cv2.bitwise_and(template, template, mask=inv_mask)
    
    # 将变换后的截图添加到背景中
    result = cv2.add(background, cv2.bitwise_and(warped_screenshot, warped_screenshot, mask=mask))
    
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
