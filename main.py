import cv2
import numpy as np
import argparse
import os
import json

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

def interactive_select_corners(image_path, num_screens=1):
    """
    允许用户交互式地选择图像中的多个手机屏幕的角点
    
    参数:
    image_path (str): 图像路径
    num_screens (int): 手机屏幕数量
    
    返回:
    list: 每个屏幕的角点坐标列表
    """
    all_corners = []
    current_screen = 0
    current_corners = []
    
    def click_event(event, x, y, flags, param):
        nonlocal current_screen, current_corners, img_copy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            current_corners.append([x, y])
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("请选择屏幕角点", img_copy)
            print(f"已选择点: ({x}, {y})")
            
            if len(current_corners) == 4:
                all_corners.append(current_corners)
                print(f"屏幕 {current_screen + 1} 的四个角点已选择")
                
                current_screen += 1
                current_corners = []
                
                if current_screen < num_screens:
                    print(f"请选择屏幕 {current_screen + 1} 的四个角点...")
                    # 重置显示图像
                    img_copy = img.copy()
                    cv2.imshow("请选择屏幕角点", img_copy)
                else:
                    print("所有屏幕角点已选择，按任意键继续...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
        
    img_copy = img.copy()
    cv2.imshow("请选择屏幕角点", img_copy)
    cv2.setMouseCallback("请选择屏幕角点", click_event)
    
    print(f"请选择屏幕 {current_screen + 1} 的四个角点 (左上, 右上, 右下, 左下)...")
    
    while len(all_corners) < num_screens:
        key = cv2.waitKey(1)
        if key == 27:  # ESC键，提前退出
            break
    
    cv2.destroyAllWindows()
    return all_corners

def save_corners(corners, filename="corners.json"):
    """保存角点坐标到文件"""
    with open(filename, 'w') as f:
        json.dump(corners, f)
    print(f"角点坐标已保存到 {filename}")

def load_corners(filename="corners.json"):
    """从文件加载角点坐标"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

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
