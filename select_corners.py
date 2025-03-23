import cv2
import numpy as np
import json
import os

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
    
    # 拖拽相关变量
    dragging = False
    drag_point_idx = -1
    drag_threshold = 15  # 检测拖拽的距离阈值
    
    # 屏幕状态标志，0:选择点中，1:微调中
    screen_state = 0
    
    # 线段颜色列表
    line_colors = [
        (0, 0, 255),    # 红色
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (255, 255, 0)   # 青色
    ]
    
    def draw_screen(img, corners):
        """绘制当前选择的屏幕角点和连线"""
        for i, point in enumerate(corners):
            # 绘制点
            cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
            # 添加序号标签
            cv2.putText(img, str(i+1), (point[0]+10, point[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 绘制连线
            if i > 0:
                cv2.line(img, tuple(corners[i-1]), tuple(corners[i]), 
                         line_colors[(i-1) % len(line_colors)], 2)
        
        # 如果有4个点，连接最后一个点和第一个点，形成闭环
        if len(corners) == 4:
            cv2.line(img, tuple(corners[3]), tuple(corners[0]), 
                     line_colors[3 % len(line_colors)], 2)
    
    def find_closest_point(x, y, corners):
        """查找距离点击位置最近的角点"""
        if not corners:
            return -1
        
        for i, point in enumerate(corners):
            dist = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist < drag_threshold:
                return i
        return -1
    
    def update_display():
        """更新显示图像"""
        img_copy = img.copy()
        # 绘制已完成的屏幕
        for screen_corners in all_corners:
            for i, point in enumerate(screen_corners):
                cv2.circle(img_copy, tuple(point), 5, (128, 128, 128), -1)
                cv2.putText(img_copy, f"S{all_corners.index(screen_corners)+1}-{i+1}", 
                            (point[0]+10, point[1]+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                if i > 0:
                    cv2.line(img_copy, tuple(screen_corners[i-1]), tuple(screen_corners[i]), 
                             (128, 128, 128), 1)
            # 闭合线段
            if len(screen_corners) == 4:
                cv2.line(img_copy, tuple(screen_corners[3]), tuple(screen_corners[0]), 
                         (128, 128, 128), 1)
        
        # 绘制当前正在选择的屏幕
        draw_screen(img_copy, current_corners)
        
        # 显示当前屏幕信息
        status_text = f"屏幕 {current_screen + 1}/{num_screens} - 已选择 {len(current_corners)}/4 点"
        cv2.putText(img_copy, status_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 根据状态显示不同的帮助信息
        if len(current_corners) == 4 and screen_state == 0:
            help_text = "拖拽点进行微调，按回车键(ENTER)确认完成，按ESC退出"
            cv2.putText(img_copy, help_text, (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            # 绘制确认按钮
            cv2.rectangle(img_copy, (20, 80), (200, 120), (0, 200, 0), -1)
            cv2.putText(img_copy, "确认(或按回车)", (40, 105), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            help_text = "按ESC退出 | 左键选择/拖拽点 | 选择4点后可进行微调"
            cv2.putText(img_copy, help_text, (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.imshow("请选择屏幕角点", img_copy)
    
    def click_event(event, x, y, flags, param):
        nonlocal current_screen, current_corners, dragging, drag_point_idx, screen_state
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查确认按钮点击（当有4个点时）
            if len(current_corners) == 4 and 20 <= x <= 200 and 80 <= y <= 120:
                # 用户点击了确认按钮
                confirm_current_screen()
                return
            
            # 检查是否点击了已有点（拖拽功能）
            drag_point_idx = find_closest_point(x, y, current_corners)
            
            if drag_point_idx >= 0:
                # 如果点击了已有点，开始拖拽
                dragging = True
            else:
                # 否则，添加新点（如果还没有4个点）
                if len(current_corners) < 4:
                    current_corners.append([x, y])
                    print(f"已选择点 {len(current_corners)}: ({x}, {y})")
                    
                    # 如果已经有4个点，切换到微调模式
                    if len(current_corners) == 4:
                        screen_state = 1
                        print("已选择4个点，您可以进行拖拽微调。完成后请点击确认按钮或按回车键。")
            
            update_display()
                    
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            # 拖拽点移动
            if drag_point_idx >= 0:
                current_corners[drag_point_idx] = [x, y]
                update_display()
                
        elif event == cv2.EVENT_LBUTTONUP:
            # 结束拖拽
            if dragging:
                dragging = False
                print(f"已移动点 {drag_point_idx + 1} 到: ({x}, {y})")
    
    def confirm_current_screen():
        """确认当前屏幕选择并移至下一屏幕"""
        nonlocal current_screen, current_corners, screen_state
        
        all_corners.append(current_corners.copy())
        print(f"屏幕 {current_screen + 1} 的四个角点已确认")
        
        current_screen += 1
        current_corners = []
        screen_state = 0
        
        if current_screen < num_screens:
            print(f"请选择屏幕 {current_screen + 1} 的四个角点...")
        else:
            print("所有屏幕角点已选择，按任意键继续...")
            
        update_display()
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    cv2.namedWindow("请选择屏幕角点")
    cv2.setMouseCallback("请选择屏幕角点", click_event)
    
    print(f"请选择屏幕 {current_screen + 1} 的四个角点 (左上, 右上, 右下, 左下)...")
    print("提示: 可以拖拽已选择的点进行微调")
    
    update_display()
    
    while len(all_corners) < num_screens:
        key = cv2.waitKey(1)
        if key == 27:  # ESC键，提前退出
            break
        elif key == 13 and len(current_corners) == 4:  # 回车键，确认当前屏幕
            confirm_current_screen()
    
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