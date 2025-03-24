import cv2
import numpy as np
import os
import tempfile
import uuid
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
import uvicorn
import json
from main import overlay_screenshot

app = FastAPI(title="手机屏幕替换API", description="将App截图覆盖到手机模板图片上的API服务")

# 更新数据模型为多维数组格式
class OverlayRequest(BaseModel):
    screens: List[List[List[float]]]  # [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [...]]

# 创建临时存储目录
TEMP_DIR = os.path.join(tempfile.gettempdir(), "overlay_api")
os.makedirs(TEMP_DIR, exist_ok=True)

# 结果存储时间（秒）
RESULT_EXPIRY = 300  # 5分钟

@app.post("/overlay/", summary="替换手机屏幕")
async def create_overlay(
    template: UploadFile = File(..., description="手机模板图片"),
    screenshots: List[UploadFile] = File(..., description="App截图文件"),
    screen_data: str = Form(..., description="以JSON格式提供的屏幕角点坐标")
):
    """
    上传手机模板图片和App截图，并提供角点坐标，将截图应用到模板上并返回结果。
    
    角点坐标格式示例：
    ```json
    {
        "screens": [
            [
                [100, 100],  // 左上
                [300, 100],  // 右上
                [300, 500],  // 右下
                [100, 500]   // 左下
            ]
        ]
    }
    ```
    """
    try:
        # 解析请求数据
        overlay_data = json.loads(screen_data)
        if not isinstance(overlay_data, dict) or "screens" not in overlay_data:
            raise HTTPException(status_code=400, detail="screen_data 格式错误，必须包含 'screens' 键")
        
        screens = overlay_data["screens"]
        
        # 检查截图数量是否与屏幕数量匹配
        if len(screenshots) != len(screens):
            raise HTTPException(
                status_code=400, 
                detail=f"提供了 {len(screenshots)} 张截图，但指定了 {len(screens)} 个屏幕"
            )
        
        # 保存上传的模板图片
        template_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(template.filename)[1]}")
        with open(template_path, "wb") as f:
            f.write(await template.read())
        
        # 读取模板图像
        template_img = cv2.imread(template_path)
        if template_img is None:
            raise HTTPException(status_code=400, detail="无法读取模板图像")
        
        result = template_img.copy()
        
        # 处理每个屏幕的截图
        for i, (corners, screenshot) in enumerate(zip(screens, screenshots)):
            # 保存截图
            screenshot_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(screenshot.filename)[1]}")
            with open(screenshot_path, "wb") as f:
                f.write(await screenshot.read())
            
            # 直接使用corners多维数组，无需转换
            # 应用截图
            result = overlay_screenshot(result, screenshot_path, corners)
            
            # 删除临时截图
            os.remove(screenshot_path)
        
        # 删除临时模板
        os.remove(template_path)
        
        # 保存结果
        result_filename = f"{uuid.uuid4()}.jpg"
        result_path = os.path.join(TEMP_DIR, result_filename)
        cv2.imwrite(result_path, result)
        
        # 返回图像，使用BackgroundTask对象
        return FileResponse(
            result_path, 
            media_type="image/jpeg", 
            filename="overlay_result.jpg",
            background=BackgroundTask(lambda: os.remove(result_path) if os.path.exists(result_path) else None)
        )
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="无效的JSON格式")
    except Exception as e:
        # 清理临时文件
        if 'template_path' in locals() and os.path.exists(template_path):
            os.remove(template_path)
        
        for screenshot in screenshots:
            temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{os.path.splitext(screenshot.filename)[1]}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/", summary="API服务信息")
async def root():
    """
    返回API服务的基本信息
    """
    return {
        "service": "手机屏幕替换API",
        "version": "1.0.0",
        "endpoints": {
            "/overlay/": "上传模板和截图，应用替换，返回结果图片",
            "/": "显示API信息"
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 