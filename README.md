# 手机屏幕替换工具

## 项目描述

这个Python工具允许您将App截图无缝替换到手机模板图片上，创建出专业级的展示效果。它使用透视变换来确保App截图正确地适应手机屏幕的角度和形状，无论手机在模板图片中的位置和角度如何。现在增加了API服务功能，可以通过HTTP请求便捷地完成图像处理。

## 功能特点

- 支持任意角度的手机模板
- 交互式选择手机屏幕区域
- 保存屏幕角点坐标以供重复使用
- 处理单个或多个手机屏幕的模板
- 高质量透视变换以确保自然效果
- 简单的命令行界面
- **REST API服务**：通过HTTP请求远程处理图像
- **多种调用方式**：支持命令行和API两种使用方式

## 设计思路

该工具的核心是使用计算机视觉技术将二维图像（App截图）映射到三维空间中的平面（手机屏幕）。这个过程包括以下几个关键步骤：

1. **定位手机屏幕**：通过手动选择屏幕的四个角点来定义屏幕区域
2. **计算透视变换**：根据原始截图和目标屏幕的几何关系计算变换矩阵
3. **应用变换**：将截图变形以匹配屏幕的角度和透视效果
4. **融合图像**：使用掩码将变换后的截图无缝融合到原始模板中

## 技术实现

### 关键算法

1. **透视变换**：使用OpenCV的`getPerspectiveTransform`和`warpPerspective`函数将平面图像映射到透视空间
2. **掩码处理**：使用凸多边形填充创建精确的掩码，确保只替换手机屏幕区域
3. **图像融合**：使用位运算和掩码技术将变换后的截图与原始模板无缝融合

### 主要函数

1. `overlay_screenshot(template, screenshot, corners)`
   - 核心函数，负责执行透视变换和图像融合
   - 接受模板图像、截图路径和屏幕角点坐标作为输入
   - 返回处理后的图像

2. `interactive_select_corners(image_path, num_screens)`
   - 允许用户通过鼠标点击在模板图像上选择手机屏幕的角点
   - 支持选择多个屏幕的角点
   - 返回所有屏幕的角点坐标列表

3. `save_corners(corners, filename)` 和 `load_corners(filename)`
   - 将角点坐标保存到JSON文件或从文件加载角点坐标
   - 便于重复使用相同的模板图像

### API 服务

API服务基于FastAPI框架实现，提供以下功能：

1. **HTTP端点**：通过`/overlay/`端点接收图像处理请求
2. **多文件上传**：支持上传模板图片和多张截图
3. **JSON参数**：使用结构化JSON数据描述屏幕角点位置
4. **异步处理**：异步处理上传的图像，提高性能
5. **异常处理**：完善的错误处理和资源清理机制
6. **临时文件管理**：自动管理和清理处理过程中的临时文件

## 使用方法

### 命令行用法

首次使用时，您需要选择手机屏幕的角点：

```bash
python main.py template.jpg screenshot.jpg -s -o result.jpg
```

这将打开一个窗口让您选择模板图像中手机屏幕的四个角点。点击顺序应为：左上、右上、右下、左下。

### 使用保存的角点

一旦保存了角点坐标，您可以重复使用它们处理多个截图：

```bash
python main.py template.jpg new_screenshot.jpg -c template_corners.json -o new_result.jpg
```

### 处理多个屏幕

对于含有多个手机的模板图像：

```bash
python main.py template.jpg screenshot1.jpg screenshot2.jpg -n 2 -s -o result.jpg
```

### API 服务使用

#### 启动API服务

```bash
python api.py
```

服务将在 http://localhost:8000 启动。您可以访问 http://localhost:8000/docs 查看交互式API文档。

#### 通过curl调用API

```bash
curl -X 'POST' \
     'http://localhost:8000/overlay/' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'template=@template.jpg;type=image/jpeg' \
     -F 'screenshots=@screenshot.jpg;type=image/jpeg' \
     -F 'screen_data={"screens":[[[100,100],[300,100],[300,500],[100,500]]]}' \
     -o result.jpg
```

#### 使用保存的角点文件

```bash
# 读取保存的角点文件并使用其中的坐标
cat template_corners.json | curl -X 'POST' \
     'http://localhost:8000/overlay/' \
     -H 'Content-Type: multipart/form-data' \
     -F 'template=@template.jpg' \
     -F 'screenshots=@screenshot.jpg' \
     -F "screen_data=<-" \
     -o result.jpg
```

## 参数说明

### 命令行参数

- `template`：手机模板图片路径（必需）
- `screenshots`：一个或多个App截图路径（必需）
- `-o, --output`：输出图片路径（默认：result.jpg）
- `-c, --corners`：包含角点坐标的JSON文件（可选）
- `-s, --select`：启用交互式选择屏幕角点（可选）
- `-n, --num-screens`：模板中手机屏幕的数量（默认：1）

### API 参数

- `template`：上传的手机模板图片（表单文件，必需）
- `screenshots`：上传的App截图文件列表（表单文件数组，必需）
- `screen_data`：以JSON格式提供的屏幕角点坐标（表单字段，必需），格式如下：

```json
{
    "screens": [
        [
            [100, 100],  // 左上角
            [300, 100],  // 右上角
            [300, 500],  // 右下角
            [100, 500]   // 左下角
        ]
    ]
}
```

## 提示与技巧

1. **角点选择**：选择角点时，尽量精确定位到屏幕的实际边角，这将影响最终效果的质量
2. **截图尺寸**：为获得最佳效果，使用分辨率较高的App截图
3. **预处理模板**：使用质量高、光线均匀的模板图片，避免强烈的反光或阴影
4. **多屏处理**：处理多个屏幕时，按照从左到右的顺序提供截图
5. **API服务部署**：在生产环境中，建议使用Nginx等反向代理服务器

## 技术要求

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- argparse
- FastAPI (API服务)
- uvicorn (ASGI服务器)
- python-multipart (处理表单数据)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 注意事项

- 在交互式选择角点时，按ESC键可以提前退出
- 确保角点选择的顺序正确：左上、右上、右下、左下
- 为获得最佳效果，模板图片和截图的分辨率应该足够高
- API服务默认监听所有网络接口(0.0.0.0)，可能需要根据安全需求进行调整

## 实例展示

此工具可用于创建产品演示图片、应用宣传材料或UI/UX设计展示。可以通过命令行方式批量处理图像，或者通过API服务集成到自动化工作流程中。
