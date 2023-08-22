
# %% ======================== 屏幕截取
# 导入库
import pyautogui
from ultralytics import YOLO
import numpy as np

# 加载模型
model = YOLO('best.pt')
# 定位大体区域
x1,y1,x2,y2 = 140, 110, 570, 510
# 分类名称
names = {0: 'start', 1: 'target', 2: 'fill', 3: 'operate', 4: 'refresh'}

# 截取整个屏幕并检测结果
def get_current():
    screenshot = pyautogui.screenshot()
    cropped = screenshot.crop((x1,y1,x2,y2))
    results = model.predict(source=cropped)
    # 获取检测到的矩形框
    b_datas = results[0].boxes.data
    print("检测原始数据:", b_datas)
    points = {}
    # 将数据按照{"start":[x1,y1,x2,y2]}的格式重组，方便调用
    for box in b_datas:
        if box[4] > 0.65: # 概率大于80%才记录
            name = names[int(box[5])]
            points[name] = np.array(box[:4], np.int32)
    print("加工后:", points)
    return points

# 根据结果获取移动信息
def get_move_info(points):

    if "operate" not in points or "target" not in points:
        raise ValueError("找不到元素")
    operate_box = points["operate"]
    # 找到操作块的中心点
    centerx_op = (operate_box[0] + operate_box[2])/2 + x1
    centery_op = (operate_box[1] + operate_box[3])/2 + y1
    # 找到终点块的中心
    target_box = points["target"]
    centerx_target = (target_box[0] + target_box[2])//2
    # 如果没有起点块，操作充当
    if "start" not in points:
        start_box = operate_box
    else: # 有起点块
        start_box = points["start"]
        # 找到起点块的中心
    centerx_start = (start_box[0] + start_box[2])//2
    # 计算距离
    drag_distance = centerx_target - centerx_start

    return centerx_op, centery_op, drag_distance

# 拖动鼠标
def darg_mouse(centerx_op, centery_op, drag_distance):
    print(f"从 ({centerx_op}, {centery_op}) 拖动到 {drag_distance}")
    # 移动到操作块中心点
    pyautogui.moveTo(centerx_op, centery_op, duration=1)
    # 按下鼠标左键
    pyautogui.mouseDown()
    # 拖动鼠标
    pyautogui.moveRel(drag_distance, 0, duration=1)
    # 松开鼠标左键
    pyautogui.mouseUp()
# %%
# 获取并识别当前验证码信息
points = get_current()
# 分析结果，找到拖动的参数
centerx_op, centery_op, drag_distance = get_move_info(points)
# 移动鼠标
darg_mouse(centerx_op, centery_op, drag_distance)
# %%
