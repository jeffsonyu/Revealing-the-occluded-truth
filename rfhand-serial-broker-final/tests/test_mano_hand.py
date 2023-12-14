import numpy as np
from pyrfuniverse.side_channel.side_channel import (
    IncomingMessage,
    OutgoingMessage,
)
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv

show_force = False


def SetValue(nd_voltage: np.ndarray, nd_force: np.ndarray):
    list_voltage = nd_voltage.reshape(-1).astype(np.float32).tolist()
    list_force = nd_force.reshape(-1).astype(np.float32).tolist()
    env.SendMessage('SetValue', nd_voltage.shape[0], nd_voltage.shape[1], list_voltage, list_force)
    env.step()


def SetMinMax(min: float, max: float):
    env.SendMessage('SetMinMax', float(min), float(max))
    env.step()


def SetColorList(nd_color: np.ndarray):
    list_color = nd_color.reshape(-1).astype(np.float32).tolist()
    env.SendMessage('SetColorList', list_color)
    env.step()


def SetShowForce(local_show_force: bool):
    if local_show_force:
        # 设置颜色映射范围
        SetMinMax(1, 10)
        # 设置颜色列表，支持多种颜色线性差值，传入n*3的ndarray颜色列表。示例：1-10 黑色->白色
        color_list = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        SetColorList(color_list)
    else:
        # 设置颜色映射范围
        SetMinMax(2.5, 5)
        # 设置颜色列表，支持多种颜色线性差值，传入n*3的ndarray颜色列表。示例：2.5-3.75-5 绿色->黄色->红色
        color_list = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
        SetColorList(color_list)


def ShowForce(msg: IncomingMessage):
    global show_force
    show_force = msg.read_bool()
    SetShowForce(show_force)


def voltage2force(voltage: np.ndarray):
    force = voltage.copy()
    # block 0-16
    force[0:6, 0:3] = force[0:6, 0:3] * 2
    # block 0-15
    force[0:6, 3:6] = force[0:6, 3:6]
    # block 0-14
    force[0:6, 7:12] = force[0:6, 7:12]
    # block 0-13
    force[0:6, 13:18] = force[0:6, 13:18]
    # block 0-12
    force[0:6, 18:21] = force[0:6, 18:21]
    # block 0-11
    force[0:6, 21:24] = force[0:6, 21:24]

    # block 0-26
    force[7:12, 0:3] = force[7:12, 0:3]
    # block 0-25
    force[7:12, 3:6] = force[7:12, 3:6]
    # block 0-24
    force[7:12, 7:12] = force[7:12, 7:12]
    # block 0-23
    force[7:12, 13:18] = force[7:12, 13:18]
    # block 0-22
    force[7:12, 18:21] = force[7:12, 18:21]
    # block 0-21
    force[7:12, 21:24] = force[7:12, 21:24]

    # block 0-36
    force[12:17, 0:3] = force[12:17, 0:3]
    # block 0-35
    force[12:17, 3:6] = force[12:17, 3:6]
    # block 0-34
    force[12:17, 7:12] = force[12:17, 7:12]
    # block 0-33
    force[12:17, 13:18] = force[12:17, 13:18]
    # block 0-32
    force[12:17, 18:21] = force[12:17, 18:21]
    # block 0-31
    force[12:17, 21:24] = force[12:17, 21:24]

    # block 0-46
    force[18:24, 0:3] = force[18:24, 0:3]
    # block 0-45
    force[18:24, 3:6] = force[18:24, 3:6]
    # block 0-44
    force[18:24, 7:12] = force[18:24, 7:12]
    # block 0-43
    force[18:24, 13:18] = force[18:24, 13:18]
    # block 0-42
    force[18:24, 18:21] = force[18:24, 18:21]
    # block 0-41
    force[18:24, 21:24] = force[18:24, 21:24]

    # block 1-1
    force[16:24, 24:27] = force[16:24, 24:27]
    # block 1-2
    force[8:15, 24:27] = force[8:15, 24:27]
    # block 2-1
    force[16:24, 28:31] = force[16:24, 28:31]
    # block 2-2
    force[8:15, 28:31] = force[8:15, 28:31]
    # block 2-3
    force[0:7, 28:31] = force[0:7, 28:31]
    # block 3-1
    force[16:24, 32:35] = force[16:24, 32:35]
    # block 3-2
    force[8:15, 32:35] = force[8:15, 32:35]
    # block 3-3
    force[0:7, 32:35] = force[0:7, 32:35]
    # block 4-1
    force[16:24, 36:39] = force[16:24, 36:39]
    # block 4-2
    force[8:15, 36:39] = force[8:15, 36:39]
    # block 4-3
    force[0:7, 36:39] = force[0:7, 36:39]
    # block 5-1
    force[16:24, 40:43] = force[16:24, 40:43]
    # block 5-2
    force[8:15, 40:43] = force[8:15, 40:43]
    # block 5-3
    force[0:7, 40:43] = force[0:7, 40:43]

    return force


# 启动窗口
env = RFUniverseBaseEnv()
# 注册UI切换回调，切换GUI上的ShowForce按钮时，show_force参数随之切换
env.AddListener("ShowForce", ShowForce)
# 设置时间步长，单位秒，影响刷新率，与手套刷新率保持一致：1/0.077≈13
env.SetTimeStep(0.077)

# 示例
while True:
    value = np.zeros([24, 48], dtype=np.float32)
    for _ in range(500):
        value = value + 0.01
        # SetValue方法：传入max: float, voltage: ndarray, force: ndarray.    shape(24*48)
        # 根据show_force决定是否进行convert
        if show_force:
            SetValue(nd_voltage=value, nd_force=voltage2force(value))
        else:
            SetValue(nd_voltage=value, nd_force=value)
