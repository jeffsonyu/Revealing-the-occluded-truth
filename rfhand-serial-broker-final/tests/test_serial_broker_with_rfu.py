import argparse
import datetime
import json
import os
import os.path as osp
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import plyer
import tqdm

sys.path.append(os.getcwd())

from serial_broker.Conversion import voltage2force, VoltageToForceParameters, ParameterManager, AmplifierManager

from serial import Serial
from serial_broker import Config, Parser

from pyrfuniverse.side_channel.side_channel import (
    IncomingMessage,
)
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv

show_force = False
conversion_params: Optional[VoltageToForceParameters] = None
env: Optional[RFUniverseBaseEnv] = None


def SetMinMax(env, min: float, max: float):
    env.SendMessage('SetMinMax', float(min), float(max))
    env.step()


def SetColorList(env, nd_color: np.ndarray, nd_value: np.ndarray):
    list_color = nd_color.reshape(-1).astype(np.float32).tolist()
    list_value = nd_value.astype(np.float32).tolist()
    env.SendMessage('SetColorList', list_color, list_value)
    env.step()


def SetValue(env, nd_voltage: np.ndarray, nd_force: np.ndarray):
    list_voltage = nd_voltage.reshape(-1).astype(np.float32).tolist()
    list_force = nd_force.reshape(-1).astype(np.float32).tolist()
    env.SendMessage('SetValue', nd_voltage.shape[0], nd_voltage.shape[1], list_voltage, list_force)
    env.step()


def SetShowForce(env, local_show_force: bool):
    if local_show_force:
        # 设置颜色列表，支持多种颜色线性差值，传入n*3的ndarray颜色列表。示例：1-10 黑色->白色
        color_list = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        value_list = np.array([0, 20], dtype=np.float32)
        SetColorList(env, color_list, value_list)
    else:
        # 设置颜色列表，支持多种颜色线性差值，传入n*3的ndarray颜色列表。示例：1->2.99999999->3->5 绿色->黄色->红色->紫色
        color_list = np.array([
            #v1: 5个色块
            #[0xF1, 0xEE, 0x3D], [0xF1, 0xEE, 0x3D],  #对应十进制的颜色 R:241, G:238, B:61 （黄色： HEX #f1ee3d） #v1
            #[0xDC, 0x35, 0x2F], [0xDC, 0x35, 0x2F],  #对应十进制的颜色 R:220, G:53, B:47 （红色： HEX #DC352F) 
            #[0xB7, 0x66, 0x9E], [0xB7, 0x66, 0x9E],
            #[0x35, 0x49, 0x8E], [0x35, 0x49, 0x8E],
            #[0x52, 0xB2, 0x44], [0x52, 0xB2, 0x44],
            
            #v2: 10个色块
            [0xDB, 0xDB, 0x2E], [0xDB, 0xDB, 0x2E], #黄
            [0xDB, 0x2E, 0x2E], [0xDB, 0x2E, 0x2E], #红
            [0x8A, 0xDB, 0x2E], [0x8A, 0xDB, 0x2E], #淡绿
            [0xDB, 0x7F, 0x2E], [0xDB, 0x7F, 0x2E], #橙
            [0x10, 0x87, 0x16], [0x10, 0x87, 0x16], #深绿
            [0xAA, 0x2A, 0xFA], [0xAA, 0x2A, 0xFA], #紫          
            [0x95, 0xCC, 0xBB], [0x95, 0xCC, 0xBB], #深兰
            [0xFC, 0x32, 0xF9], [0xFC, 0x32, 0xF9], #粉
            [0x00, 0xFA, 0xFA], [0x00, 0xFA, 0xFA], #淡兰
            [0x3F, 0x2A, 0xFA], [0x3F, 0x2A, 0xFA], #深蓝
                      
        ], dtype=np.float32) / 0xFF
        #value_list = np.array([2.5, 2.999, 3, 3.499, 3.5, 3.999, 4, 4.499, 4.5, 4.999], dtype=np.float32)  #v1
        value_list = np.array([2.500, 2.749, 2.750, 2.999, 3.000, 3.249, 3.250, 3.499, 3.500, 3.749, 3.750, 3.999, 4.000, 4.249, 4.250, 4.499, 4.500, 4.749, 4.750, 4.999], dtype=np.float32)
        SetColorList(env, color_list, value_list)


def ShowForce(msg: IncomingMessage):
    global show_force, env
    show_force = msg.read_bool()
    SetShowForce(env, show_force)


def set_value_thread(env, q: queue.Queue):
    global show_force, conversion_params
    print("starting set_value_thread")

    while True:
        if q.empty():
            env.step()
        else:
            value = q.get()
            if show_force:
                SetValue(env, nd_voltage=value, nd_force=voltage2force(value, params=conversion_params))
            else:
                SetValue(env, nd_voltage=value, nd_force=value)


def save_thread(q: queue.Queue):
    tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not osp.exists("log"):
        os.makedirs("log", exist_ok=True)
    f_handle_1 = open(osp.join("log", tag + '.jsonl'), "w")
    f_handle_2 = open(osp.join("log", tag + '[20,18].txt'), "w")

    while True:
        if q.empty():
            time.sleep(0.02)
        else:
            res = q.get()

            f_handle_1.write(json.dumps(res) + "\n")
            f_handle_2.write(str(res['data'][20][18]) + "\n")

            f_handle_1.flush()
            f_handle_2.flush()


def custom_mapping_fn(x: np.ndarray):
    x[6][37] = x[6][38]
    return x


# MIN = 0
# MAX = 5.1

def main(args):
    global conversion_params, env
    # 启动窗口
    env = RFUniverseBaseEnv()
    # 注册UI切换回调，切换GUI上的Convert时，show_force参数随之切换
    env.AddListener("ShowForce", ShowForce)
    # 设置时间步长，单位秒，影响刷新率，与手套刷新率保持一致：1/0.077≈13
    env.SetTimeStep(0.02)

    opt = Config.load_from_yaml(args.config)
    # show_plot = args.plot

    # Open file / serial port
    serial = Serial(port=opt.dev.port, baudrate=opt.dev.baudrate, timeout=opt.dev.timeout)

    # Init parser
    P = Parser()
    P.begin(serial)
    it = P.get_iterator()

    # Init queue
    q1 = queue.Queue()
    q2 = queue.Queue()

    # Init thread
    t1 = threading.Thread(target=set_value_thread, args=(env, q1), daemon=True)
    t1.start()

    t2 = threading.Thread(target=save_thread, args=(q2,), daemon=True)
    t2.start()

    # Init Calibration
    M = ParameterManager()
    M.read_from_csv(args.calibration)
    conversion_params = M.params
    print("---------------------------")
    print("conversion_params:")

    print("---------------------------")

    # Init Amplifier
    amp = AmplifierManager()
    amp.read_from_csv(args.amplifier)
    K = amp.get()

    try:
        with tqdm.tqdm(total=0) as pbar:
            for res in it:
                if res is None:
                    print("parser error")
                    raise KeyboardInterrupt
                pbar.update(len(res["data"]))
                pbar.set_description(f"pressure=(vis:{q1.qsize()},save:{q2.qsize()})")

                res["data"] = [custom_mapping_fn(_x) * K for _x in res["data"]]

                q1.put(res["data"][-1])
                [q2.put({'data': res["data"][_i].tolist(), 'index': int(res['index'][_i]), 'ts': int(res['ts'][_i])}) for _i in range(len(res["data"]))]
    except KeyboardInterrupt:
        P.shutdown()
        serial.close()
        return


def as_completed(args: argparse.Namespace) -> argparse.Namespace:
    if not osp.exists(args.config):
        f = plyer.filechooser.open_file(title="Select a config file", filters="*.yaml")
        if len(f) > 0:
            args.config = f[0]
        else:
            raise FileNotFoundError

    if not osp.exists(args.calibration):
        f = plyer.filechooser.open_file(title="Select a calibration file", filters="*.csv")
        if len(f) > 0:
            args.calibration = f[0]
        else:
            raise FileNotFoundError

    if not osp.exists(args.amplifier):
        f = plyer.filechooser.open_file(title="Select a amplifier file", filters="*.csv")
        if len(f) > 0:
            args.amplifier = f[0]
        else:
            raise FileNotFoundError

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./manifests/config.yaml")
    parser.add_argument("--calibration", type=str, default="./manifests/calibration.csv")
    parser.add_argument("--amplifier", type=str, default="./manifests/amplifier.csv")

    args = parser.parse_args()
    main(as_completed(args))
