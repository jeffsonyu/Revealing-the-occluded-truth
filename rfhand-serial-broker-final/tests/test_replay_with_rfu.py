import argparse
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
from serial_broker.utils import SetValue, SetColorList
from pyrfuniverse.side_channel.side_channel import (
    IncomingMessage,
)
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv

show_force = False
conversion_params: Optional[VoltageToForceParameters] = None
env: Optional[RFUniverseBaseEnv] = None


def SetShowForce(env, local_show_force: bool):
    if local_show_force:
        # 设置颜色列表，支持多种颜色线性差值，传入n*3的ndarray颜色列表。示例：1-10 黑色->白色
        color_list = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        value_list = np.array([0, 20], dtype=np.float32)
        SetColorList(env, color_list, value_list)
    else:
        # 设置颜色列表，支持多种颜色线性差值，传入n*3的ndarray颜色列表。示例：1->2.99999999->3->5 绿色->黄色->红色->紫色
        color_list = np.array([
            [0xF1, 0xEE, 0x3D], [0xF1, 0xEE, 0x3D],
            [0xDC, 0x35, 0x2F], [0xDC, 0x35, 0x2F],
            [0xB7, 0x66, 0x9E], [0xB7, 0x66, 0x9E],
            [0x35, 0x49, 0x8E], [0x35, 0x49, 0x8E],
            [0x52, 0xB2, 0x44], [0x52, 0xB2, 0x44],
            # [0xFF, 0xFF, 0x00], [0xFF, 0xFF, 0x00]
        ], dtype=np.float32) / 0xFF
        value_list = np.array([2.5, 2.999, 3, 3.499, 3.5, 3.999, 4, 4.499, 4.5, 4.999], dtype=np.float32)
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
            if value is None:
                return
            if show_force:
                SetValue(env, nd_voltage=value, nd_force=voltage2force(value, params=conversion_params))
            else:
                SetValue(env, nd_voltage=value, nd_force=value)


def custom_mapping_fn(x: np.ndarray):
    x[6][37] = x[6][38]
    return np.array(x)


def main(args):
    global conversion_params, env
    # 启动窗口
    env = RFUniverseBaseEnv()
    # 注册UI切换回调，切换GUI上的Convert时，show_force参数随之切换
    env.AddListener("ShowForce", ShowForce)
    # 设置时间步长，单位秒，影响刷新率，与手套刷新率保持一致：1/0.077≈13
    env.SetTimeStep(0.02)

    # Init queue
    q1 = queue.Queue()

    # Init thread
    t1 = threading.Thread(target=set_value_thread, args=(env, q1), daemon=True)
    t1.start()

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

    last_frame_t = None
    last_update_t = time.time_ns()
    try:
        with tqdm.tqdm(total=0) as pbar:
            with open(args.log, "r") as f:
                while True:
                    jline = f.readline()
                    if jline is None or jline == '':
                        print("EOF")
                        raise KeyboardInterrupt

                    res = json.loads(jline)

                    last_frame_t = res["ts"] if last_frame_t is None else last_frame_t
                    time.sleep(max(0, ((res["ts"] - last_frame_t) * 1 / args.speed - (time.time_ns() - last_update_t))) * 1e-9)
                    last_frame_t = res["ts"]
                    last_update_t = time.time_ns()

                    pbar.update()
                    pbar.set_description(f"pressure=(vis:{q1.qsize()})")

                    res["data"] = custom_mapping_fn(res["data"]) * K

                    q1.put(res["data"])
    except KeyboardInterrupt:
        pass

    finally:
        q1.put(None)
        t1.join()
        env.close()


def as_completed(args: argparse.Namespace) -> argparse.Namespace:
    curr_dir = os.getcwd()
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

    if args.log is None or not osp.exists(args.log):
        f = plyer.filechooser.open_file(title="Select a log file", filters=["*.jsonl"])
        if len(f) > 0:
            args.log = f[0]
        else:
            raise FileNotFoundError
    os.chdir(curr_dir)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=str, default="./manifests/calibration.csv")
    parser.add_argument("--amplifier", type=str, default="./manifests/amplifier.csv")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--speed", type=float, default=1.)

    args = parser.parse_args()
    main(as_completed(args))
