import argparse
import os
import sys

import cv2
import numpy as np
import tqdm

sys.path.append(os.getcwd())

from serial import Serial
from serial_broker import Config, Parser

MIN = 0
MAX = 5


def plot_once(im: np.ndarray, v_min=None, v_max=None):
    if v_min is None or v_max is None:
        v_min, v_max = im.min(), im.max()
    else:
        v_min, v_max = max(im.min(), v_min), min(im.max(), v_max)

    im = im - v_min
    im = im / (v_max - v_min) * 255
    im = im.astype(np.uint8)
    im = cv2.resize(im, (480, 240), interpolation=cv2.INTER_NEAREST)
    im = cv2.applyColorMap(cv2.convertScaleAbs(im, alpha=1), cv2.COLORMAP_JET)
    cv2.imshow('Converted Data', im)


def main(args):
    opt = Config.load_from_yaml(args.config)
    show_plot = args.plot

    # Open file / serial port
    serial = Serial(port=opt.dev.port, baudrate=opt.dev.baudrate, timeout=opt.dev.timeout)
    # Init parser
    P = Parser()
    P.begin(serial)
    it = P.get_iterator()

    try:
        with tqdm.tqdm(total=0) as pbar:
            for res in it:
                if res is None:
                    print("parser error")
                    raise KeyboardInterrupt
                pbar.update(len(res["data"]))
                if show_plot:
                    im = res["data"][-1]
                    # im = P.draw_data(converted_data)
                    plot_once(im)
                    cv2.waitKey(1)
    except KeyboardInterrupt:
        P.shutdown()
        serial.close()
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./manifests/config.yaml")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    main(args)
