import struct
import threading
import time
from io import BytesIO
from typing import List, Tuple, BinaryIO, Union, Optional

import cv2
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from serial import Serial


class Parser:
    TX_FLAG: np.uint16 = np.uint16(0xFFFF)
    DATA_LENGTH: np.uint16 = np.uint16(0x900)
    FMT: str = "HH1152HH"
    PACKET_LENGTH: int = struct.calcsize(FMT)
    SHAPE: Tuple[int, int] = (24, 48)
    RING_BUFFER_LEN: int = 4096

    buffer: BinaryIO = BytesIO()
    fig: plt.figure = None

    def __init__(self) -> None:
        self.fig = plt.figure()
        self._is_running = False
        self._running_thread: threading.Thread = None
        self._stop_ev: threading.Event = threading.Event()
        self._reset_ring_buffer()
        pass

    def __del__(self):
        self.shutdown()

    @property
    def is_running(self):
        return self._is_running

    @property
    def last_ring_buffer_idx(self):
        return max(0, self._ring_buffer_pointer - 1)

    @classmethod
    def read_uint16(cls, data: bytes) -> np.uint16:
        assert len(data) == 2, "length of data is not 2 bytes"
        x = struct.unpack("H", data)
        return np.uint16(x[0])

    @classmethod
    def init_checksum(cls) -> np.uint16:
        return np.uint16(0x0000)

    @classmethod
    def apply_checksum(cls, checksum: np.uint16, data: bytes) -> int:
        assert len(data) % 2 == 0, "length of data is not even"
        arr = np.frombuffer(data, dtype=np.uint16)
        arr_checksum = np.bitwise_xor.reduce(arr)
        return checksum ^ arr_checksum

    @classmethod
    def verify_packet(cls, data: bytes):
        checksum = cls.init_checksum()
        checksum = cls.apply_checksum(checksum, data[:-2])  # data
        return checksum == cls.read_uint16(data[-2:])

    @staticmethod
    def data2voltage(data: np.ndarray, m_v_ref: float = 2.51):
        # _tmp = np.vectorize(lambda x: x & 0x0FFF, data)
        _tmp = np.bitwise_and(data, 0x0FFF).astype(np.float64)
        _tmp /= 4095
        return _tmp * m_v_ref * 2

    @classmethod
    def parse_packet(cls, data: bytes) -> Union[None, np.uint16]:
        if not cls.verify_packet(data):
            return None
        else:
            hand = struct.unpack(cls.FMT, data)
            hand_np = np.array(hand[2:-1], dtype=np.uint16).reshape(cls.SHAPE)
            return hand_np

    def _reset_ring_buffer(self):
        self._ring_buffer: np.ndarray = np.ndarray(shape=(self.RING_BUFFER_LEN, *self.SHAPE), dtype=np.float64)
        self._ring_buffer_ts: np.ndarray = np.ndarray(shape=(self.RING_BUFFER_LEN,), dtype=np.uint64)
        self._ring_buffer_pointer: int = np.array([0], dtype=np.int64)

    def _sync(self, buffer: bytes) -> Tuple[bool, List[Union[None, np.ndarray]]]:
        if buffer is not None:
            self.buffer.write(buffer)

        # Reset buffer to head
        self.buffer.seek(0)

        while True:
            byte_pair = self.buffer.read(2)
            if len(byte_pair) != 2:
                return False, []
            if self.read_uint16(byte_pair) == self.TX_FLAG:
                self.buffer.seek(self.buffer.tell() - 2)
                self.buffer = BytesIO(self.buffer.read())  # Drop all data before TX_FLAG
                break
            else:
                self.buffer.seek(self.buffer.tell() - 1)

        # Early return if buffer is not long enough
        if len(self.buffer.getbuffer()) - self.buffer.tell() < self.PACKET_LENGTH:
            self.buffer.read()
            return False, []

        res = []
        while True:
            pkt = self.buffer.read(self.PACKET_LENGTH)
            if len(pkt) < self.PACKET_LENGTH:
                self.buffer = BytesIO(pkt)
                break
            elif not self.verify_packet(pkt):
                self.buffer = BytesIO(self.buffer.read())
                break
            else:
                res.append(self.parse_packet(pkt))

        self.buffer.read()  # move pointer to end
        return len(res) > 0, res

    def submit_buffer(self, buffer: bytes) -> List[Union[None, np.ndarray]]:
        if self._is_running:
            return None
        else:
            suceess, res = self._sync(buffer)
            while self.buffer.tell() >= self.PACKET_LENGTH:
                suceess, ret = self._sync(None)
                if suceess:
                    res.extend(ret)
                else:
                    break

            return res

    def begin(self, serial: Serial) -> Optional[Exception]:
        if self._is_running:
            return Exception("Parser is already running")

        self._reset_ring_buffer()

        def _run():
            while True:
                start_t = time.time()
                if self._stop_ev.is_set():
                    break
                # Read from file / serial port
                buffer = serial.read_all()

                # Submit readbufer to parser
                suceess, res = self._sync(buffer)
                while self.buffer.tell() >= self.PACKET_LENGTH:
                    suceess, ret = self._sync(None)
                    if suceess:
                        res.extend(ret)
                    else:
                        break

                for item in res:
                    self._ring_buffer[self._ring_buffer_pointer % self.RING_BUFFER_LEN] = self.data2voltage(item)
                    self._ring_buffer_ts[self._ring_buffer_pointer % self.RING_BUFFER_LEN] = time.time_ns()
                    self._ring_buffer_pointer += 1

                sleep_t = max([0, 0.04 + start_t - time.time()])
                time.sleep(sleep_t)

        self._running_thread = threading.Thread(group=None, target=_run, args=())
        self._stop_ev.clear()
        self._running_thread.start()
        self._is_running = True
        return None

    def shutdown(self) -> Optional[Exception]:
        if not self._is_running:
            return Exception("Parser is not running")
        self._stop_ev.set()
        self._running_thread.join()
        self._running_thread = None
        self._is_running = False
        return None

    def peek(self, index: int = None) -> Tuple[np.ndarray, np.uint64, np.int64]:
        _idx = int(self.last_ring_buffer_idx) if index is None else index
        return self._ring_buffer[_idx % self.RING_BUFFER_LEN], self._ring_buffer_ts[_idx % self.RING_BUFFER_LEN], _idx

    def get_iterator(self, timeout=10):
        last_idx = self.last_ring_buffer_idx

        while True:
            res = {"data": [], "ts": [], "index": []}

            start_t = time.time()
            while last_idx >= self.last_ring_buffer_idx:
                time.sleep(0.05)
                if time.time() - start_t > timeout:
                    return None

            next_last_idx = self.last_ring_buffer_idx

            for curr_idx in range(int(last_idx), int(next_last_idx)):
                curr_data, curr_ts, _ = self.peek(curr_idx)
                res["data"].append(curr_data)
                res["ts"].append(curr_ts)
                res["index"].append(curr_idx)

            last_idx = next_last_idx
            yield res

    def draw_data(self, data: np.ndarray):
        ax = sns.heatmap(data, linewidth=0.5, cmap='coolwarm')
        with BytesIO() as buff:
            self.fig.savefig(buff, format='raw')
            buff.seek(0)
            im_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        im = im_data.reshape((int(h), int(w), -1))
        im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
        plt.clf()
        return im
