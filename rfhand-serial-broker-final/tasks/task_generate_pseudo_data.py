import argparse
import struct
import numpy as np

def gen_pseudo_packet() -> bytes:
    fmt = "HH1152HH"
    checksum = np.uint16(0x0000)
    tx_flag = np.uint16(0xFFFF) # 0xFFFF for TX
    data_length =  np.uint16(0x900) # data_lenght = 24 * 48 * 2
    data = np.random.randint(0, 2**12, size=1152, dtype=np.uint16) # 12bit ADC data
    
    checksum ^= tx_flag
    checksum ^= data_length
    data_checksum = np.bitwise_xor.reduce(data)
    checksum ^= data_checksum
    
    buffer = struct.pack(fmt, tx_flag, data_length, *data, checksum)
    return buffer
    
    
def main(args):
    with open(args.output, 'wb') as f:
        for _ in range(args.count):
            f.write(gen_pseudo_packet())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="pseudo_data.txt")
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    main(args)