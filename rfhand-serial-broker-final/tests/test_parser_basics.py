import os
import sys

sys.path.append(os.getcwd())

from serial_broker import Parser

if __name__ == '__main__':
    P = Parser()
    print(P.read_uint16(b'\x00\x09'))
    print(P.init_checksum())
    with open('./tests/pseudo_data.txt', 'rb') as f:
        buffer = f.read()

    checksum = P.init_checksum()
    print("tx_flag:", P.read_uint16(buffer[0:2]))
    print("data_length:", P.read_uint16(buffer[2:4]))
    print("computed_checksum:", P.apply_checksum(checksum, buffer[4:2308]))
    print("actual_checksum:", P.read_uint16(buffer[2308:2310]))

    pass
