import os
import sys
import time

import cv2

sys.path.append(os.getcwd())

from serial_broker import Parser

DEBUG_PLOTTING = True
if __name__ == '__main__':
    # Init parser
    P = Parser()
    READ_LEN = 128  # FOR SIMULATION ONLY

    # Open file / serial port
    f = open('./tests/output.bin', 'rb')

    start_time = time.time()

    res = []
    while True:
        # Read from file / serial port
        buffer = f.read(READ_LEN)
        # Set break condition
        if buffer == b'':
            break

        # Submit readbufer to parser
        parsed_data = P.submit_buffer(buffer)

        # Do what ever you want with parsed data, here we just collect it
        res.extend(parsed_data)
        if len(parsed_data) == 0:
            continue
        elif DEBUG_PLOTTING:
            im = P.draw_data(P.data2voltage(parsed_data[0]))
            cv2.imshow('Parsed Matrix', im)
            cv2.waitKey(1)

    end_time = time.time()

    print(res)
    print("Time elapsed:", end_time - start_time)

    # res = [P.data2voltage(x) for x in res]
    # for i, data_set in enumerate(res):
    #     plt.figure()
    #     ax = sns.heatmap( data_set , linewidth = 0.5 , cmap = 'coolwarm' )
    #     plt.title( "2-D Heat Map" )
    #     # plt.savefig(fr"C:\Users\jiang\Desktop\testFigure\{i}.png")
    #     plt.show()

    f.close()
