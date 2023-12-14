import argparse

def main(args):
    buf: str
    with open(args.input, 'r') as f:
        buf = f.read()
    arr = []
    for i in buf.split(' '):
        if i != ' ' and i != '':
            arr.append(int(i, 16))
        
    with open(args.output, 'wb') as f:
        f.write(bytearray(arr))
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)