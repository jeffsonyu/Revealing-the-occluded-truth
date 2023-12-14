from serial_broker.Conversion import ParameterManager

P = ParameterManager()

P.read_from_csv("../manifests/calibration.csv")

print(P._A, P._b, P._k, P._d, sep="\n")