import numpy as np

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
