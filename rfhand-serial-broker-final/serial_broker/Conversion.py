import csv
import dataclasses
from typing import Dict, Any, Union

import numpy as np


@dataclasses.dataclass
class VoltageToForceParameters:
    A: Union[float, np.ndarray]
    b: Union[float, np.ndarray]
    k: Union[float, np.ndarray]
    d: Union[float, np.ndarray]


def force2voltage(force: np.ndarray, params: VoltageToForceParameters) -> np.ndarray:
    res = params.A - (params.A - params.b) * np.exp(-np.power(params.k * force, params.d))
    res[np.isnan(res)] = 0
    return res


def voltage2force(voltage: np.ndarray, params: VoltageToForceParameters) -> np.ndarray:
    res = 1 / params.k * np.power(-np.log((params.A - voltage) / (params.A - params.b)), 1 / params.d)
    res[np.isnan(res)] = 0
    return res


class ParameterManager:
    __SHAPE__ = (24, 48)  # (X, Y)
    _A: np.ndarray = np.zeros(__SHAPE__)
    _b: np.ndarray = np.zeros(__SHAPE__)
    _k: np.ndarray = np.zeros(__SHAPE__)
    _d: np.ndarray = np.zeros(__SHAPE__)

    def __init__(self):
        pass

    def read_from_csv(self, path: str):
        try:
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    start_x = int(row['start_x'])
                    start_y = int(row['start_y'])
                    end_x = int(row['end_x'])
                    end_y = int(row['end_y'])
                    self._A[start_x:end_x, start_y:end_y] = float(row['A'])
                    self._b[start_x:end_x, start_y:end_y] = float(row['b'])
                    self._k[start_x:end_x, start_y:end_y] = float(row['k'])
                    self._d[start_x:end_x, start_y:end_y] = float(row['d'])
        except Exception as e:
            print("error reading from csv file")
            raise e

    @property
    def dict_params(self) -> Dict[str, Any]:
        return {
            'A': self._A,
            'b': self._b,
            'k': self._k,
            'd': self._d
        }

    @property
    def params(self) -> VoltageToForceParameters:
        return VoltageToForceParameters(A=self._A, b=self._b, k=self._k, d=self._d)


class AmplifierManager:
    __SHAPE__ = (24, 48)  # (X, Y)
    _K: np.ndarray = np.zeros(__SHAPE__)

    def __init__(self):
        pass

    def read_from_csv(self, path: str):
        try:
            with open(path, "r") as csvfile:
                lines = csvfile.readlines()
                assert len(lines) >= self.__SHAPE__[0]
                lines = lines[:self.__SHAPE__[0]]
                for i, row in enumerate(lines):
                    row_split = list(map(lambda x: float(x) if x not in ['', '\n'] else 0, row.split(",")))
                    assert len(row_split) >= self.__SHAPE__[1]
                    self._K[i] = row_split[:self.__SHAPE__[1]]
        except Exception as e:
            print("error reading from csv file")
            raise e

    def get(self, x: int = None, y: int = None) -> Union[np.ndarray, float]:
        if x is None and y is None:
            return self._K
        elif x is None:
            return self._K[:, y]
        elif y is None:
            return self._K[x, :]
        else:
            return self._K[x, y]

    @property
    def dict_params(self) -> Dict[str, Any]:
        return {
            'A': self._K,
        }
