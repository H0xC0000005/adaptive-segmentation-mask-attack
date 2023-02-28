import typing
import numpy as np
import pandas as pd


class StatsLogger:
    def __init__(self):
        self.log_dict: dict[str, list]
        self.log_dict = {}

    def log_variable(self,
                     variable_name: str,
                     value: typing.Any):
        # only support same type logging for a variable
        if variable_name not in self.log_dict:
            self.log_dict[variable_name] = [value]
        else:
            if not isinstance(value, type(self.log_dict[variable_name][-1])):
                raise TypeError(f"in stats logger, trying to log type {type(value)} into "
                                f"variable type {type(self.log_dict[variable_name][-1])}")
            else:
                self.log_dict[variable_name].append(value)

    def export_ndarray(self,
                       export_variables: typing.Collection[str]) -> np.ndarray:
        nested_arr = []
        length_check = None
        for var_name in export_variables:
            cur_slice: list
            cur_slice = self.log_dict[var_name]
            if length_check is None:
                length_check = len(cur_slice)
            else:
                if len(cur_slice) != length_check:
                    raise ValueError(f"in export ndarr, got following input with len {len(cur_slice)} and name "
                                     f"{var_name} that different from previous len {length_check}")
                else:
                    nested_arr.append(cur_slice)
        arr = np.array(nested_arr)
        return arr

    def save_variables(self,
                       saved_variables: typing.Collection[str],
                       save_path: str) -> None:
        arr = self.export_ndarray(export_variables=saved_variables)
        # Save array to CSV
        np.savetxt(save_path, arr, delimiter=',', header=','.join(saved_variables), comments='')


