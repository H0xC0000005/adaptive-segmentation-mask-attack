from __future__ import annotations

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

    def export_dataframe(self,
                         export_variables: list | tuple) -> pd.DataFrame:
        nested_arr = []
        length_check = None
        for var_name in export_variables:
            if var_name not in self.log_dict:
                continue
            cur_slice: list
            cur_slice = self.log_dict[var_name]
            if length_check is None:
                length_check = len(cur_slice)
            else:
                if len(cur_slice) != length_check:
                    raise ValueError(f"in export ndarr, got following input with len {len(cur_slice)} and name "
                                     f"{var_name} that different from previous len {length_check}")
            nested_arr.append(cur_slice)
        df = pd.DataFrame(nested_arr)
        df = df.transpose(copy=True)
        df.columns = export_variables
        return df

    def save_variables(self,
                       saved_variables: list | tuple,
                       save_path: str) -> None:
        arr = self.export_dataframe(export_variables=saved_variables)
        # Save array to CSV
        arr.to_csv(save_path)
