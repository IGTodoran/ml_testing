"""
Create Individual Household Electric Power Consumption data
"""
import pandas as pd

from pathlib import Path


class HouseholdElectricPower:
    """
    Prepares the Individual Household Electric Power Consumption data, downloaded from
    https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
    """

    def __init__(self, data_path: Path):
        self.data = pd.read_csv(
            data_path, delimiter=";", usecols=["Date", "Time", "Global_active_power"], low_memory=False
        )

    def preprocess_data(self) -> tuple[pd.DataFrame, float, float]:
        """
        concatenante the "Date" and "Time" columns to create a "datetime" column
        fix the data types of the "datetime" column and the "Global_active_power" column
        """
        df = self.data.assign(
            datetime=lambda x: pd.to_datetime(x["Date"] + " " + x["Time"]),
            Global_active_power=lambda x: pd.to_numeric(x["Global_active_power"], errors="coerce"),
        )
        df = df.dropna(subset=["Global_active_power"])
        df.sort_values(by="datetime", ascending=True, inplace=True)
        df = df.set_index("datetime")
        df.drop(["Date", "Time"], axis=1, inplace=True)

        # normalize the data
        max_power = df["Global_active_power"].max()
        min_power = df["Global_active_power"].min()
        df["Global_active_power"] = (df["Global_active_power"] - min_power) / (max_power - min_power)
        return df, max_power, min_power
