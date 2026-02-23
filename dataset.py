import numpy as np
import pandas as pd


class TabularDataset:
    def __init__(self, wide_df: pd.DataFrame, calendar: pd.DataFrame):
        self.df = wide_df
        self.cal = calendar
        self.categories = wide_df.columns

    def _last_year_progress(self, cat, date):
        last_year = date - pd.DateOffset(years=1)
        mask = (
            (self.df.index.year == last_year.year)
            & (self.df.index.month == last_year.month)
            & (self.df.index.day <= date.day)
        )
        return self.df.loc[mask, cat].sum()

    def _previous_month_total(self, cat, date):
        prev = date - pd.DateOffset(months=1)
        mask = (
            (self.df.index.year == prev.year)
            & (self.df.index.month == prev.month)
        )
        return self.df.loc[mask, cat].sum()

    def build_samples(self):
        samples = []

        for date in self.df.index:
            day = date.day
            month_mask = (
                (self.df.index.year == date.year)
                & (self.df.index.month == date.month)
            )

            for cat in self.categories:
                cumulative = self.df.loc[month_mask & (self.df.index.day <= day), cat].sum()

                remaining = self.df.loc[month_mask & (self.df.index.day > day), cat].sum()

                if remaining == 0 and day == self.df.loc[month_mask].index.day.max():
                    continue

                last7 = self.df.loc[:date].tail(7)[cat].sum()
                last14 = self.df.loc[:date].tail(14)[cat].sum()
                last28 = self.df.loc[:date].tail(28)[cat].sum()

                sample = {
                    "category": cat,
                    "month": date.month,
                    "day_of_month": day,
                    "days_left": self.cal.loc[date, "days_left"],
                    "work_days_left": self.cal.loc[date, "work_days_left"],
                    "cumulative_sales": cumulative,
                    "sales_last_7": last7,
                    "sales_last_14": last14,
                    "sales_last_28": last28,
                    "sales_same_month_lastyear": self._last_year_progress(cat, date),
                    "sales_previous_month_total": self._previous_month_total(cat, date),
                    "target": remaining,
                }

                samples.append(sample)

        return pd.DataFrame(samples)
