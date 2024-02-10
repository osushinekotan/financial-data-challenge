# type: ignore


import re

import numpy as np
import pandas as pd
import rootutils

rootutils.setup_root(search_from="../", indicator=".project-root", pythonpath=True)

from src.experiment.base import BaseFeatureExtractor


class TermExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df["Term_bins_01"] = pd.cut(input_df["Term"], bins=[-1, 50, 100, 150, 200, 250, 999], labels=False)
        output_df["LogTerm"] = np.log1p(input_df["Term"])

        return output_df


class DisbursementDateExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        ts = pd.to_datetime(input_df["DisbursementDate"])

        output_df = pd.DataFrame()
        output_df = output_df.assign(
            DisbursementDate_year=ts.dt.year,
            DisbursementDate_month=ts.dt.month,
            DisbursementDate_day=ts.dt.day,
        )

        output_df["DisbursementDate_ym"] = ts.dt.strftime("%Y%m").astype(float)
        return output_df


class ApprovalDateExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        ts = pd.to_datetime(input_df["ApprovalDate"])

        output_df = pd.DataFrame()
        output_df = output_df.assign(
            ApprovalDate_year=ts.dt.year,
            ApprovalDate_month=ts.dt.month,
            ApprovalDate_day=ts.dt.day,
        )

        output_df["ApprovalDate_ym"] = ts.dt.strftime("%Y%m").astype(float)
        return output_df


class DateFeatureExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        approval_date = pd.to_datetime(input_df["ApprovalDate"])
        disbursement_date = pd.to_datetime(input_df["DisbursementDate"])

        output_df = pd.DataFrame()
        output_df["diff_date"] = (disbursement_date - approval_date).dt.days
        output_df["diff_date_minus_dummpy"] = output_df["diff_date"] < 0

        return output_df


class DisbursementGrossExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df["DisbursementGross_log"] = np.log1p(input_df["DisbursementGross"])

        return output_df


class GrAppvExtractorV1(DisbursementGrossExtractorV1):
    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df["GrAppv_log"] = np.log1p(input_df["GrAppv"])

        return output_df


class SBA_AppvExtractorV1(DisbursementGrossExtractorV1):
    def transform(self, input_df):
        output_df = pd.DataFrame()
        output_df["SBA_Appv_log"] = np.log1p(input_df["SBA_Appv"])

        return output_df


class RevLineCrExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        mapping = {"N": 0, "Y": 1, "T": 1, "0": 0}  # T, 0 ???
        output_df = pd.DataFrame()
        output_df["RevLineCr_dummy"] = [mapping[x] if x in mapping else x for x in input_df["RevLineCr"].values]

        return output_df.astype(float)


class LowDocExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df):
        dummpy_mapping = {"N": 0, "Y": 1, "T": 1, "0": 0}  # T, 0 ???
        output_df = pd.DataFrame()
        output_df["LowDoc_dummy"] = input_df["LowDoc"].map(dummpy_mapping, na_action="ignore")

        label_mapping = {
            "N": 0,
            "Y": 1,
            "0": 0,
            "A": 2,
            "S": 3,
            "C": 4,
        }  # 0, A, S ,C ???
        output_df["LowDoc_label"] = input_df["LowDoc"].map(label_mapping, na_action="ignore")
        return output_df.astype(float)


class EmployeeMoneyFeatureExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()

        # Employee
        output_df["Create_plus_RetainedJob"] = input_df["CreateJob"] + input_df["RetainedJob"]
        output_df["NoEmpALL"] = input_df["CreateJob"] + input_df["RetainedJob"] + input_df["NoEmp"]

        output_df["CreateJob_pre_logTerm"] = input_df["CreateJob"] / np.log1p(input_df["Term"])
        output_df["RetainedJob_pre_logTerm"] = input_df["RetainedJob"] / np.log1p(input_df["Term"])
        output_df["CreateJob_pre_Term"] = input_df["CreateJob"] / (input_df["Term"] + 1)
        output_df["RetainedJob_pre_Term"] = input_df["RetainedJob"] / (input_df["Term"] + 1)

        output_df["CreateJob_pre_NoEmp"] = input_df["CreateJob"] / (input_df["NoEmp"] + 1)
        output_df["RetainedJob_pre_NoEmp"] = input_df["RetainedJob"] / (input_df["NoEmp"] + 1)

        # Money
        output_df["DisbursementGross_pre_logTerm"] = input_df["DisbursementGross"] / np.log1p(input_df["Term"])
        output_df["DisbursementGross_pre_Term"] = input_df["DisbursementGross"] / (input_df["Term"] + 1)
        output_df["DisbursementGross_diff_GrAppv"] = input_df["GrAppv"] - input_df["DisbursementGross"]
        output_df["DisbursementGross_diff_SBA_Appv"] = input_df["SBA_Appv"] - input_df["DisbursementGross"]
        output_df["DisbursementGross_ratio_GrAppv"] = (
            output_df["DisbursementGross_diff_GrAppv"] / input_df["DisbursementGross"]
        )
        output_df["DisbursementGross_ratio_SBA_Appv"] = (
            output_df["DisbursementGross_diff_SBA_Appv"] / input_df["DisbursementGross"]
        )

        # Employee + Money
        output_df["DisbursementGross_pre_NoEmp"] = input_df["DisbursementGross"] / (input_df["NoEmp"] + 1)
        output_df["DisbursementGross_pre_CreateJob"] = input_df["DisbursementGross"] / (input_df["CreateJob"] + 1)
        output_df["DisbursementGross_pre_RetainedJob"] = input_df["DisbursementGross"] / (input_df["RetainedJob"] + 1)

        output_df["DisbursementGross_ratio_GrAppv_pre_NoEmp"] = output_df["DisbursementGross_diff_GrAppv"] / (
            input_df["NoEmp"] + 1
        )
        output_df["DisbursementGross_ratio_SBA_Appv_pre_NoEmp"] = output_df["DisbursementGross_diff_SBA_Appv"] / (
            input_df["NoEmp"] + 1
        )
        output_df["DisbursementGross_ratio_GrAppv_pre_CreateJob"] = output_df["DisbursementGross_diff_GrAppv"] / (
            input_df["CreateJob"] + 1
        )
        output_df["DisbursementGross_ratio_SBA_Appv_pre_CreateJob"] = output_df["DisbursementGross_diff_SBA_Appv"] / (
            input_df["CreateJob"] + 1
        )
        output_df["DisbursementGross_ratio_GrAppv_pre_RetainedJob"] = output_df["DisbursementGross_diff_GrAppv"] / (
            input_df["RetainedJob"] + 1
        )
        output_df["DisbursementGross_ratio_SBA_Appv_pre_RetainedJob"] = output_df["DisbursementGross_diff_SBA_Appv"] / (
            input_df["RetainedJob"] + 1
        )

        return output_df.astype(float)


class UrbanRuralFeatureExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()
        mapping = {0: np.nan, 1: 0, 2: 1}

        output_df["UrbanRural_dummy"] = input_df["UrbanRural"].map(mapping)
        return output_df.astype(float)


class NewExistFeatureExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()
        output_df["NewExist_dummy"] = input_df["NewExist"] - 1
        return output_df.astype(float)


class StateFeatureExtractorV1(BaseFeatureExtractor):
    def transform(self, input_df: pd.DataFrame) -> pd.DataFrame:
        output_df = pd.DataFrame()
        output_df["same_State_BankState_dummy"] = input_df["State"] == input_df["BankState"]  # TODO: nan
        return output_df.astype(float)
