import os
import sqlite3
from pathlib import Path

import alphatims
import alphatims.bruker
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
from alphatims.bruker import TimsTOF
from loguru import logger
from tqdm.auto import tqdm
from functools import lru_cache


def __iter_dict_arrays(x: dict[str, np.ndarray]):
    """Iterate over a dictionary of arrays, yielding dictionaries of values.

    This is meant to be an internal function.

    Examples
    --------
        >>> x = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
        >>> for y in __iter_dict_arrays(x):
        ...     print(y)
        {'a': 1, 'b': 4}
        {'a': 2, 'b': 5}
        {'a': 3, 'b': 6}
    """
    if not isinstance(x, dict):
        msg = f"Expected a dictionary, got {type(x)}"
        raise ValueError(msg)
    lengths = {k: len(v) for k, v in x.items()}
    if not (len({len(y) for y in x.values()}) == 1):
        msg = f"All elements should be the same length, got: {lengths}"
        raise ValueError(msg)

    for i in range(len(x[next(iter(x.keys()))])):
        yield {k: v[i] for k, v in x.items()}


def _iter_timstof_data(timstof_file: TimsTOF, *, progbar=True, safe=False):
    boundaries = [
        (start, end)
        for start, end in zip(
            timstof_file.push_indptr[:-1], timstof_file.push_indptr[1:]
        )
        if end - start >= 1
    ]
    # Note: I used to have a check here to see if end - start >= min_peaks
    # But it would check that the minimum number of peaks are present
    # per IMS push, not per scan. Which can be problematic.
    # remember ... in alphatims lingo, a scan is a full IMS ramp,
    # whilst a push is a single acquisition.
    contexts = timstof_file.convert_from_indices(
        raw_indices=[x[0] for x in boundaries],
        return_rt_values=True,
        return_quad_mz_values=True,
        return_mobility_values=True,
        return_precursor_indices=True,
        return_frame_indices=True,
        return_push_indices=True,
        return_scan_indices=True,
        raw_indices_sorted=True,
    )
    logger.info(pa.Table.from_pydict(contexts))
    # >>> pl.DataFrame(contexts)
    # shape: (7_447_578, 5)
    # ┌───────────────┬─────────────┬──────────────┬──────────────┬──────────────┐
    # │ precursor_ind ┆ rt_values   ┆ mobility_val ┆ quad_low_mz_ ┆ quad_high_mz │
    # │ ices          ┆ ---         ┆ ues          ┆ values       ┆ _values      │
    # │ ---           ┆ f32         ┆ ---          ┆ ---          ┆ ---          │
    # │ i64           ┆             ┆ f32          ┆ f32          ┆ f32          │
    # ╞═══════════════╪═════════════╪══════════════╪══════════════╪══════════════╡
    # │ 0             ┆ 0.640523    ┆ 1.371029     ┆ -1.0         ┆ -1.0         │
    # │ 0             ┆ 0.640523    ┆ 1.36996      ┆ -1.0         ┆ -1.0         │
    # │ 0             ┆ 0.640523    ┆ 1.36889      ┆ -1.0         ┆ -1.0         │
    # │ 0             ┆ 0.640523    ┆ 1.367821     ┆ -1.0         ┆ -1.0         │
    # │ …             ┆ …           ┆ …            ┆ …            ┆ …            │
    # │ 2             ┆ 1259.964782 ┆ 0.930313     ┆ 625.0        ┆ 650.0        │
    # │ 2             ┆ 1259.964782 ┆ 0.925969     ┆ 625.0        ┆ 650.0        │
    # │ 2             ┆ 1259.964782 ┆ 0.804063     ┆ 425.0        ┆ 450.0        │
    # │ 2             ┆ 1259.964782 ┆ 0.784422     ┆ 425.0        ┆ 450.0        │
    # └───────────────┴─────────────┴──────────────┴──────────────┴──────────────┘

    if safe:
        context2 = timstof_file.convert_from_indices(
            raw_indices=[x[1] - 1 for x in boundaries],
            return_rt_values=True,
            return_quad_mz_values=True,
            return_mobility_values=True,
            return_precursor_indices=True,
            return_frame_indices=True,
            return_scan_indices=True,
            return_push_indices=True,
            raw_indices_sorted=True,
        )
        for k, v in context2.items():
            if not np.all(v == contexts[k]):
                msg = f"Not all fields in the timstof context share values for {k} "
                raise ValueError(msg)

    num_pushes = len(boundaries)
    logger.info(f"Found {num_pushes} pushes")

    # TODO sort here the contexts by precursor index, then by rt
    # remember to sort the boundaries too.

    my_iter = zip(boundaries, __iter_dict_arrays(contexts))
    my_progbar = tqdm(
        my_iter,
        disable=not progbar,
        desc="Quad/rt groups",
        total=num_pushes,
        mininterval=0.2,
        maxinterval=5,
    )

    for (start, end), context in my_progbar:
        query_range = range(start, end)

        out = timstof_file.convert_from_indices(
            raw_indices=query_range,
            return_corrected_intensity_values=True,
            return_mz_values=True,
            raw_indices_sorted=True,
        )
        out = {k: v.astype(np.float32) for k, v in out.items()}
        for k, v in context.items():
            out[k] = v

        if timstof_file.acquisition_mode in {"ddaPASEF", "noPASEF"}:
            if context["precursor_indices"] == 0:
                out["precursor_mz_values"] = -1.0
                out["precursor_charge"] = -1
                out["precursor_intensity"] = -1
            else:
                context_precursor = timstof_file._precursors.iloc[
                    context["precursor_indices"] - 1
                ]
                prec_mz = context_precursor.MonoisotopicMz
                charge = context_precursor.Charge
                prec_intensity = context_precursor.Intensity
                out["precursor_intensity"] = prec_intensity
                if np.isnan(prec_mz):
                    out["precursor_mz_values"] = context_precursor.LargestPeakMz
                    # TODO: handle better missing charge states
                    out["precursor_charge"] = 2
                else:
                    out["precursor_mz_values"] = prec_mz
                    out["precursor_charge"] = charge

        yield out


def timstof_to_lance(data, output_ds, *, dia_method=None, progbar=True):
    my_iter = _iter_timstof_data(data, progbar=progbar, safe=False)
    # every 10k chunks of the iterator, concatenate and yield them

    chunk = []
    first_chunk = True
    for i, x in enumerate(my_iter):
        if dia_method is not None:
            if x["precursor_indices"] == 0:
                in_frag = __in_any_fragmentation(
                    x["mz_values"], x["scan_indices"], dia_method
                )
                x["mz_values"] = x["mz_values"][in_frag]
                x["corrected_intensity_values"] = x["corrected_intensity_values"][
                    in_frag
                ]
                if len(x["mz_values"]) == 0:
                    continue

        chunk.append(x)
        if i % 10000 == 0:
            lance.write_dataset(
                # pd.DataFrame(chunk),
                pa.RecordBatch.from_pylist(chunk),
                output_ds,
                mode="append" if not first_chunk else "overwrite",
                max_rows_per_file=512 * 512,
                max_rows_per_group=512,
            )
            first_chunk = False
            chunk = []
    if chunk:
        lance.write_dataset(
            # pd.DataFrame(chunk),
            pa.RecordBatch.from_pylist(chunk),
            output_ds,
            mode="append" if not first_chunk else "overwrite",
        )


def convert_bruker(bruker_directory: os.PathLike, output_ds: os.PathLike):
    # bruker_d_folder_name = "data/3817_TIMS2_2col-80m_37_1_Slot1-46_1_4768.d"
    # out_ds = "data/3817_TIMS2_2col-80m_37_1_Slot1-46_1_4768.lance"
    data = alphatims.bruker.TimsTOF(bruker_directory)
    dia_method = get_dia_method_df(Path(bruker_directory) / "analysis.tdf")
    timstof_to_lance(data, output_ds=output_ds, dia_method=dia_method)


def get_dia_method_df(tdf_path: os.PathLike) -> pd.DataFrame:
    """Get the DIA method from a TDF file.

    Parameters
    ----------
    tdf_path : os.PathLike
        Path to the TDF file.

    Returns
    -------
    pd.DataFrame
        A dataframe with the DIA method.
    """
    con = sqlite3.connect(tdf_path)
    try:
        out = pd.read_sql("SELECT * FROM DiaFrameMsMsWindows", con)
    finally:
        con.close()
    """
    WindowGroup  ScanNumBegin  ScanNumEnd  IsolationMz  IsolationWidth  CollisionEnergy
              1           182         226      937.395          125.21        42.697154
              1           226         270      913.180          173.56        41.534553
              1           270         315      891.895          215.05        40.371951
              1           315         359      847.130          209.54        39.182927
              1           359         403      796.115          190.49        38.020325
            ...           ...         ...          ...             ...              ...
              5           624         668      556.540          165.58        31.018293
              5           668         712      512.925          170.11        29.855691
              5           712         756      478.430          155.06        28.693089
              5           756         800      460.055          119.71        27.530488
              5           800         844      442.070           83.74        26.367886
    """
    return out


def __in_any_fragmentation_scan_params(scan: int, method_df: pd.DataFrame):
    method_subset = method_df[
        (method_df["ScanNumBegin"] <= scan) & (method_df["ScanNumEnd"] >= scan)
    ]
    if method_subset.empty:
        return None, None

    half_iso_width = method_subset["IsolationWidth"].values / 2
    iso_win_start = method_subset["IsolationMz"].values - half_iso_width
    iso_win_start = np.min(iso_win_start)
    iso_win_end = method_subset["IsolationMz"].values + half_iso_width
    iso_win_end = np.max(iso_win_end)
    return iso_win_start, iso_win_end


def __in_any_fragmentation(x: np.ndarray, scan: int, method_df: pd.DataFrame):
    # Check in an array of mzs if any of them are in the DIA method.
    # it is meant to be used to drop all peaks that are never fragmented.
    iso_win_start, iso_win_end = __in_any_fragmentation_scan_params(scan, method_df)
    if iso_win_start is None:
        return np.zeros_like(x, dtype=bool)
    return (x >= iso_win_start) & (x <= iso_win_end)
