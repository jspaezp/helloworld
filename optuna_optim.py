import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass
from tdf2lance.ms_representation import _grouped_std
from loguru import logger
from tdf2lance.ms_representation import PseudoScan
from psims.mzml.writer import MzMLWriter

import tempfile


import warnings

import lance
import pandas as pd

from tdf2lance.lance_ms import iter_precursor_frames, neighbor_denoise_frames
import streamlit as st
from matplotlib import pyplot as plt
import random
from typing import Literal
from tqdm.auto import tqdm

import time
import subprocess
from contextlib import contextmanager
from functools import partial
from tdf2lance.utils import StreamingMeanCalculator


@dataclass
class Timeit:
    run_name: str = "run"
    now: float = None
    run_time: float = 0

    def __post_init__(self):
        if self.now is None:
            self.now = time.monotonic()

    def __enter__(self):
        self.now = time.monotonic()
        return self

    def __exit__(self, *args):
        self.run_time = time.monotonic() - self.now
        logger.info(f"{self.run_name} took {self.run_time:.03f}s to run")


def plot_square(ax, xmin, xmax, ymin, ymax, color="red", **kwargs):
    # xs = [50, 90, 90, 50, 50]
    # ys = [100, 100, 130, 130, 100]
    # ax.plot(xs, ys, color="red")
    xs = [xmin, xmax, xmax, xmin, xmin]
    ys = [ymin, ymin, ymax, ymax, ymin]
    ax.plot(xs, ys, color=color, **kwargs)


def plot_squares(ax, xmins, xmaxs, ymins, ymaxs, color="red", **kwargs):
    for xmin, xmax, ymin, ymax in zip(xmins, xmaxs, ymins, ymaxs):
        plot_square(ax, xmin, xmax, ymin, ymax, color=color, **kwargs)


def _stats_per_label(arr, labels, unique_labels, counts, weights, aggregated_weights):
    arr = arr.astype(np.float64)

    arr_len = max(2, unique_labels.max() + 2)
    max_arr = np.zeros(arr_len)
    weighted_avg_arr = np.zeros(arr_len)

    np.maximum.at(max_arr, labels, arr)
    min_arr = max_arr.copy()
    np.minimum.at(min_arr, labels, arr)
    std_arr = _grouped_std(
        labels=labels, unique_labels=unique_labels, float_array=arr, counts=counts
    )
    np.add.at(weighted_avg_arr, labels, arr * weights)
    weighted_avg_arr /= aggregated_weights

    return {
        "max": max_arr,
        "min": min_arr,
        "std": std_arr,
        "weighted_avg": weighted_avg_arr,
    }


@dataclass
class CentroidedPrecursorIndex:
    intensities: np.ndarray
    mzs: np.ndarray
    mobilities: np.ndarray
    quad_low: np.ndarray
    quad_high: np.ndarray
    mobility_min: np.ndarray
    mobility_max: np.ndarray
    mobility_std: np.ndarray
    mz_min: np.ndarray
    mz_max: np.ndarray
    mz_std: np.ndarray
    rt: np.ndarray
    frame_index: np.ndarray

    def __post_init__(self):
        # Check that everything is the same length
        assert len(self.intensities) == len(self.mzs)
        assert len(self.intensities) == len(self.mobilities)
        assert len(self.intensities) == len(self.quad_low)
        assert len(self.intensities) == len(self.quad_high)
        assert len(self.intensities) == len(self.mobility_min)
        assert len(self.intensities) == len(self.mobility_max)
        assert len(self.intensities) == len(self.mobility_std)
        assert len(self.intensities) == len(self.mz_min)
        assert len(self.intensities) == len(self.mz_max)
        assert len(self.intensities) == len(self.mz_std)
        assert len(self.intensities) == len(self.rt)
        assert len(self.intensities) == len(self.frame_index)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return cls(
            df["intensities"].to_numpy(),
            df["mzs"].to_numpy(),
            df["mobilities"].to_numpy(),
            df["quad_low"].to_numpy(),
            df["quad_high"].to_numpy(),
            df["mobility_min"].to_numpy(),
            df["mobility_max"].to_numpy(),
            df["mobility_std"].to_numpy(),
            df["mz_min"].to_numpy(),
            df["mz_max"].to_numpy(),
            df["mz_std"].to_numpy(),
            df["rt"].to_numpy(),
            df["frame_index"].to_numpy(),
        )

    def dbscan_combine_traces(
        self,
        *,
        rt_scaling=10,
        mobility_scaling=0.01,
        quad_scaling=5,
        mz_scaling=0.01,
        dbscan_p=2,
        dbscan_min_samples=4,
        use_meanshift=False,
        keep_unclustered=True,
    ):
        from scipy.spatial import KDTree
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import MeanShift

        logger.debug("Parameters for clustering:")
        logger.debug(f"rt_scaling = {rt_scaling}")
        logger.debug(f"mobility_scaling = {mobility_scaling}")
        logger.debug(f"quad_scaling = {quad_scaling}")
        logger.debug(f"mz_scaling = {mz_scaling}")
        logger.debug(f"dbscan_p = {dbscan_p}")
        logger.debug(f"dbscan_min_samples = {dbscan_min_samples}")
        logger.debug(f"use_meanshift = {use_meanshift}")
        logger.debug(f"keep_unclustered = {keep_unclustered}")

        # This is optimized for this specific dataset ..
        # Where cycle time is 1.7s~
        # rt_scaling = 1.75 * 4
        # mobility_scaling = 0.01
        # quad_scaling = 5
        # mz_scaling = 0.02
        # dbscan_min_samples = 2
        # dbscan_p = 0.5  # How much like manhattan distance is it?

        build_arr = np.stack(
            [
                self.mzs / mz_scaling,
                self.rt / rt_scaling,
                self.mobilities / mobility_scaling,
                self.quad_low / quad_scaling,
                self.quad_high / quad_scaling,
            ],
            axis=1,
        )

        if use_meanshift:
            ms = MeanShift(
                bandwidth=1, bin_seeding=True, cluster_all=False, min_bin_freq=5
            ).fit(build_arr)
            # Misleading but api compliant ...
            dbscan = ms

        else:
            kdtree = KDTree(
                build_arr,
                leafsize=100,
                balanced_tree=False,
                copy_data=False,
                compact_nodes=False,
            )
            d = kdtree.sparse_distance_matrix(kdtree, max_distance=2, p=dbscan_p)
            dbscan = DBSCAN(
                eps=1,
                min_samples=dbscan_min_samples,
                metric="precomputed",
                p=dbscan_p,
            ).fit(d)

        uniq_labs, pos, counts = np.unique(
            dbscan.labels_, return_index=True, return_counts=True
        )
        clustered_peaks = sum(counts[uniq_labs != -1])
        unclustered_peaks = sum(counts[uniq_labs == -1])
        logger.debug(
            f"Found {len(uniq_labs)} clusters with "
            f"{clustered_peaks} clustered peaks and "
            f" {unclustered_peaks} unclustered peaks"
        )

        uniq_labs, pos, counts = np.unique(
            dbscan.labels_, return_index=True, return_counts=True
        )

        # Counts is the number of peaks in each cluster
        arr_len = max(max(uniq_labs) + 2, 2)
        b_counts = np.zeros(arr_len)
        np.add.at(b_counts, uniq_labs, counts)
        del counts  # I am deleting it for my own safety ...
        # Since these counts do not match in position with the rest
        # of the arrays. The 'counts' start with the unclustered peaks
        # whilst the b_counts have the unclustered peaks at the end (position -1).

        # Quad lows and highs of each cluster
        b_quad_lows = np.zeros(arr_len)
        b_quad_highs = np.zeros(arr_len)
        np.add.at(b_quad_lows, dbscan.labels_, self.quad_low.astype(np.float64))
        np.add.at(b_quad_highs, dbscan.labels_, self.quad_high.astype(np.float64))
        b_quad_lows /= b_counts
        b_quad_highs /= b_counts

        # Intensities of each cluster
        b_intensities = np.zeros(arr_len)
        np.add.at(
            b_intensities,
            dbscan.labels_,
            self.intensities.astype(np.float64),
        )

        # Little report ...
        clustered_intensities = b_intensities[:-1].sum()
        unclustered_intensities = b_intensities[-1]
        tot_intensities = clustered_intensities + unclustered_intensities
        logger.debug(
            f"Found {len(b_intensities)-1} clusters with "
            f"{clustered_intensities}"
            f"({100*clustered_intensities/tot_intensities:.03f}) "
            f"clustered intensity and {unclustered_intensities} unclustered intensity"
        )

        # Now we need to average the mzs, mobilities and rts of each cluster
        # It is nice to have some distribution metrics ...

        mz_stats = _stats_per_label(
            arr=self.mzs,
            labels=dbscan.labels_,
            unique_labels=uniq_labs,
            counts=b_counts,
            weights=self.intensities,
            aggregated_weights=b_intensities,
        )

        mobility_stats = _stats_per_label(
            arr=self.mobilities,
            labels=dbscan.labels_,
            unique_labels=uniq_labs,
            counts=b_counts,
            weights=self.intensities,
            aggregated_weights=b_intensities,
        )

        rt_stats = _stats_per_label(
            arr=self.rt,
            labels=dbscan.labels_,
            unique_labels=uniq_labs,
            counts=b_counts,
            weights=self.intensities,
            aggregated_weights=b_intensities,
        )

        peak_widths = rt_stats["max"] - rt_stats["min"]
        msg = f"Detected peak width stats: mean={peak_widths.mean():.03f}s"
        msg += f" std={peak_widths.std():.03f}s"
        msg += f" median={np.median(peak_widths):.03f}s"
        logger.info(msg)

        out = {
            "intensities": b_intensities,
            "mzs": mz_stats["weighted_avg"],
            "mobilities": mobility_stats["weighted_avg"],
            "quad_low": b_quad_lows,
            "quad_high": b_quad_highs,
            "mobility_min": mobility_stats["min"],
            "mobility_max": mobility_stats["max"],
            "mobility_std": mobility_stats["std"],
            "mz_min": mz_stats["min"],
            "mz_max": mz_stats["max"],
            "mz_std": mz_stats["std"],
            "rt": rt_stats["weighted_avg"],
            "rt_start": rt_stats["min"],
            "rt_end": rt_stats["max"],
            "num_agg_peaks": b_counts,
        }
        if keep_unclustered:
            # TODO consider if this can be optimized by first allocating the full size
            # of the array including the unclustered peaks.
            unclustered_indices = np.where(dbscan.labels_ == -1)[0]
            unclustered_out = {
                "intensities": self.intensities.astype(np.float64)[unclustered_indices],
                "mzs": self.mzs.astype(np.float64)[unclustered_indices],
                "mobilities": self.mobilities.astype(np.float64)[unclustered_indices],
                "quad_low": self.quad_low.astype(np.float64)[unclustered_indices],
                "quad_high": self.quad_high.astype(np.float64)[unclustered_indices],
                "rt": self.rt.astype(np.float64)[unclustered_indices],
                "num_agg_peaks": np.ones(len(unclustered_indices)),
                "mz_std": np.zeros(len(unclustered_indices)),
                "mobility_std": np.zeros(len(unclustered_indices)),
            }
            unclustered_out["mobility_min"] = unclustered_out["mobilities"]
            unclustered_out["mobility_max"] = unclustered_out["mobilities"]
            unclustered_out["mz_min"] = unclustered_out["mzs"]
            unclustered_out["mz_max"] = unclustered_out["mzs"]
            unclustered_out["rt_start"] = unclustered_out["rt"]
            unclustered_out["rt_end"] = unclustered_out["rt"]

            if not set(out.keys()) == set(unclustered_out.keys()):
                msg = (
                    "Keys of clustered and unclustered peaks do not match: "
                    f"{set(out.keys())} vs {set(unclustered_out.keys())}: "
                    f"Diff = {set(out.keys()) ^ set(unclustered_out.keys())}"
                )
                raise ValueError(msg)
            out = {k: np.concatenate([v, unclustered_out[k]]) for k, v in out.items()}

        return out, dbscan.labels_


@dataclass
class RTCentroidedPrecursorIndex:
    intensities: np.ndarray
    mzs: np.ndarray
    mobilities: np.ndarray
    quad_low: np.ndarray
    quad_high: np.ndarray
    mobility_min: np.ndarray
    mobility_max: np.ndarray
    mobility_std: np.ndarray
    mz_min: np.ndarray
    mz_max: np.ndarray
    mz_std: np.ndarray
    rt: np.ndarray
    rt_start: np.ndarray
    rt_end: np.ndarray
    num_agg_peaks: np.ndarray

    def meanshift_cluster(
        self,
        *,
        rt_scaling: float,
        mobility_scaling: float,
        quad_scaling: float,
        min_bin_freq: int,
        bandwidth: float,
        cluster_all: bool,
        use_meanshift: bool,
        expand_dist: float,
    ):
        from scipy.spatial import KDTree
        from sklearn.cluster import DBSCAN, MeanShift

        logger.debug("Parameters for clustering:")
        logger.debug(f"rt_scaling = {rt_scaling} seconds")
        logger.debug(f"mobility_scaling = {mobility_scaling}")
        logger.debug(f"quad_scaling = {quad_scaling}")
        logger.debug(f"min_bin_freq = {min_bin_freq}")
        logger.debug(f"bandwidth = {bandwidth}")
        logger.debug(f"cluster_all = {cluster_all}")
        logger.debug(f"use_meanshift = {use_meanshift}")
        logger.debug(f"expansion_dist = {expand_dist}")

        build_arr = np.stack(
            [
                self.rt / rt_scaling,
                self.mobilities / mobility_scaling,
                self.quad_low / quad_scaling,
                self.quad_high / quad_scaling,
                # self.rt_start / (rt_scaling * 2),
                # self.rt_end / (rt_scaling * 2),
                # TODO consider adding rt starts and ends here...
            ],
            axis=1,
        )

        clustering = "MeanShift" if use_meanshift else "DBSCAN"
        logger.info(f"Running {clustering} clustering (combining elution groups)...")
        if use_meanshift:
            # TODO experiment doing this iterative increasing the bandwidth.
            generation_cluster_counts = []
            ms = MeanShift(
                bandwidth=bandwidth,
                bin_seeding=True,
                # TODO consider making cluster all false
                # and then assign within threshold to that cluster...
                cluster_all=False,
                min_bin_freq=min_bin_freq,
                max_iter=200,
            ).fit(build_arr)

            logger.info("Clustering overhanding peaks...")
            unclustered_kdt = KDTree(
                build_arr[ms.labels_ == -1],
                leafsize=100,
                balanced_tree=False,
                copy_data=False,
                compact_nodes=False,
            )
            centers_kdt = KDTree(
                ms.cluster_centers_,
                leafsize=100,
                balanced_tree=False,
                copy_data=False,
                compact_nodes=False,
            )
            tmp2 = unclustered_kdt.sparse_distance_matrix(
                centers_kdt,
                max_distance=expand_dist * bandwidth,
                p=1,
                output_type="coo_matrix",
            )
            # Cols in the coo matrix are the index in the cluster centers
            # Rows are the index in the unclustered peaks that match the former
            # cluster center.
            order = np.lexsort((tmp2.row, tmp2.data))
            unclustp_indices, center_indices, dists = (
                tmp2.row[order],
                tmp2.col[order],
                tmp2.data[order],
            )
            uniq_ucpii = np.unique(unclustp_indices, return_index=True)[1]
            logger.info(f"Adding {len(uniq_ucpii)} unclustered peaks to clusters")

            unlabelled = np.where(ms.labels_ == -1)[0]
            ms.labels_[unlabelled[unclustp_indices[uniq_ucpii]]] = center_indices[
                uniq_ucpii
            ]

            ## Second round ...
            unlabelled = np.where(ms.labels_ == -1)[0]
            generation_cluster_counts.append(ms.labels_.max())
            gen = 1
            while len(unlabelled) > min_bin_freq * 2:
                try:
                    gen += 1
                    logger.info(f"Starting round {gen} of clustering ...")
                    ms2 = MeanShift(
                        bandwidth=bandwidth,
                        bin_seeding=True,
                        cluster_all=False,
                        min_bin_freq=min_bin_freq,
                        max_iter=100,
                    ).fit(build_arr[unlabelled])
                    logger.info(
                        f"Adding round {gen} of clusters,"
                        f" {ms2.labels_.max() + 1}; {np.sum(ms2.labels_ >= 0)} points ..."
                    )
                    generation_cluster_counts.append(ms2.labels_.max())
                    keep_ms2 = ms2.labels_ >= 0

                    ms.labels_[unlabelled[keep_ms2]] = (
                        ms2.labels_[keep_ms2] + ms.labels_.max() + 1
                    )
                    unlabelled = np.where(ms.labels_ == -1)[0]
                except ValueError as e:
                    if "max() arg is an empty sequence" in str(e):
                        logger.info("No unclustered peaks left to cluster")
                        break
                    else:
                        raise e
            else:
                logger.info("No unclustered peaks left to cluster")

        else:
            logger.debug("building kdtree")
            kdtree = KDTree(
                build_arr,
                leafsize=100,
                balanced_tree=False,
                copy_data=False,
                compact_nodes=False,
            )
            logger.debug("building distance matrix")
            d = kdtree.sparse_distance_matrix(
                kdtree, max_distance=1.75 * bandwidth, p=0.5
            )

            logger.debug("running dbscan")
            dbscan = DBSCAN(
                eps=bandwidth,
                min_samples=min_bin_freq,
                metric="precomputed",
                p=0.5,
            ).fit(d)
            ms = dbscan

        uniq_labs, pos, counts = np.unique(
            ms.labels_, return_index=True, return_counts=True
        )
        clustered_peaks = sum(counts[uniq_labs != -1])
        unclustered_peaks = sum(counts[uniq_labs == -1])
        logger.debug(
            f"Found {len(uniq_labs)} clusters with {clustered_peaks} "
            f"clustered peaks and {unclustered_peaks} unclustered peaks"
        )

        # use max(uniq_labs)+1 instead of len(unique_labs)
        # Counts is the number of peaks in each cluster
        arr_len = max(max(uniq_labs) + 2, 2)
        b_counts = np.zeros(arr_len)
        np.add.at(b_counts, uniq_labs, counts)
        del counts  # I am deleting it for my own safety ...
        # Since these counts do not match in position with the rest
        # of the arrays. The 'counts' start with the unclustered peaks
        # whilst the b_counts have the unclustered peaks at the end (position -1).

        # Quad lows and highs of each cluster
        b_quad_lows = np.zeros(arr_len)
        b_quad_highs = np.zeros(arr_len)
        np.add.at(b_quad_lows, ms.labels_, self.quad_low.astype(np.float64))
        np.add.at(b_quad_highs, ms.labels_, self.quad_high.astype(np.float64))
        b_quad_lows /= b_counts
        b_quad_highs /= b_counts

        # Intensities of each cluster
        b_intensities = np.zeros(arr_len)
        np.add.at(
            b_intensities,
            ms.labels_,
            self.intensities.astype(np.float64),
        )

        # Little report ...
        clustered_intensities = b_intensities[:-1].sum()
        unclustered_intensities = b_intensities[-1]
        tot_intensities = clustered_intensities + unclustered_intensities
        logger.info(
            f"Found {len(b_intensities)-1} clusters with "
            f"{clustered_intensities}"
            f" ({100*clustered_intensities/tot_intensities:.03f}) "
            f"clustered intensity and {unclustered_intensities} unclustered intensity"
        )

        # Now we need to average the mzs, mobilities and rts of each cluster
        # It is nice to have some distribution metrics ...

        mz_lists = [None] * arr_len
        for i in range(arr_len):
            mz_lists[i] = self.mzs[ms.labels_ == i]

        itensity_lists = [None] * arr_len
        for i in range(arr_len):
            itensity_lists[i] = self.intensities[ms.labels_ == i]

        mobility_stats = _stats_per_label(
            arr=self.mobilities,
            labels=ms.labels_,
            unique_labels=uniq_labs,
            counts=b_counts,
            weights=self.intensities,
            aggregated_weights=b_intensities,
        )

        rt_stats = _stats_per_label(
            arr=self.rt,
            labels=ms.labels_,
            unique_labels=uniq_labs,
            counts=b_counts,
            weights=self.intensities,
            aggregated_weights=b_intensities,
        )

        generation = np.zeros(arr_len)
        cum_gen = 0
        for x in generation_cluster_counts:
            generation[cum_gen : cum_gen + x] = generation.max() + 1

        out = {
            "intensities": itensity_lists[:-1],
            "mzs": mz_lists[:-1],
            "mobilities": mobility_stats["weighted_avg"][:-1],
            "quad_low": b_quad_lows[:-1],
            "quad_high": b_quad_highs[:-1],
            "mobility_min": mobility_stats["min"][:-1],
            "mobility_max": mobility_stats["max"][:-1],
            "mobility_std": mobility_stats["std"][:-1],
            "rt": rt_stats["weighted_avg"][:-1],
            "rt_start": rt_stats["min"][:-1],
            "rt_end": rt_stats["max"][:-1],
            "num_agg_peaks": b_counts[:-1],
            "generation": generation[:-1],
        }
        return out, ms.labels_


def _empty_scan(rt, id):
    scan_id = f"scan={id}"
    return {
        "mz_array": np.array([220]),
        "intensity_array": np.array([220]),
        "scan_start_time": rt,
        "id": scan_id,
        "params": [
            "MS1 Spectrum",
            {"ms level": 1},
            {"total ion current": 220},
        ],
    }, scan_id


def iter_scans(pseudoscans):
    # I am binning here every 0.02 dalton ... same as comet ...
    offset = 1
    curr_rt = pseudoscans["rt"][0] - 1
    empty_spec, last_precursor_scan_id = _empty_scan(curr_rt, offset)
    offset += 1

    yield empty_spec

    for i in range(len(pseudoscans["rt"])):
        mz = pseudoscans["mzs"][i]
        mz_order = np.argsort(mz)
        mz = mz[mz_order]
        intensity = pseudoscans["intensities"][i]
        intensity = intensity[mz_order]

        rt = pseudoscans["rt"][i]
        quad_low = pseudoscans["quad_low"][i]
        quad_high = pseudoscans["quad_high"][i]
        quad_mid = (quad_low + quad_high) / 2

        if rt > curr_rt + 1:
            empty_spec, last_precursor_scan_id = _empty_scan(curr_rt, i + offset)
            yield empty_spec
            curr_rt = rt
            offset += 1
        if quad_high - quad_low > 50:
            logger.warning(
                f"Quad width is {quad_high - quad_low} at RT {rt} for mz {mz}"
            )
        yield {
            "mz_array": mz,
            "intensity_array": intensity,
            "scan_start_time": rt,
            "id": f"scan={i+offset}",
            "params": [
                "MSn Spectrum",
                {"ms level": 2},
                {"total ion current": sum(intensity)},
            ],
            "precursor_information": {
                "activation": [
                    "beam-type collisional dissociation",
                    {"collision energy": 25},
                ],
                "mz": quad_mid,
                "intensity": sum(intensity),
                "charge": 2,
                "isolation_window": [quad_low, quad_mid, quad_high],
                "scan_id": last_precursor_scan_id,
            },
        }


def pseudoscans_to_mzml(pseudoscans, target_file):
    scans = list(iter_scans(pseudoscans))
    logger.info(f"Writing {len(scans)} scans to {target_file}")

    with MzMLWriter(open(str(target_file), "wb"), close=True) as out:
        # Add default controlled vocabularies
        out.controlled_vocabularies()
        out.file_description(["MS1 spectrum", "MSn spectrum"])
        # Open the run and spectrum list sections
        with out.run(id="my_analysis"):
            with out.spectrum_list(count=len(scans)):
                for products in scans:
                    # Write MSn scans
                    out.write_spectrum(**products)


@dataclass
class NeighborDenoiseConfig:
    neighbor_n: int = 1  # 3 seems good
    dbscan_n: int = 3  # 4 seems good
    mz_tolerance: float = 0.011
    mobility_tolerance: float = 0.02
    sides: Literal["left", "right", "both"] = "both"

    @staticmethod
    def from_trial(trial):
        return NeighborDenoiseConfig(
            neighbor_n=trial.suggest_int("nd_neighbor_n", 1, 3),
            dbscan_n=trial.suggest_int("nd_dbscan_n", 1, 4),
            # TODO consider redefining this as a fraction of the
            # frame centroiding tolerance.
            mz_tolerance=trial.suggest_float("nd_mz_tolerance", 0.005, 0.015),
            mobility_tolerance=trial.suggest_float("nd_mobility_tolerance", 0.01, 0.12),
            sides="both",
        )


@dataclass
class IMSCentroidConfig:
    mz_tolerance: float = 0.013  # This seems good
    mobility_tolerance: float = 0.012
    dbscan_n: int = 3  # seems good
    # keep_unclustered: bool = True
    keep_unclustered: bool = False

    @staticmethod
    def from_trial(trial):
        return IMSCentroidConfig(
            mz_tolerance=trial.suggest_float("ic_mz_tolerance", 0.012, 0.022),
            mobility_tolerance=trial.suggest_float(
                "ic_mobility_tolerance", 0.008, 0.018
            ),
            dbscan_n=trial.suggest_int("ic_dbscan_n", 2, 5),
            # keep_unclustered=True, # This is better but is ver slow ...
            keep_unclustered=False,
        )


@dataclass
class CombineTracesConfig:
    """Configuration to combine points into traces along the RT dimension."""

    rt_scaling: float = 1.06  # seems to be similar to the peak width ...
    # Now defined as a fraction of the peak width
    mobility_scaling: float = 0.021  # 0.006-0.008 seem good 80% of c_mobility_scaling
    mz_scaling: float = 0.02  # 0.004-0.007 seem good if using the dbscan workflow.
    dbscan_min_samples: int = 2  # 2-4 seem good
    dbscan_p: float = 0.5  # 0.25-0.75 seem good
    # keep_unclustered: bool = True
    keep_unclustered: bool = False

    @staticmethod
    def from_trial(trial):
        return CombineTracesConfig(
            mz_scaling=trial.suggest_float(
                "t_mz_scaling",
                0.006,
                0.022,
            ),  # TODO reparametrize as a fraction of the mz tolerance in other steps
            # TODO implement a way to infer this parameter.
            rt_scaling=trial.suggest_float("t_rt_scaling", 0.4, 1.4),
            mobility_scaling=trial.suggest_float("t_mobility_scaling", 0.009, 0.022),
            dbscan_min_samples=2,  # 1 works a lil better, but takes forever
            # dbscan_min_samples=trial.suggest_int("t_dbscan_min_samples", 1, 3),
            dbscan_p=0.5,
            keep_unclustered=False,
            # keep_unclustered=trial.suggest_categorical(
            #     "t_keep_unclustered", [True, False]
            # ),
        )


@dataclass
class ClusterTracesConfig:
    """Configuration to cluster traces"""

    rt_scaling: float = 0.09
    # Now defined as number of cycle times for dbscan but
    # as a fraction of the peak width for meanshift.
    mobility_scaling: float = (
        0.0096  # 0.002-0.01 seem good for dbscan ... 0.01 for meanshift
    )
    min_bin_freq: int = 2  # 1-5 looks good
    binwidth: float = 1
    cluster_all: bool = False
    use_meanshift: bool = True
    expand_dists: float = 0

    @staticmethod
    def from_trial(trial):
        # use_meanshift = trial.suggest_categorical("c_use_meanshift", [True, False])
        use_meanshift = True
        if use_meanshift:
            return ClusterTracesConfig(
                rt_scaling=trial.suggest_float("c_rt_scaling", 0.01, 0.25),
                mobility_scaling=trial.suggest_float(
                    "c_mobility_scaling", 0.008, 0.022
                ),
                min_bin_freq=trial.suggest_int("c_min_bin_freq", 2, 9),
                binwidth=1,
                cluster_all=False,
                use_meanshift=True,
                expand_dists=0,
            )
        else:
            return ClusterTracesConfig(
                rt_scaling=trial.suggest_float("c_rt_scaling", 0.9, 5.1),
                mobility_scaling=trial.suggest_float(
                    "c_mobility_scaling", 0.001, 0.012
                ),
                min_bin_freq=trial.suggest_int("c_min_bin_freq", 4, 15),
                binwidth=0.6,
                cluster_all=False,
                use_meanshift=False,
            )


def neighbor_denoise(
    config: NeighborDenoiseConfig,
    ims_centroid_config: IMSCentroidConfig,
    targetfile=None,
):
    out_ds = "currfile.lance"
    ds = lance.dataset(out_ds)

    precursor_index = 5
    out = []

    start = 120
    runtime = 15 * 60

    logger.info(
        "Running neighbor denoise ... On precursor"
        f" {precursor_index} start {start} runtime {runtime}"
    )

    neighbor_denoise_partial = partial(
        neighbor_denoise_frames,
        quad_tolerance=None,
        mz_tolerance=config.mz_tolerance,
        mobility_tolerance=config.mobility_tolerance,
        sides=config.sides,
        yield_noisy=False,
    )

    def _inner():
        for f in neighbor_denoise_partial(
            iter_precursor_frames(
                ds,
                precursor_index=precursor_index,
                start_rt=start,
                end_rt=runtime + runtime,
            ),
            n=config.neighbor_n,
        ):
            if sum(f.n_peaks) == 0:
                continue

            if f.rt > (start + runtime):
                break

            # suppress EfficiencyWarning warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "Precomputed sparse input was not sorted by row values."
                        " Use the function sklearn.neighbors.sort_graph_by_row_values"
                        " to sort the input by row values,"
                        " with warn_when_not_sorted=False"
                        " to remove this warning."
                    ),
                )

                # TODO consider making anothr round of neighbor simplification here ...
                # maybe a first round keeping unclustered and a second round not keeping
                asdasd = f.dbscan_sinplify_frame(
                    n=ims_centroid_config.dbscan_n,
                    mz_tolerance=ims_centroid_config.mz_tolerance,
                    mobility_tolerance=ims_centroid_config.mobility_tolerance,
                    quad_tolerance=5,
                    eps=1e-8,
                    dbscan_n=ims_centroid_config.dbscan_n,
                    keep_unclustered=ims_centroid_config.keep_unclustered,
                )
                yield asdasd

    for asdasd in neighbor_denoise_partial(_inner(), n=1):
        out.append(pd.DataFrame(asdasd.as_record()))

    out = pd.concat(out)
    if targetfile is None:
        targetfile = f"{precursor_index}_currfile.parquet"

    out.to_parquet(targetfile)


def plot_clustering(pseudoscan_labels, clustering_dict, pseudoscans_dict, outfile):
    fig, ax = plt.subplots(figsize=(25, 15))
    ax.scatter(
        clustering_dict["rt"][pseudoscan_labels == -1],
        clustering_dict["mobilities"][pseudoscan_labels == -1],
        c="black",
        s=5,
    )
    ax.scatter(
        clustering_dict["rt"],
        clustering_dict["mobilities"],
        c=np.concatenate([pseudoscans_dict["generation"], [-1]])[pseudoscan_labels],
        alpha=0.1,
        s=5,
    )
    fig.savefig("latest_generations.png", dpi=96)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(25, 15))

    plot_order = np.argsort(pseudoscan_labels)
    start_clusters = np.searchsorted(pseudoscan_labels[plot_order], 0)

    ax.set_xlabel("rt (s)")
    ax.set_ylabel("Mobility 1/K0")
    # Gray points for non clustered
    ax.scatter(
        clustering_dict["rt"],
        clustering_dict["mobilities"],
        c="black",
        alpha=0.2,
        s=5,
    )
    ax.scatter(
        clustering_dict["rt"][plot_order][:start_clusters],
        clustering_dict["mobilities"][plot_order][:start_clusters],
        c="gray",
        alpha=0.2,
        s=5,
    )
    fig.savefig("latest_unclustered.png", dpi=96)
    ax.scatter(
        clustering_dict["rt"][plot_order][start_clusters:],
        clustering_dict["mobilities"][plot_order][start_clusters:],
        c=pseudoscan_labels[plot_order][start_clusters:],
        alpha=0.2,
        s=5,
    )
    plot_squares(
        xmaxs=pseudoscans_dict["rt_end"],
        xmins=pseudoscans_dict["rt_start"],
        ymaxs=pseudoscans_dict["mobility_max"],
        ymins=pseudoscans_dict["mobility_min"],
        ax=ax,
        color="red",
        alpha=0.3,
    )

    logger.info(f"Saving clustering plot to {outfile}")
    fig.savefig(outfile, dpi=96)
    fig.savefig("latest.png", dpi=96)
    # plt.show()
    plt.close(fig)


def main(
    *,
    traces_config: CombineTracesConfig,
    cluster_config: ClusterTracesConfig,
    pq_file=None,
    plot=True,
):
    logger.info("Reading parquet file ...")
    # df = pd.read_parquet("5_3817_TIMS2_2col-80m_37_1_Slot1-46_1_4768.lance.parquet")
    if pq_file is None:
        logger.warning("No parquet file provided, using default")
        pq_file = "5_currfile.parquet"
    df = pd.read_parquet(pq_file)
    first_quad_low = df["quad_low"].min()
    unique_rts = np.sort(np.unique(df[df["quad_low"] == first_quad_low]["rt"]))
    cycle_time = np.median(np.diff(unique_rts))
    logger.info(f"Cycle time is {cycle_time:.03f}s")

    # Randomly pick 1000 peaks among the top 10% most intense peaks
    # and calculate the peak width distribution
    # logger.info("Calculating peak widths ...")
    sdf = df[df["quad_low"] == first_quad_low].sort_values(
        "intensities", ascending=False
    )
    hsdf = sdf.head(int(len(sdf) * 0.25))
    hsdf = hsdf.sample(min(1000, len(hsdf)))
    mzs = hsdf["mzs"].values
    intensities = hsdf["intensities"].values
    # Remove mzs that are very close to each other
    mz_order = np.argsort(mzs)
    mzs = mzs[mz_order]
    intensities = intensities[mz_order]

    mz_diffs = np.concatenate([np.array([0.0]), np.diff(mzs)])
    mzs = mzs[mz_diffs > 0.02]
    intensities = intensities[mz_diffs > 0.02]

    # Slow but easy version ...

    calc_widths = []
    for mz, inten in tqdm(zip(mzs, intensities)):
        sdf2 = sdf[(sdf["mzs"] > mz - 0.02) & (sdf["mzs"] < mz + 0.02)]
        # Keep only peaks more than 1% of the max intensity
        sdf2 = sdf2[(sdf2["intensities"] > 0.01 * inten)]
        rts = np.sort(sdf2["rt"].values)
        # keep only rts that are less than 3 cycles from each orther
        rt_diff = np.diff(rts)
        rt_diff_lt3 = rt_diff < (3 * cycle_time)

        # https://stackoverflow.com/a/38161867
        # Get start, stop index pairs for islands/seq. of 1s
        idx_pairs = np.where(np.diff(np.hstack(([False], rt_diff_lt3 == 1, [False]))))[
            0
        ].reshape(-1, 2)
        if len(idx_pairs) < 1:
            continue

        # Get the island lengths, whose argmax would give us the ID of longest island.
        # Start index of that island would be the desired output
        start_end_longest_seq = idx_pairs[np.diff(idx_pairs, axis=1).argmax(), :]
        rts = rts[start_end_longest_seq[0] : start_end_longest_seq[1] + 1]

        calc_widths.append(rts[-1] - rts[0])

    calc_widths = np.array(calc_widths)
    median_calc_width = np.median(calc_widths)
    logger.info(
        f"Peak width stats: mean={calc_widths.mean():.03f}s"
        f" std={calc_widths.std():.03f}s"
        f" median={np.median(calc_widths):.03f}s"
    )

    # mass_bins = np.arange(400, 1000, 0.5).astype(np.float64)
    # mass_shifts = 0.4
    # mass_bins += mass_shifts

    # rt_arr = np.zeros((len(unique_rts), len(mass_bins) - 1))
    # for (rt, ql), sdf in tqdm(
    #     df.groupby(["rt", "quad_low"]), desc="Calculting peak widths"
    # ):
    #     if ql != first_quad_low:
    #         continue
    #     rt_arr[unique_rts == rt, :] = np.histogram(
    #         sdf["mzs"], bins=mass_bins, weights=sdf["intensities"]
    #     )[0]

    # df_bkp = df
    # df = df.head(50000)

    logger.info("Filtering df")
    with Timeit("Filtering df"):
        df = df[(df["mzs"] > df["quad_high"] + 1) | (df["mzs"] < df["quad_low"] - 1)]

    logger.info("Clustering traces ...")
    with Timeit("Clustering traces"):
        try:
            dbscan_clustering, labels = CentroidedPrecursorIndex.from_dataframe(
                df
            ).dbscan_combine_traces(
                rt_scaling=traces_config.rt_scaling * median_calc_width,
                mobility_scaling=traces_config.mobility_scaling,
                quad_scaling=5,
                mz_scaling=traces_config.mz_scaling,
                dbscan_min_samples=traces_config.dbscan_min_samples,
                dbscan_p=traces_config.dbscan_p,
                use_meanshift=False,
                keep_unclustered=traces_config.keep_unclustered,
            )
        except IndexError as e:
            logger.error(f"Error clustering: {e}")
            return 0

    with Timeit("Meanshift clustering"):
        rtcp = RTCentroidedPrecursorIndex(**dbscan_clustering)
        try:
            pseudoscans, pseudoscan_labels = rtcp.meanshift_cluster(
                # might be better in terms of cycle time ...
                rt_scaling=cluster_config.rt_scaling * median_calc_width,
                mobility_scaling=cluster_config.mobility_scaling,
                quad_scaling=5,
                # min_bin_freq=cluster_config.min_bin_freq,
                min_bin_freq=cluster_config.min_bin_freq,
                bandwidth=cluster_config.binwidth,
                cluster_all=cluster_config.cluster_all,
                use_meanshift=True,
                expand_dist=cluster_config.expand_dists,
            )
            # Maybe have a two stage where unclustered get assigned to closest within
            # a range.
        except IndexError as e:
            logger.error(f"Error clustering: {e}")
            return 0

    if plot:
        with Timeit("Plotting clustering"):
            plot_clustering(
                pseudoscan_labels, dbscan_clustering, pseudoscans, pq_file + "_ms.png"
            )

    # with Timeit("Meanshift clustering"):
    #     rtcp = RTCentroidedPrecursorIndex(**dbscan_clustering)
    #     try:
    #         pseudoscans, pseudoscan_labels = rtcp.meanshift_cluster(
    #             # might be better in terms of cycle time ...
    #             rt_scaling=cluster_config.rt_scaling * cycle_time,
    #             mobility_scaling=cluster_config.mobility_scaling,
    #             quad_scaling=5,
    #             min_bin_freq=cluster_config.min_bin_freq,
    #             bandwidth=cluster_config.binwidth,
    #             cluster_all=cluster_config.cluster_all,
    #             use_meanshift=cluster_config.use_meanshift,
    #         )
    #     except IndexError as e:
    #         logger.error(f"Error clustering: {e}")
    #         return 0

    # with Timeit("Plotting clustering"):
    #     plot_clustering(
    #         pseudoscan_labels, dbscan_clustering, pseudoscans, pq_file + ".png"
    #     )

    # parent scans could be frame ids ...
    # or just an integerized retention time ...
    if len(pseudoscans["rt"]) == 0:
        logger.error("No pseudoscans found ...")
        return 0

    with tempfile.NamedTemporaryFile(suffix=".mzML", dir="tmp") as f:
        with tempfile.TemporaryDirectory(dir="tmp") as d:
            pseudoscans_to_mzml(pseudoscans, f.name)
            with Timeit("Running SAGE"):
                subprocess.run(
                    [  # noqa: S603, S607
                        "sage",
                        "--fasta",
                        "UP000005640_9606.fasta",
                        "-o",
                        d,
                        "--write-pin",
                        "sageconfig.json",
                        f.name,
                    ],
                    check=True,
                )

            pin_file = pd.read_csv(f"{d}/results.sage.pin", sep="\t")

    peptide_pinfile = (
        pin_file.groupby(["Label", "Peptide"])
        .agg({"ln(-poisson)": "min"})
        .reset_index()
    )  # I am using min to penalize having multiple times the same peptide
    deltascores = (
        peptide_pinfile["ln(-poisson)"][peptide_pinfile["Label"] == 1].sum()
        - peptide_pinfile["ln(-poisson)"][peptide_pinfile["Label"] == -1].sum()
    )

    return deltascores


from pathlib import Path


def _copy(self, target):
    import shutil

    assert self.is_file()
    shutil.copy(str(self), str(target))  # str() only there for Python < (3, 6)


Path.copy = _copy


def optuna_objective(trial):
    traces_config = CombineTracesConfig.from_trial(trial)
    cluster_config = ClusterTracesConfig.from_trial(trial)
    # traces_config = CombineTracesConfig()
    # cluster_config = ClusterTracesConfig()
    # cluster_config.expand_dists = trial.suggest_float("expand_dist", 5, 100)

    with tempfile.NamedTemporaryFile(dir="tmp", suffix=".parquet") as d:
        Path("5_currfile.parquet").copy(d.name)
        # neighbor_denoise(
        #     NeighborDenoiseConfig.from_trial(trial),
        #     IMSCentroidConfig.from_trial(trial),
        #     targetfile=d.name,
        # )
        out = main(
            traces_config=traces_config,
            cluster_config=cluster_config,
            pq_file=d.name,
            plot=False,
        )

    return out


if __name__ == "__main__":
    import optuna

    # if not Path("5_3817_TIMS2_2col-80m_37_1_Slot1-46_1_4768.lance.parquet").exists():

    if not Path("5_currfile.parquet").exists():
        neighbor_denoise(
            NeighborDenoiseConfig(),
            IMSCentroidConfig(),
        )
    baseline = main(
        traces_config=CombineTracesConfig(),
        cluster_config=ClusterTracesConfig(),
        plot=True,
    )
    logger.info(f"Baseline score: {baseline}")

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="thp1_clupmneigh_parallel5",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(optuna_objective, n_trials=500, gc_after_trial=True, n_jobs=4)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
