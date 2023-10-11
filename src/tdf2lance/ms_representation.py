from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from loguru import logger
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


def _grouped_std(
    labels: NDArray[np.int64],
    unique_labels: NDArray[np.int64],
    float_array: NDArray[np.float64],
    counts: NDArray[np.float64],
):
    arr_len = max(unique_labels.max() + 2, 2)
    squared_vals = np.zeros(arr_len)
    summed_vals = np.zeros(arr_len)

    np.add.at(
        squared_vals,
        labels,
        float_array**2,
    )
    np.add.at(
        summed_vals,
        labels,
        float_array,
    )
    # s0 = sum(1 for x in samples) # counts
    # s1 = sum(x for x in samples)  # sum
    # s2 = sum(x*x for x in samples) # sum of squares
    # std_dev = math.sqrt((s0 * s2 - s1 * s1)/(s0 * s0))
    denom = counts * counts
    enumerator = (counts * squared_vals) - (summed_vals * summed_vals)
    enumerator = np.clip(enumerator, 0, None)
    std_dev = np.sqrt(enumerator / denom)
    return std_dev


def test_grouped_std():
    labels = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, -1, -1], dtype=np.int64)
    unique_labels = np.unique(labels)
    values = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=np.float64
    )
    counts = np.zeros(len(unique_labels))
    np.add.at(
        counts,
        labels,
        np.ones_like(values, dtype=float),
    )

    expected = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    expected[0] = np.std(values[labels == 0])
    expected[1] = np.std(values[labels == 1])
    expected[2] = np.std(values[labels == 2])
    expected[3] = np.std(values[labels == -1])

    real_out = _grouped_std(
        labels=labels, unique_labels=unique_labels, float_array=values, counts=counts
    )
    assert np.allclose(real_out, expected)


def _find_neighbor_ranges(
    sorted_arr, sorted_other, tolerance, eps, *, min_keep=1
) -> dict[int, range]:
    """Find the ranges of values in sorted_arr that are within tolerance of sorted_other.

    Note: Arrays need to be sorted in ascending order.
    """
    eps_tol = tolerance + eps

    # I am not sure if this optimization saves that much time ...
    # if len(sorted_arr) < 200 and (len(sorted_other) != len(sorted_arr)):
    #     if (sorted_arr[0] > (sorted_other[-1] + eps_tol)) or (
    #         sorted_arr[-1] < (sorted_other[0] - eps_tol)
    #     ):
    #         return {}
    li = np.searchsorted(sorted_other - (eps_tol), sorted_arr, side="left")
    ri = np.searchsorted(sorted_other + (eps_tol), sorted_arr, side="right")

    try:
        assert not np.any(li < ri), "li must be smaller than ri"  # noqa: S101
    except AssertionError as e:
        msg = f"{e} for {li} and {ri}, maybe one of the arrays is not sorted?"
        # I am leaving this as an assertion, since it should never happen.
        # And I would like to skip it by running with -O
        raise AssertionError(msg) from e
    pos_with_neighbors = np.where((li - ri) >= min_keep)[0]
    neighbor_ranges_list = {i: range(ri[i], li[i]) for i in pos_with_neighbors}

    return neighbor_ranges_list


def _is_sorted_ascending(arr):
    return np.all(np.diff(arr) >= 0)


def _intersect_range(range_left: range, range_right: range) -> range | None:
    """Return the intersection of two ranges."""
    if range_left.stop < range_right.start:
        return None
    if range_right.stop < range_left.start:
        return None
    return range(
        max(range_left.start, range_right.start), min(range_left.stop, range_right.stop)
    )


def _intersect_range_dict(
    ranges_left: dict[int, range], ranges_right: dict[int, range]
) -> dict[int, range]:
    """Return the intersection of two range dicts."""
    out = {}
    for k in ranges_left:
        if k not in ranges_right:
            continue
        intersection = _intersect_range(ranges_left[k], ranges_right[k])
        if intersection is None:
            continue
        out[k] = intersection
    return out


def _expand_range(rng: range, exclude: set[int] | None = None) -> NDArray[np.int64]:
    """Expand a range a numpy array.

    The output array is the indices that would be included in the range.
    If exclude is provided, the values in exclude will be excluded from the range.
    """
    out = np.arange(rng.start, rng.stop)
    if exclude is not None:
        out = out[~np.isin(out, exclude)]
    return out


def _glimpse_array(arr, n=5):
    if len(arr) <= 2 * n:
        return arr.__str__()
    else:
        return arr[:n].__str__() + " ... " + arr[-n:].__str__()


def _empty_frame_or_dist(return_distances, frame: Frame, other_mzs):
    if return_distances:
        return coo_matrix(
            ([], ([], [])),
            shape=(len(frame.concatenated_mzs), len(other_mzs)),
            dtype=np.float64,
        )
    else:
        return Frame(rt=frame.rt, pushes=[], frame_index=frame.frame_index)


# TODO make a provate method for testing, where
# a push is generated from blobs.
@dataclass
class Push:
    mzs: NDArray
    intensities: NDArray
    scan_index: int
    mobility: float
    precursor_index: int
    push_index: int
    quad_low_mz: float
    quad_high_mz: float

    def glimpse(self) -> str:
        """Return a string representation of the push."""
        return (
            f"Push(scan_index={self.scan_index}, mobility={self.mobility}, "
            f"precursor_index={self.precursor_index}, push_index={self.push_index}, "
            f"quad_low_mz={self.quad_low_mz}, quad_high_mz={self.quad_high_mz}, "
            f"mzs={_glimpse_array(self.mzs)}, "
            f"intensities={_glimpse_array(self.intensities)})"
        )

    def __post_init__(self):
        if not self.mzs.shape == self.intensities.shape:
            msg = "mzs and intensities must have the same shape"
            raise ValueError(msg)

    def __len__(self):
        return len(self.mzs)

    @cached_property
    def is_mz_sorted(self):
        return _is_sorted_ascending(self.mzs)

    def searchsorted(self, mz):
        if not self.is_mz_sorted:
            msg = "mzs are not sorted"
            raise ValueError(msg)
        return np.searchsorted(self.mzs, mz)

    def find_frame_neighbors(
        self, other, mz_tolerance=0.1, eps=1e-8
    ) -> dict[int, range]:
        """Find the frame indices of the closest mz values in other frame.

        For example, if we have two frames, A and B, and a tolerance t.
        And we want to find the pairs of values between A and B that are
        within `t`+`eps` of each other.

        """
        not_sorted_msg = "mzs are not sorted"
        if not self.is_mz_sorted:
            raise ValueError(not_sorted_msg)
        if not other.is_mz_sorted:
            raise ValueError(not_sorted_msg)

        this = self.mzs
        other = other.mzs
        neighbor_list = _find_neighbor_ranges(this, other, mz_tolerance, eps)

        return neighbor_list

    def count_frame_neighbors(self, other, mz_tolerance, eps=1e-8) -> dict[int, int]:
        """Count the number of mz values in other frame within tolerance of each mz.

        Returns
        -------
        dict[int, int]

        See Also
        --------
        Push.find_frame_neighbors
        """
        frame_neighbors = self.find_frame_neighbors(other, mz_tolerance, eps)
        out = {i: len(r) for i, r in frame_neighbors.items()}
        return out

    def count_frame_neighbors_list(self, other_list, mz_tolerance, eps=1e-8) -> NDArray:
        """Count the number of neighboring peaks in other frames.

        This function is similar to `count_frame_neighbors` but for a list of frames.

        Parameters
        ----------
        other_list : list
            A list of `Frame` objects.
        mz_tolerance : float
            The mz tolerance to use to find neighboring peaks.
        eps : float
            A small value to add to the tolerance to account for floating point errors.
            In practice if eps is 0, the function will return neighbors that are
            LESS THAN the tolerance apart, whilst any small value will return neighbors
            that are LESS THAN OR EQUAL TO the tolerance apart.
        """
        distance_dicts = [
            self.count_frame_neighbors(other, mz_tolerance, eps) for other in other_list
        ]
        out = None
        for d in distance_dicts:
            if out is None:
                out = d
            else:
                for k, v in d.items():
                    out[k] = out.get(k, 0) + v
        return out

    def filter_n_frame_neighbors_list(
        self, other_list, n, mz_tolerance, eps=1e-8
    ) -> Push:
        """Filter mz values that have less than n neighbors in other frames.

        This function returns a new `Push` object with the subset of mz and
        intensity values that have at least n neighbors in the other frames.

        Parameters
        ----------
        other_list : list
            A list of `Frame` objects.
        n : int
            The minimum number of neighbors in the other frames.
        mz_tolerance : float
            The mz tolerance to use to find neighboring peaks.
        eps : float
            A small value to add to the tolerance to account for floating point errors.
            In practice if eps is 0, the function will return neighbors that are
            LESS THAN the tolerance apart, whilst any small value will return neighbors
            that are LESS THAN OR EQUAL TO the tolerance apart.
        """
        nns = self.count_frame_neighbors_list(other_list, mz_tolerance, eps)
        if nns is not None:
            keep = [k for k, v in nns.items() if v >= n]
            keep.sort()
        else:
            keep = []

        return Push(
            mzs=self.mzs[keep],
            intensities=self.intensities[keep],
            scan_index=self.scan_index,
            mobility=self.mobility,
            precursor_index=self.precursor_index,
            quad_low_mz=self.quad_low_mz,
            quad_high_mz=self.quad_high_mz,
            push_index=self.push_index,
        )

    def __getitem__(self, i):
        return type(self)(
            mzs=self.mzs[i],
            intensities=self.intensities[i],
            scan_index=self.scan_index,
            mobility=self.mobility,
            precursor_index=self.precursor_index,
            quad_low_mz=self.quad_low_mz,
            quad_high_mz=self.quad_high_mz,
            push_index=self.push_index,
        )

    def get_sparse_peak_distance(
        self,
        other,
        mz_tolerance,
        mobility_tolerance,
        quad_tolerance,
    ):
        """Get the sparse distance matrix between the peaks in this push and other push.

        The output will be a matrix of size len(self) x len(other). The values will be
        the sum of the distances in the three dimensions (mz, mobility, quad) scaled by
        the tolerance in each dimension IF the peaks are within those tolerances,
        otherwise it will be the fill value.

        Parameters
        ----------
        other : Push
            The other push to compare to.
        mz_tolerance : float
            The mz tolerance to use to find neighboring peaks.
        mobility_tolerance : float
            The mobility tolerance to use to find neighboring peaks.
            This is not used for filtering, but is used to compute the distance.
        quad_tolerance : float
            The quad tolerance to use to find neighboring peaks.
            This is not used for filtering, but is used to compute the distance.
        """

        mobility_distance = np.abs(self.mobility - other.mobility) / mobility_tolerance
        quad_low_distance = (
            np.abs(self.quad_low_mz - other.quad_low_mz) / quad_tolerance
        )
        quad_high_distance = (
            np.abs(self.quad_high_mz - other.quad_high_mz) / quad_tolerance
        )
        common_distance = quad_low_distance + quad_high_distance + mobility_distance

        # Check if the peaks are within the tolerances
        neighs = _find_neighbor_ranges(self.mzs, other.mzs, mz_tolerance, eps=1e-8)
        iis = []
        rrs = []
        ds = np.zeros(sum(len(r) for r in neighs.values()), dtype="float")

        # TODO consider a single array allocation and then
        # doing implicit unbuffered assignment
        ii = 0
        for i, r in neighs.items():
            # distances[i, r] = common_distance
            # distances[i, r] += np.abs(self.mzs[i] - other.mzs[r]) / mz_tolerance
            rs = list(r)
            iis.extend([i] * len(rs))
            rrs.extend(rs)
            cd = common_distance + (np.abs(self.mzs[i] - other.mzs[rs]) / mz_tolerance)
            cd += 1e-8
            np.add.at(ds, range(ii, ii + len(rs)), cd)

        distances = sparse.coo_matrix(
            (ds, (iis, rrs)),
            shape=(len(self), len(other)),
            dtype="float",
        )

        return distances


@dataclass
class Frame:
    rt: float
    pushes: list[Push]
    frame_index: int

    def __post_init__(self):
        """Checks the passed arguments for validity.

        Check that the `scan_index` attribute of each `Push` object is unique
        and that the `mobilities` attribute of `Push` objects is sorted in
        ascending order.

        Raises:
            ValueError: If the `scan_index` attribute of any `Push` object
                is not unique or if the `mobilities` attribute of `Push`
                objects is not sorted in ascending order.
        """
        if not len({push.scan_index for push in self.pushes}) == len(self.pushes):
            msg = "scan_index must be unique"
            raise ValueError()

        # Check that pushes are sorted by ion mobility
        if not _is_sorted_ascending(self.mobilities):
            which_unsorted = np.where(np.diff(self.mobilities) < 0)[0]
            msg = (
                f"mobilities are not sorted: at index {which_unsorted} "
                f"for {self.mobilities}"
            )
            raise ValueError(msg)

        # check that pushes have unique ids
        if len({push.scan_index for push in self.pushes}) != len(self.pushes):
            msg = "scan_index must be unique"
            raise ValueError(msg)

    def glimpse(self):
        """Return a string representation of the frame."""
        return (
            f"Frame: rt={self.rt}, frame_index={self.frame_index} \n"
            f"   scan_indices={_glimpse_array(self.scan_indices)}\n"
            f"   mobilities={_glimpse_array(self.mobilities)}\n"
            f"   precursor_indices={_glimpse_array(self.precursor_indices)}\n"
            f"   quad_low={_glimpse_array(self.quad_low_mzs)}\n"
            f"   quad_high={_glimpse_array(self.quad_high_mzs)}\n"
            f"   npeaks={_glimpse_array(self.n_peaks)}\n"
        )

    @cached_property
    def n_peaks(self):
        """
        Return the number of peaks in each `Push` object as a numpy array.

        Returns
        -------
        np.ndarray
            An array of integers representing the number of peaks
            in each `Push` object.
        """
        return np.array([len(push) for push in self.pushes])

    @property
    def scan_indices(self):
        """
        Return a list of the `scan_index` attribute of each `Push` object.

        Returns:
        list
            A list of integers representing the `scan_index` attribute
            of each `Push` object.
        """
        return [push.scan_index for push in self.pushes]

    @cached_property
    def concatenated_intensities(self):
        return np.concatenate([push.intensities for push in self.pushes])

    @cached_property
    def concatenated_mzs(self):
        return np.concatenate([push.mzs for push in self.pushes])

    @cached_property
    def concatenated_mobilities(self):
        return np.concatenate(
            [np.repeat(push.mobility, len(push)) for push in self.pushes]
        )

    @cached_property
    def concatenated_quad_low_values(self):
        return np.concatenate(
            [np.repeat(push.quad_low_mz, len(push)) for push in self.pushes]
        )

    @cached_property
    def concatenated_quad_high_values(self):
        return np.concatenate(
            [np.repeat(push.quad_high_mz, len(push)) for push in self.pushes]
        )

    @cached_property
    def mobilities(self) -> NDArray:
        out = np.array([push.mobility for push in self.pushes])
        return out

    @cached_property
    def precursor_indices(self):
        return [push.precursor_index for push in self.pushes]

    @cached_property
    def quad_low_mzs(self):
        return np.array([push.quad_low_mz for push in self.pushes])

    @cached_property
    def quad_high_mzs(self):
        return np.array([push.quad_high_mz for push in self.pushes])

    def __getitem__(self, i):
        return type(self)(self.rt, [self.pushes[i]], self.frame_index)

    def find_push_neighbors(
        self,
        other,
        mobility_tolerance=0.015,
        quad_tolerance=5,
        eps=1e-8,
    ) -> dict[int, range]:
        """Return which pushes in other are neighbors of pushes in this frame.

        Two pushes are considered neighbors if they are at most `mobility_tolerance`
        apart in mobility and `quad_tolerance` apart in quad_low/quad_high.

        For "normal" tims DIA data, the quad_low/quad_high would be a value very close
        to 0, since there is no overlap in isolation windows, and therefore could be
        considered a pretty independend dimension. Although in practide this does not
        really matter, since the quad_low/quad_high are far enought within a frame that
        nearly no value will make them overlap.

        For "weird" tims DIA data, like midia/slice/synchro, the quad_low/quad_high
        the value of the quad tolerance should be set to something; since the dimensions
        are not independent.
        """

        mobility_neighborhoods = _find_neighbor_ranges(
            self.mobilities, other.mobilities, mobility_tolerance, eps
        )
        quad_low_neighborhoods = _find_neighbor_ranges(
            self.quad_low_mzs, other.quad_low_mzs, quad_tolerance, eps
        )
        quad_high_neighborhoods = _find_neighbor_ranges(
            self.quad_high_mzs, other.quad_high_mzs, quad_tolerance, eps
        )

        # Find the intersection of the neighborhoods
        # This is the set of pushes that are neighbors in all dimensions
        # (mobility, quad_low, quad_high)

        combined_neighbors = _intersect_range_dict(
            quad_low_neighborhoods, quad_high_neighborhoods
        )
        combined_neighbors = _intersect_range_dict(
            combined_neighbors, mobility_neighborhoods
        )

        return combined_neighbors

    def filter_n_peak_neighbors(
        self,
        others: Frame | list[Frame],
        n=1,
        *,
        mz_tolerance=0.015,
        mobility_tolerance=0.015,
        quad_tolerance=5,
        eps=1e-8,
        skip_same_index=False,
    ):
        """Filter pushes that have less than n neighbors in the same frame."""
        if skip_same_index:
            if not isinstance(others, Frame):
                msg = "skip_same_index can only be used if other is a Frame"
                raise ValueError(msg)
            if len(self.pushes) != len(others.pushes):
                msg = (
                    "pushes must have the same length to use the "
                    "`skip_same_index` option"
                )
                raise ValueError(msg)

        if isinstance(others, Frame):
            others = [others]

        neighborhood_lists = [
            self.find_push_neighbors(
                other,
                mobility_tolerance=mobility_tolerance,
                eps=eps,
                quad_tolerance=quad_tolerance,
            )
            for other in others
        ]

        out_pushes = []

        for i in range(len(self.pushes)):
            right_pushes = []
            for other, neighborhood_list in zip(others, neighborhood_lists):
                if i not in neighborhood_list:
                    break

                r = neighborhood_list[i]
                for k in r:
                    if skip_same_index:
                        if self.pushes[i].scan_index == other.pushes[k].scan_index:
                            continue
                    right_pushes.append(other.pushes[k])

            left_push = self.pushes[i]
            out_push = left_push.filter_n_frame_neighbors_list(
                right_pushes, n=n, mz_tolerance=mz_tolerance, eps=eps
            )
            if len(out_push) > 0:
                out_pushes.append(out_push)

        return Frame(self.rt, out_pushes, self.frame_index)

    def faster_filter_n_peak_mz_neighbors(
        self,
        other,
        n,
        mz_tolerance,
        *,
        mobility_tolerance=None,
        quad_tolerance=None,
        eps=1e-6,
        return_distances=False,
    ):
        # Since neighbors along the mz axis are a lot more sparse than along the
        # mobility axis, we can speed up the search by first filtering along the
        # mz axis.

        # Fist get all the mzs, and make an index array to sort them (and then be
        # able to sort them back)
        this_mzs = self.concatenated_mzs
        this_idx = np.arange(len(this_mzs))
        this_mzs_order = np.argsort(this_mzs)
        this_peaks_per_frame = self.n_peaks
        this_peak_push_idx = np.concatenate(
            [np.repeat(i, p) for i, p in enumerate(this_peaks_per_frame)]
        )

        if isinstance(other, Frame):
            other_mzs = other.concatenated_mzs
            other_idx = np.arange(len(this_mzs))
            other_mzs_order = np.argsort(this_mzs)
            other_peaks_per_frame = other.n_peaks
            # could be optimized using out for cocnatenate
            other_peak_push_idx = np.concatenate(
                [np.repeat(i, p) for i, p in enumerate(other_peaks_per_frame)]
            )
            other_mobilities = other.mobilities
            other_quad_low_mzs = other.quad_low_mzs
        elif isinstance(other, list):
            # Do the same but concatenate all the mzs
            other_mzs = np.concatenate([f.concatenated_mzs for f in other])
            other_idx = np.concatenate(
                [np.arange(len(f.concatenated_mzs)) for f in other]
            )
            other_mzs_order = np.argsort(other_mzs)
            other_peaks_per_frame = np.concatenate([f.n_peaks for f in other])
            other_peak_push_idx = np.concatenate(
                [np.repeat(i, p) for i, p in enumerate(other_peaks_per_frame)]
            )
            other_mobilities = np.concatenate([f.mobilities for f in other])
            other_quad_low_mzs = np.concatenate([f.quad_low_mzs for f in other])

        # Sort the mzs
        this_mzs = this_mzs[this_mzs_order]
        this_idx = this_idx[this_mzs_order]
        mzsorted_this_peak_push_idx = this_peak_push_idx[this_mzs_order]
        other_mzs = other_mzs[other_mzs_order]
        other_idx = other_idx[other_mzs_order]
        mzsorted_other_peak_push_idx = other_peak_push_idx[other_mzs_order]

        # def op1():
        #     # Find the indices of the mzs in frame2 that are within the mz tolerance
        #     # of the mzs in frame1
        #     nns = _find_neighbor_ranges(
        #         this_mzs, other_mzs, tolerance=mz_tolerance, eps=eps, min_keep=n
        #     )
        #     nns = {k: v for k, v in nns.items() if len(v) >= n}

        #     if len(nns) == 0:
        #         return [], [], []
        #         return _empty_frame_or_dist(return_distances, self, other_mzs)

        #     # OPTIMIZE: pre-allocate the indices array
        #     # left_idxs = np.concatenate([[k] * len(v) for k, v in nns.items()])
        #     # right_idxs = np.concatenate([v for _, v in nns.items()])

        #     array_len = sum(len(v) for v in nns.values())
        #     left_idxs = np.zeros(array_len, dtype=np.int64)
        #     right_idxs = np.zeros(array_len, dtype=np.int64)
        #     idx = 0
        #     for k, v in nns.items():
        #         lv = len(v)
        #         left_idxs[idx : idx + lv] = k
        #         right_idxs[idx : idx + lv] = v
        #         idx += lv

        #     # Now we can just do a pair-wise comparisson of the peaks for their
        #     # mobilities
        #     if mobility_tolerance is not None:
        #         this_mobilities = self.mobilities[
        #             mzsorted_this_peak_push_idx[left_idxs]
        #         ]
        #         tmp_other_mobilities = other_mobilities[
        #             mzsorted_other_peak_push_idx[right_idxs]
        #         ]
        #         mobility_diffs = np.abs(this_mobilities - tmp_other_mobilities)
        #         mob_keep = mobility_diffs <= (mobility_tolerance + eps)
        #         left_idxs = left_idxs[mob_keep]
        #         right_idxs = right_idxs[mob_keep]

        #     if quad_tolerance is not None:
        #         this_quads = self.quad_low_mzs[mzsorted_this_peak_push_idx[left_idxs]]
        #         tmp_other_quads = other_quad_low_mzs[
        #             mzsorted_other_peak_push_idx[right_idxs]
        #         ]
        #         quad_diffs = np.abs(this_quads - tmp_other_quads)
        #         quad_keep = quad_diffs <= (quad_tolerance + eps)
        #         left_idxs = left_idxs[quad_keep]
        #         right_idxs = right_idxs[quad_keep]

        #     # if mobility_tolerance is not None or quad_tolerance is not None:
        #     #     nns = dict()
        #     #     for k, v in zip(left_idxs, right_idxs):
        #     #         if k not in nns:
        #     #             nns[k] = []
        #     #         # OPTIMIZE: reimplement as unique count + subset
        #     #         nns[k].append(v)

        #     # keep = np.array([k for k, v in nns.items() if len(v) >= n])
        #     unique_left_idxs, counts = np.unique(left_idxs, return_counts=True)
        #     keep_indices = np.in1d(left_idxs, unique_left_idxs[counts >= n])
        #     keep = unique_left_idxs[counts >= n]
        #     left_idxs, right_idxs = left_idxs[keep_indices], right_idxs[keep_indices]
        #     return left_idxs, right_idxs, keep

        # def op2():
        left_kdt_vals = [this_mzs / mz_tolerance]
        right_kdt_vals = [other_mzs / mz_tolerance]
        if mobility_tolerance is not None:
            left_kdt_vals.append(
                self.mobilities[mzsorted_this_peak_push_idx] / mobility_tolerance
            )
            right_kdt_vals.append(
                other_mobilities[mzsorted_other_peak_push_idx] / mobility_tolerance
            )
        if quad_tolerance is not None:
            left_kdt_vals.append(
                self.quad_low_mzs[mzsorted_this_peak_push_idx] / quad_tolerance
            )
            right_kdt_vals.append(
                other_quad_low_mzs[mzsorted_other_peak_push_idx] / quad_tolerance
            )

        left_kdt = KDTree(
            np.stack(left_kdt_vals, axis=1),
            leafsize=50,
            balanced_tree=True,
            copy_data=False,
            compact_nodes=False,
        )
        right_kdt = KDTree(
            np.stack(right_kdt_vals, axis=1),
            leafsize=50,
            balanced_tree=True,
            copy_data=False,
            compact_nodes=False,
        )
        nns = left_kdt.query_ball_tree(right_kdt, r=1, p=1)
        array_len = sum(len(v) for v in nns if len(v) >= n)
        left_idxs = np.zeros(array_len, dtype=np.int64)
        right_idxs = np.zeros(array_len, dtype=np.int64)
        idx = 0
        for k, v in enumerate(nns):
            if len(v) < n:
                continue
            lv = len(v)
            left_idxs[idx : idx + lv] = k
            right_idxs[idx : idx + lv] = v
            idx += lv
        unique_left_idxs, counts = np.unique(left_idxs, return_counts=True)
        keep = unique_left_idxs[counts >= n]

        # Now we revert the index array to get the indices of the mzs in frame1
        if len(keep) == 0:
            return _empty_frame_or_dist(return_distances, self, other_mzs)

        # if required, calculate the paired distances between the peaks
        if return_distances:
            mz_distance = (
                np.abs(this_mzs[left_idxs] - other_mzs[right_idxs]) / mz_tolerance
            )
            frame_indices_l = mzsorted_this_peak_push_idx[left_idxs]
            frame_indices_r = mzsorted_other_peak_push_idx[right_idxs]
            mobility_distance = (
                np.abs(
                    self.mobilities[frame_indices_l] - other_mobilities[frame_indices_r]
                )
                / mobility_tolerance
            )
            quad_distance = (
                np.abs(
                    self.quad_low_mzs[frame_indices_l]
                    - other_quad_low_mzs[frame_indices_r]
                )
                / quad_tolerance
            )
            distances = np.sqrt(
                mz_distance**2 + mobility_distance**2 + quad_distance**2
            )
            distances += eps
            distance_order = np.argsort(distances)

            out_dist_matrix = coo_matrix(
                (
                    distances[distance_order],
                    (
                        this_idx[left_idxs][distance_order],
                        other_idx[right_idxs][distance_order],
                    ),
                ),
                shape=(len(this_mzs), len(other_mzs)),
                dtype=np.float64,
            )
            return out_dist_matrix

        keep = this_idx[keep]
        keep_order = np.argsort(keep)
        keep = keep[keep_order]

        cum_peaks = 0
        keep_pushes = []
        for i, p in enumerate(this_peaks_per_frame):
            st = np.searchsorted(keep, cum_peaks, side="left")
            en = np.searchsorted(keep, cum_peaks + p, side="left")

            if en - st >= 1:
                final_indices = keep[st:en] - cum_peaks
                keep_pushes.append(self.pushes[i][final_indices])
            cum_peaks += p

        out_frame = Frame(rt=self.rt, pushes=keep_pushes, frame_index=self.frame_index)
        return out_frame

    def get_peak_sparse_distance(
        self,
        mz_tolerance,
        mobility_tolerance,
        quad_tolerance,
    ):
        """Get the sparse distance matrix between peaks in this frame.

        The sparse distance matrix is a matrix where each row and column correspond
        to a peak in this frame. The value at row i and column j is the distance
        between peak i and peak j.

        The distance is the sum of the distances in the three dimensions
        (mz, mobility, quad) scaled by the tolerance in each dimension.

        The index is the cumulative number of peaks in the previous frames.
        For instance if we have two frames, A and B, with 10 and 20 peaks respectively.
        Then the index of the first peak in B is 10. And we would have a matrix
        of size 30x30.
        """
        n_peaks = self.n_peaks
        n = n_peaks.sum()

        rows = []
        cols = []
        data = []

        # Use the neighboring function to find the indices of the peaks
        # that are within the tolerance of each other.

        push_neighbors = self.find_push_neighbors(
            self,
            mobility_tolerance=mobility_tolerance,
            quad_tolerance=quad_tolerance,
        )

        i = 0
        for j, n_peaks_j in enumerate(n_peaks):
            ii = 0
            if n_peaks_j == 0:
                continue

            if j not in push_neighbors:
                i += n_peaks_j
                continue

            push_j = self.pushes[j]
            j_neighbors = list(push_neighbors[j])
            for k, n_peaks_k in enumerate(n_peaks):
                if n_peaks_k == 0:
                    continue

                if k not in j_neighbors:
                    ii += n_peaks_k
                    continue

                push_k = self.pushes[k]
                if j == k:
                    distances = push_j.get_sparse_peak_distance(
                        push_k,
                        mz_tolerance=mz_tolerance,
                        mobility_tolerance=mobility_tolerance,
                        quad_tolerance=quad_tolerance,
                    )
                    # out[i : i + n_peaks_j, i : i + n_peaks_k] = distances
                    # np.add.at(
                    #     out,
                    #     # (slice(i, i + n_peaks_j), slice(i, i + n_peaks_k)),
                    #     (i + distances.row, i + distances.col),
                    #     distances.data,
                    # )
                    rows.append(i + distances.row)
                    cols.append(i + distances.col)
                    data.append(distances.data)
                elif k > j:
                    distances = push_j.get_sparse_peak_distance(
                        push_k,
                        mz_tolerance=mz_tolerance,
                        mobility_tolerance=mobility_tolerance,
                        quad_tolerance=quad_tolerance,
                    )
                    # out[i : i + n_peaks_j, ii : ii + n_peaks_k] = distances
                    # out[ii : ii + n_peaks_k, i : i + n_peaks_j] = distances.T
                    rows.append(i + distances.row)
                    cols.append(ii + distances.col)
                    data.append(distances.data)

                    cols.append(i + distances.row)
                    rows.append(ii + distances.col)
                    data.append(distances.data)

                    # np.add.at(
                    #     out,
                    #     # (slice(i, i + n_peaks_j), slice(ii, ii + n_peaks_k)),
                    #     (i + distances.row, ii + distances.col),
                    #     distances.data,
                    # )
                    # np.add.at(
                    #     out,
                    #     (ii + distances.col, i + distances.row),
                    #     distances.data,
                    # )

                else:
                    # If we are in the same frame, we only need to
                    # compute the upper triangle since the distance matrix is symmetric.
                    ii += n_peaks_k
                    continue
                ii += n_peaks_k

            i += n_peaks_j

        data = np.concatenate(data) if data else np.array([], dtype="float")
        out = sparse.coo_matrix(
            (data, (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
            dtype="float",
        )
        return out

    @classmethod
    def from_dataframe(cls, df):
        push_list = []
        if len(set(df.frame_indices)) != 1:
            msg = "scan_index must be the same for all rows"
            raise ValueError(msg)
        for _, row in df.iterrows():
            push = Push(
                mzs=row.mz_values,
                intensities=row.corrected_intensity_values,
                scan_index=row.scan_indices,
                mobility=row.mobility_values,
                precursor_index=row.precursor_indices,
                quad_low_mz=row.quad_low_mz_values,
                quad_high_mz=row.quad_high_mz_values,
                push_index=row.push_indices,
            )
            push_list.append(push)

        push_list.sort(key=lambda x: x.mobility)
        return cls(
            rt=df.rt_values.iloc[0],
            pushes=push_list,
            frame_index=df.frame_indices.iloc[0],
        )

    def join(self, other):
        if self.frame_index != other.frame_index:
            msg = "frame_index must be the same"
            raise ValueError(msg)
        if self.rt != other.rt:
            msg = "rt must be the same"
            raise ValueError(msg)

        pushes = self.pushes + other.pushes
        pushes.sort(key=lambda x: x.mobility)

        # Raise error if there are duplicated scan indices
        if len(set(pushes.scan_index)) != len(pushes):
            msg = "scan_index must be unique"
            raise ValueError(msg)

        return Frame(self.rt, pushes, self.frame_index)

    def dbscan_sinplify_frame(
        self,
        n=3,
        mz_tolerance=0.01,
        mobility_tolerance=0.01,
        quad_tolerance=5,
        eps=1e-5,
        *,
        dbscan_n=None,
        keep_unclustered=False,
    ):
        if dbscan_n is None:
            dbscan_n = n

        # TODO: check if adding some form of intensity ratio filtering would be useful
        outdists = self.scaled_kdtree_distance(
            mz_tolerance=mz_tolerance + eps,
            mobility_tolerance=mobility_tolerance + eps,
            quad_tolerance=quad_tolerance + eps,
        )

        if isinstance(outdists, Frame) or len(outdists.data) == 0:
            if isinstance(outdists, Frame):
                logger.warning(self.n_peaks)
                logger.warning(
                    "Got a frame back, returning empty arrays. This shoudl not happen"
                )
            # return all empty arrays
            out = {
                "intensities": np.array([]),
                "mzs": np.array([]),
                "mobilities": np.array([]),
                "quad_low": np.array([]),
                "quad_high": np.array([]),
                "mobility_min": np.array([]),
                "mobility_max": np.array([]),
                "mz_min": np.array([]),
                "mz_max": np.array([]),
            }
            out = CentroidedFrame(rt=self.rt, frame_index=self.frame_index, **out)
            return out

        db = DBSCAN(metric="precomputed", eps=1, min_samples=dbscan_n).fit(outdists)
        uniq_labs, pos, counts = np.unique(
            db.labels_, return_index=True, return_counts=True
        )

        arr_len = max(2, uniq_labs.max() + 2)
        counts = np.zeros(arr_len)
        b_averaged_mzs = np.zeros(arr_len)
        b_mz_max = np.zeros(arr_len)
        b_mz_min = np.full(arr_len, max(self.concatenated_mzs))
        b_averaged_mobilities = np.zeros(arr_len)
        b_mobility_max = np.zeros(arr_len)
        b_mobility_min = np.full(arr_len, max(self.mobilities))
        b_intensities = np.zeros(arr_len)
        b_quad_low = np.zeros(arr_len)
        b_quad_high = np.zeros(arr_len)

        float_intensities = self.concatenated_intensities.astype(float)
        float_mzs = self.concatenated_mzs.astype(float)
        float_mobilities = self.concatenated_mobilities.astype(float)

        np.add.at(
            counts,
            db.labels_,
            np.float64(1),
        )
        # TODO consider just using a median ...
        np.add.at(
            b_intensities,
            db.labels_,
            float_intensities,
        )

        # The standard deviation can be calculated as the
        # sqrt of the sqrt(((sum of squared mzs) / count) - squared(sum of mzs / count))

        ## Calculate the standard deviation of the mzs
        std_dev_mz = _grouped_std(
            labels=db.labels_,
            unique_labels=uniq_labs,
            float_array=float_mzs,
            counts=counts,
        )

        ## Calculate the standard deviation of the mobilities
        std_dev_mobility = _grouped_std(
            labels=db.labels_,
            unique_labels=uniq_labs,
            float_array=float_mobilities,
            counts=counts,
        )

        np.add.at(
            b_averaged_mzs,
            db.labels_,
            float_intensities * float_mzs,
        )
        # Now we need to divide by the sum of the weights
        b_averaged_mzs /= b_intensities

        np.add.at(
            b_averaged_mobilities,
            db.labels_,
            float_intensities * float_mobilities,
        )
        b_averaged_mobilities /= b_intensities

        np.minimum.at(
            b_mz_min,
            db.labels_,
            float_mzs,
        )
        np.maximum.at(
            b_mz_max,
            db.labels_,
            float_mzs,
        )
        np.minimum.at(
            b_mobility_min,
            db.labels_,
            float_mobilities,
        )
        np.maximum.at(
            b_mobility_max,
            db.labels_,
            float_mobilities,
        )

        np.add.at(
            b_quad_low,
            db.labels_,
            float_intensities * self.concatenated_quad_low_values.astype(float),
        )
        b_quad_low /= b_intensities

        np.add.at(
            b_quad_high,
            db.labels_,
            float_intensities * self.concatenated_quad_high_values.astype(float),
        )
        b_quad_high /= b_intensities

        out = {
            "intensities": b_intensities[:-1],
            "mzs": b_averaged_mzs[:-1],
            "mobilities": b_averaged_mobilities[:-1],
            "quad_low": b_quad_low[:-1],
            "quad_high": b_quad_high[:-1],
            "mobility_min": b_mobility_min[:-1],
            "mobility_max": b_mobility_max[:-1],
            "mobility_std": std_dev_mobility[:-1],
            "mz_min": b_mz_min[:-1],
            "mz_max": b_mz_max[:-1],
            "mz_std": std_dev_mz[:-1],
            "n_peaks": counts[:-1],
        }
        if keep_unclustered:
            unclustered_indices = np.where(db.labels_ == -1)[0]
            unclustered_out = {
                "intensities": float_intensities[unclustered_indices],
                "mzs": float_mzs[unclustered_indices],
                "mobilities": float_mobilities[unclustered_indices],
                "quad_low": self.concatenated_quad_low_values[unclustered_indices],
                "quad_high": self.concatenated_quad_high_values[unclustered_indices],
                "mobility_std": np.zeros(len(unclustered_indices)),
                "mz_std": np.zeros(len(unclustered_indices)),
                "n_peaks": np.ones(len(unclustered_indices)),
            }
            unclustered_out["mobility_min"] = unclustered_out["mobilities"]
            unclustered_out["mobility_max"] = unclustered_out["mobilities"]
            unclustered_out["mz_min"] = unclustered_out["mzs"]
            unclustered_out["mz_max"] = unclustered_out["mzs"]

            out = {k: np.concatenate([v, unclustered_out[k]]) for k, v in out.items()}
        return CentroidedFrame(rt=self.rt, frame_index=self.frame_index, **out)

    def scaled_kdtree_distance(
        self,
        mz_tolerance=0.01,
        mobility_tolerance=0.01,
        quad_tolerance=5,
    ):
        kdt = KDTree(
            np.stack(
                [
                    self.concatenated_mzs / mz_tolerance,
                    self.concatenated_mobilities / mobility_tolerance,
                    self.concatenated_quad_low_values / quad_tolerance,
                ],
                axis=1,
            ),
            leafsize=50,
            balanced_tree=False,
            copy_data=False,
            compact_nodes=True,
        )

        sdm = kdt.sparse_distance_matrix(kdt, 1, p=1, output_type="coo_matrix")
        return sdm


@dataclass
class CentroidedFrame:
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
    n_peaks: np.ndarray
    rt: float
    frame_index: int

    def remove_in_precursor(self) -> CentroidedFrame:
        keep = (self.mzs < self.quad_high) & (self.mzs > self.quad_low)
        return CentroidedFrame(
            intensities=self.intensities[keep],
            mzs=self.mzs[keep],
            mobilities=self.mobilities[keep],
            quad_low=self.quad_low[keep],
            quad_high=self.quad_high[keep],
            mobility_min=self.mobility_min[keep],
            mobility_max=self.mobility_max[keep],
            mobility_std=self.mobility_std[keep],
            mz_min=self.mz_min[keep],
            mz_max=self.mz_max[keep],
            mz_std=self.mz_std[keep],
            rt=self.rt,
            frame_index=self.frame_index,
        )

    @cached_property
    def peak_labels(self):
        return np.array(list(range(len(self.intensities))))

    @cached_property
    def total_intensity(self):
        return self.intensities.sum()

    def as_record(self):
        return {
            "intensities": self.intensities,
            "mzs": self.mzs,
            "mobilities": self.mobilities,
            "quad_low": self.quad_low,
            "quad_high": self.quad_high,
            "mobility_min": self.mobility_min,
            "mobility_max": self.mobility_max,
            "mobility_std": self.mobility_std,
            "mz_min": self.mz_min,
            "mz_max": self.mz_max,
            "mz_std": self.mz_std,
            "rt": self.rt,
            "frame_index": self.frame_index,
        }

    def split_ims_clusters(self, *, remove_in_quad=True):
        from scipy.spatial.distance import pdist, squareform

        b_mobility_min = self.mobility_min
        b_mobility_max = self.mobility_max
        b_averaged_mobilities = self.mobilities

        metric_mat = np.vstack(
            [b_mobility_min / 3, b_averaged_mobilities, b_mobility_max / 3]
        )
        d = squareform(pdist(metric_mat.T, "euclidean"))
        if remove_in_quad:
            b_quad_low = self.quad_low
            b_quad_high = self.quad_high
            b_averaged_mzs = self.mzs
            in_quad_range = ((b_quad_high + 1) > b_averaged_mzs) * (
                (b_quad_low - 1) < b_averaged_mzs
            )
            d[in_quad_range, :] = 1e4
            d[:, in_quad_range] = 1e4
        db_2pass = DBSCAN(metric="precomputed", eps=0.02, min_samples=3).fit(d)

        uniq_labs, pos, counts = np.unique(
            db_2pass.labels_, return_index=True, return_counts=True
        )
        arr_len = max((uniq_labs.max() + 2), 2)
        cb_averaged_mobilities = np.zeros(arr_len)
        counts_2 = np.zeros(arr_len)
        np.add.at(
            counts_2,
            db_2pass.labels_,
            np.ones_like(b_averaged_mobilities),
        )
        np.add.at(
            cb_averaged_mobilities,
            db_2pass.labels_,
            b_averaged_mobilities,
        )
        cb_averaged_mobilities /= counts_2

        out_pseudo_scans = []
        for i, u in enumerate(uniq_labs):
            if u == -1:
                continue
            keep = db_2pass.labels_ == u
            out_pseudo_scans.append(
                PseudoScan(
                    rt=self.rt,
                    mobility_center=cb_averaged_mobilities[i],
                    frame_index=self.frame_index,
                    mz_values=self.mzs[keep],
                    intensity_values=self.intensities[keep],
                    mobility_values=self.mobilities[keep],
                    quad_low_mz_values=self.quad_low[keep],
                    quad_high_mz_values=self.quad_high[keep],
                )
            )
        return out_pseudo_scans

    def plot(self, ax=None, background: Frame | None = None, **kwargs):
        try:
            from matplotlib import pyplot as plt
        except ImportError as e:
            err = (
                "Matplotlib is required for plotting."
                " Please install it with `pip install matplotlib`"
            )
            raise ImportError(err) from e

        if ax is None:
            fig, ax = plt.subplots()

        mz_width = 10
        mobility_width = 0.1
        maxint = np.argmax(self.intensities)
        x1, x2, y1, y2 = (
            self.mzs[maxint] - (mz_width / 2),
            self.mzs[maxint] + (mz_width / 2),
            self.mobilities[maxint] - (mobility_width / 2),
            self.mobilities[maxint] + (mobility_width / 2),
        )

        axins = ax.inset_axes(
            [0.1, 0.5, 0.47, 0.47],
            xlim=(x1, x2),
            ylim=(y1, y2),
            # xticklabels=[],
            yticklabels=[],
        )
        for a in [ax, axins]:
            if background is not None:
                a.scatter(
                    x=background.concatenated_mzs,
                    y=background.concatenated_mobilities,
                    c="gray",
                    s=np.log(background.concatenated_intensities),
                    alpha=0.5,
                )

            a.scatter(
                self.mzs,
                self.mobilities,
                c=np.log(self.intensities),
                s=np.log(self.intensities),
                **kwargs,
            )
        ax.indicate_inset_zoom(axins, edgecolor="black")
        return ax

    def faster_filter_n_peak_mz_neighbors(
        self, other, *, n, mobility_tolerance, quad_tolerance, mz_tolerance, eps
    ):
        mz_tolerance += eps
        mobility_tolerance += eps

        self_stack = [
            self.mzs / mz_tolerance,
            self.mobilities / mobility_tolerance,
        ]
        if quad_tolerance is not None:
            quad_tolerance += eps
            self_stack.append(self.quad_low / quad_tolerance)

        self_kdt = KDTree(
            np.stack(
                self_stack,
                axis=1,
            ),
            leafsize=50,
            balanced_tree=False,
            copy_data=False,
            compact_nodes=False,
        )
        if isinstance(other, CentroidedFrame):
            other_stack = [
                other.mzs / mz_tolerance,
                other.mobilities / mobility_tolerance,
            ]
            if quad_tolerance is not None:
                other_stack.append(other.quad_low / quad_tolerance)
            other_kdt = KDTree(
                np.stack(
                    other_stack,
                    axis=1,
                ),
                leafsize=50,
                balanced_tree=False,
                copy_data=False,
                compact_nodes=False,
            )
        elif isinstance(other, list):
            other_stack = [
                np.concatenate([o.mzs for o in other]) / mz_tolerance,
                np.concatenate([o.mobilities for o in other]) / mobility_tolerance,
            ]

            if quad_tolerance is not None:
                other_stack.append(
                    np.concatenate([o.quad_low for o in other]) / quad_tolerance
                )

            other_kdt = KDTree(
                np.stack(
                    other_stack,
                    axis=1,
                ),
                leafsize=50,
                balanced_tree=False,
                copy_data=False,
                compact_nodes=False,
            )
        else:
            msg = "other must be a CentroidedFrame or a list of CentroidedFrames"
            raise ValueError(msg)

        nns = self_kdt.query_ball_tree(other_kdt, r=1, p=1)
        keep_left = np.zeros(len(self.mzs), dtype=bool)
        for i, w in enumerate(nns):
            if len(w) >= n:
                keep_left[i] = True

        out = CentroidedFrame(
            intensities=self.intensities[keep_left],
            mzs=self.mzs[keep_left],
            mobilities=self.mobilities[keep_left],
            quad_low=self.quad_low[keep_left],
            quad_high=self.quad_high[keep_left],
            mobility_min=self.mobility_min[keep_left],
            mobility_max=self.mobility_max[keep_left],
            mobility_std=self.mobility_std[keep_left],
            mz_min=self.mz_min[keep_left],
            mz_max=self.mz_max[keep_left],
            mz_std=self.mz_std[keep_left],
            n_peaks=self.n_peaks[keep_left],
            rt=self.rt,
            frame_index=self.frame_index,
        )
        return out


@dataclass
class PseudoScan:
    rt: float
    mobility_center: float
    frame_index: int
    mz_values: np.ndarray
    intensity_values: np.ndarray
    mobility_values: np.ndarray
    quad_low_mz_values: np.ndarray
    quad_high_mz_values: np.ndarray

    @cached_property
    def total_intensity(self):
        return self.intensity_values.sum()

    @cached_property
    def n_peaks(self):
        return len(self.intensity_values)

    @cached_property
    def mz_fingerprint(self):
        # The fingerprint is a binned representation of the mzs
        # The bins are 10 da wide from 0 to 2000 da

        # And the values are the added intensities in each bin
        bins = np.arange(0, 2000, 10)
        out = np.zeros(len(bins) - 1)
        np.add.at(out, np.digitize(self.mz_values, bins), self.intensity_values)
        return out

    def as_dataframe(self):
        return pd.DataFrame(
            {
                "rt": self.rt,
                "mobility_center": self.mobility_center,
                "frame_index": self.frame_index,
                "mz_values": self.mz_values,
                "intensity_values": self.intensity_values,
                "mobility_values": self.mobility_values,
                "quad_low_mz_values": self.quad_low_mz_values,
                "quad_high_mz_values": self.quad_high_mz_values,
            }
        )
