from typing import Generator, Iterable, Literal

import lance
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from tdf2lance.ms_representation import CentroidedFrame, Frame


def iter_precursor_frames(
    ds: lance.LanceDataset,
    precursor_index: int,
    *,
    start_rt: None | float = None,
    end_rt: None | float = None,
    progress: bool = True,
) -> Generator[Frame, None, None]:
    # logger.debug(f"Counting frames for precursor index {precursor_index}")
    # tot = ds.count_rows(filter=f"precursor_indices = {precursor_index}")
    tot = None
    pbar = tqdm(
        total=tot,
        desc=f"Precursor frames, prec={precursor_index}",
        mininterval=1,
        maxinterval=2,
        disable=not progress,
    )
    last_frame_index = None
    batch_buffer = None
    last_npeaks = None
    filter_string = f"precursor_indices = {precursor_index}"
    if start_rt is not None:
        filter_string = filter_string + f" AND rt_values > {start_rt}"
    if end_rt is not None:
        filter_string = filter_string + f" AND rt_values < {end_rt}"
    for batch in ds.to_batches(filter=filter_string):
        pbar.update(len(batch))
        batch = batch.to_pandas()
        if batch_buffer is not None:
            batch = pd.concat([batch_buffer, batch])
            batch_buffer = None

        uniq_inds = np.sort(batch["frame_indices"].unique())
        num_uniq_inds = len(uniq_inds)
        curr_rt = batch["rt_values"].iloc[0]
        curr_frame_index = batch["frame_indices"].iloc[0]
        for fii in range(num_uniq_inds - 1):
            fi = uniq_inds[fii]
            subbatch = batch[batch["frame_indices"] == fi]
            curr_rt = subbatch["rt_values"].iloc[0]
            curr_frame_index = subbatch["frame_indices"].iloc[0]
            if last_frame_index is not None and last_frame_index == curr_frame_index:
                logger.warning(
                    f"Duplicate frame index {curr_frame_index} at RT {curr_rt:.2f}"
                )

            last_frame_index = curr_frame_index
            out_frame = Frame.from_dataframe(subbatch)
            tot_peaks = sum(out_frame.n_peaks)
            if last_npeaks is not None and last_npeaks > 100 * tot_peaks:
                continue
                # TODO figure out why this overhanging peaks exist ...
                # In the meantime I will just skip them, since they screw up the
                # denoise step.
                # logger.warning(
                #     f"Current frame has {tot_peaks} but previous frame had {last_npeaks}"
                #     "This discrepancy usually means an instrument or parsing error."
                # )
            last_npeaks = tot_peaks
            yield Frame.from_dataframe(subbatch)

        batch_buffer = batch[batch["frame_indices"] == uniq_inds[-1]]
        pbar.set_postfix({"RT": f"{curr_rt:.2f}", "FrameID": f"{curr_frame_index}"})


def neighbor_denoise_frames(
    frames: Iterable[Frame] | Iterable[CentroidedFrame],
    *,
    n=4,
    mz_tolerance=0.01,
    mobility_tolerance=0.015,
    quad_tolerance=5,
    epsilon=1e-8,
    sides: Literal["both"] | Literal["left"] | Literal["right"] = "both",
    yield_noisy=False,
):
    last_frame = None
    current_frame = None
    next_frame = None
    for frame in frames:
        last_frame = current_frame
        current_frame = next_frame
        next_frame = frame

        filled_buffer = (
            (last_frame is not None)
            and (next_frame is not None)
            and (current_frame is not None)
        )
        if filled_buffer:
            if sides == "both":
                n = int(n * 1.5)
                comp = [last_frame, next_frame]
            elif sides == "left":
                comp = [last_frame]
            elif sides == "right":
                comp = [next_frame]
            out_frame = current_frame.faster_filter_n_peak_mz_neighbors(
                comp,
                n=n,
                mobility_tolerance=mobility_tolerance,
                quad_tolerance=quad_tolerance,
                mz_tolerance=mz_tolerance,
                # skip_same_index=False,
                eps=epsilon,
            )
            if yield_noisy:
                yield out_frame, current_frame
            else:
                yield out_frame
