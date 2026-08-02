"""Visualization helpers for source--receiver data."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pytomoatt.src_rec import SrcRec


def _axis_limits(values: np.ndarray, padding: float = 0.1) -> tuple[float, float]:
    """Calculate padded limits, including for a constant-valued coordinate."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    pad = span * padding if span else max(abs(lower) * padding, 0.5)
    return lower - pad, upper + pad


def _endpoint_travel_time_limits(
    distances: np.ndarray,
    travel_times: np.ndarray,
    padding: float = 0.05,
) -> tuple[float, float]:
    """Calculate travel-time limits from the minimum/maximum distances."""
    minimum_distance = np.min(distances)
    maximum_distance = np.max(distances)
    if np.isclose(minimum_distance, maximum_distance):
        return _axis_limits(travel_times, padding=padding)

    minimum_distance_times = travel_times[
        np.isclose(distances, minimum_distance)
    ]
    maximum_distance_times = travel_times[
        np.isclose(distances, maximum_distance)
    ]
    endpoint_times = np.array([
        np.min(minimum_distance_times),
        np.max(maximum_distance_times),
    ])
    return _axis_limits(endpoint_times, padding=padding)


def _check_columns(data, columns: set[str], name: str) -> None:
    missing = columns.difference(data.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"{name} is missing required columns: {missing_names}")


def plot_src_rec(
    src_rec: "SrcRec",
    *,
    color_by: Literal["depth", "weight"] = "depth",
    fname: str | PathLike[str] | None = None,
    **kwargs,
) -> Figure:
    """Plot source and receiver locations with two source-depth sections.

    Parameters
    ----------
    src_rec
        A :class:`~pytomoatt.src_rec.SrcRec` instance.
    color_by
        Source attribute used for color mapping: ``"depth"`` or
        ``"weight"``. Receiver symbols remain red so they can be
        distinguished from sources.
    fname
        Optional output path. The format is inferred by Matplotlib from the
        filename extension.
    **kwargs
        Additional keyword arguments passed to Matplotlib's
        :meth:`~matplotlib.axes.Axes.scatter` for source points. Common
        options include ``cmap``, ``s``, ``alpha``, ``marker``, ``vmin``,
        ``vmax``, ``norm``, ``edgecolors`` and ``linewidths``.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Notes
    -----
    This function does not add temporary columns to ``src_rec`` or otherwise
    mutate it.
    """
    sources = src_rec.src_points
    _check_columns(sources, {"evlo", "evla", "evdp"}, "src_points")
    if sources.empty:
        raise ValueError("Cannot plot an SrcRec object without sources")

    source_values = sources[["evlo", "evla", "evdp"]].to_numpy(dtype=float)
    finite_sources = np.isfinite(source_values).all(axis=1)
    if not finite_sources.any():
        raise ValueError("src_points contains no finite source coordinates")

    source_lon, source_lat, source_depth = source_values[finite_sources].T

    receivers = src_rec.receivers
    if receivers is None or receivers.empty:
        receiver_records = src_rec.rec_points
        if receiver_records is not None and not receiver_records.empty:
            receivers = receiver_records.drop_duplicates(subset="staname")

    if receivers is None or receivers.empty:
        receiver_lon = np.empty(0)
        receiver_lat = np.empty(0)
    else:
        _check_columns(receivers, {"stlo", "stla"}, "receivers")
        receiver_values = receivers[["stlo", "stla"]].to_numpy(dtype=float)
        finite_receivers = np.isfinite(receiver_values).all(axis=1)
        receiver_lon, receiver_lat = receiver_values[finite_receivers].T

    all_lon = np.concatenate((source_lon, receiver_lon))
    all_lat = np.concatenate((source_lat, receiver_lat))
    lon_limits = _axis_limits(all_lon)
    lat_limits = _axis_limits(all_lat)
    depth_limits = _axis_limits(np.concatenate((source_depth, [0.0])), padding=0.05)

    if color_by not in {"depth", "weight"}:
        raise ValueError("color_by must be either 'depth' or 'weight'")

    if color_by == "weight":
        _check_columns(sources, {"weight"}, "src_points")
        color_values = np.asarray(sources.loc[finite_sources, "weight"], dtype=float)
        if not np.isfinite(color_values).all():
            raise ValueError("src_points contains non-finite source weights")
        colorbar_label = "Source weight"
        cmap = "plasma"
    else:
        color_values = source_depth
        colorbar_label = "Source depth (km)"
        cmap = "viridis"

    figure = plt.figure(figsize=(8, 8), layout="constrained")
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(4, 1.35),
        height_ratios=(4, 1.35),
    )
    map_axis = figure.add_subplot(grid[0, 0])
    latitude_depth_axis = figure.add_subplot(grid[0, 1], sharey=map_axis)
    longitude_depth_axis = figure.add_subplot(grid[1, 0], sharex=map_axis)
    colorbar_host = figure.add_subplot(grid[1, 1])
    colorbar_host.set_axis_off()
    colorbar_axis = colorbar_host.inset_axes((0.05, 0.52, 0.9, 0.12))

    scatter_options = {
        "c": color_values,
        "cmap": cmap,
        "s": 8,
        "label": "Sources",
    }
    scatter_options.update(kwargs)
    source_scatter = map_axis.scatter(source_lon, source_lat, **scatter_options)
    latitude_depth_axis.scatter(source_depth, source_lat, **scatter_options)
    longitude_depth_axis.scatter(source_lon, source_depth, **scatter_options)

    if receiver_lon.size:
        map_axis.scatter(
            receiver_lon,
            receiver_lat,
            c="tab:red",
            edgecolors="white",
            linewidths=0.5,
            label="Receivers",
            marker="v",
            s=55,
        )

    map_axis.set(
        xlabel="Longitude",
        ylabel="Latitude",
        xlim=lon_limits,
        ylim=lat_limits,
    )
    map_axis.legend()

    latitude_depth_axis.set(
        xlabel="Depth (km)",
        ylabel="Latitude",
        xlim=depth_limits,
        ylim=lat_limits,
    )
    longitude_depth_axis.set(
        xlabel="Longitude",
        ylabel="Depth (km)",
        xlim=lon_limits,
        ylim=depth_limits,
    )
    longitude_depth_axis.invert_yaxis()

    colorbar = figure.colorbar(
        source_scatter,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(colorbar_label)

    if fname is not None:
        figure.savefig(fname, dpi=300, bbox_inches="tight")

    return figure


def fig_ev_st_distribution_dep(
    src_rec: "SrcRec",
    fname: str | PathLike[str] | None = None,
    *,
    color_by: Literal["depth", "weight"] = "depth",
    **kwargs,
) -> Figure:
    """Backward-compatible name for :func:`plot_src_rec`."""
    return plot_src_rec(src_rec, color_by=color_by, fname=fname, **kwargs)


def plot_travel_time(
    src_rec: "SrcRec",
    *,
    color="tab:blue",
    fname: str | PathLike[str] | None = None,
    fig: Figure | None = None,
    ylim="adaptive",
    **kwargs,
) -> Figure:
    """Plot absolute travel time against epicentral distance.

    Parameters
    ----------
    src_rec
        A :class:`~pytomoatt.src_rec.SrcRec` instance whose ``rec_points``
        contains ``dist_deg`` and ``tt`` columns.
    color
        Any Matplotlib-compatible color specification for the points.
    fname
        Optional output path. The format is inferred by Matplotlib from the
        filename extension.
    fig
        Optional existing Matplotlib figure. Points are added to its current
        axis; an axis is created when the figure has none.
    ylim
        Y-axis scaling strategy. ``"adaptive"`` (default) uses travel times
        at the minimum and maximum epicentral distances, ``"auto"`` or
        ``None`` uses Matplotlib autoscaling, ``"inherit"`` preserves the
        current limits of an existing figure, and a ``(min, max)`` pair sets
        explicit limits.
    **kwargs
        Additional keyword arguments passed to Matplotlib's
        :meth:`~matplotlib.axes.Axes.scatter`.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure. Its axis is available as ``figure.axes[0]`` for
        adding lines, annotations, or other content.
    """
    records = src_rec.rec_points
    _check_columns(records, {"dist_deg", "tt"}, "rec_points")
    if records.empty:
        raise ValueError("Cannot plot travel times without receiver records")

    values = records[["dist_deg", "tt"]].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    if not finite.any():
        raise ValueError("rec_points contains no finite distance--time pairs")

    inherited_limits = None
    inherit_requested = isinstance(ylim, str) and ylim == "inherit"
    if fig is None:
        if inherit_requested:
            raise ValueError(
                "ylim='inherit' requires an existing figure and axis"
            )
        figure, axis = plt.subplots(figsize=(6, 4.5), layout="constrained")
    elif isinstance(fig, Figure):
        figure = fig
        if inherit_requested and not figure.axes:
            raise ValueError(
                "ylim='inherit' requires a figure with an existing axis"
            )
        axis = figure.gca()
        if inherit_requested:
            inherited_limits = axis.get_ylim()
    else:
        raise TypeError("fig must be a matplotlib.figure.Figure or None")

    scatter_options = {"color": color, "s": 4}
    scatter_options.update(kwargs)
    axis.scatter(values[finite, 0], values[finite, 1], **scatter_options)
    axis.set(
        xlabel="Epicentral distance (degree)",
        ylabel="Travel time (s)",
    )

    travel_times = values[finite, 1]
    distances = values[finite, 0]
    if isinstance(ylim, str):
        if ylim == "adaptive":
            axis.set_ylim(
                _endpoint_travel_time_limits(distances, travel_times)
            )
        elif ylim == "inherit":
            if inherited_limits is None:
                raise ValueError(
                    "ylim='inherit' requires an existing figure and axis"
                )
            axis.set_ylim(inherited_limits)
        elif ylim != "auto":
            raise ValueError(
                "ylim must be 'adaptive', 'auto', 'inherit', None, "
                "or a (min, max) pair"
            )
    elif ylim is not None:
        try:
            limits = np.asarray(ylim, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                "ylim must be 'adaptive', 'auto', 'inherit', None, "
                "or a (min, max) pair"
            ) from None
        if limits.shape != (2,):
            raise ValueError(
                "ylim must be 'adaptive', 'auto', 'inherit', None, "
                "or a (min, max) pair"
            )
        lower_limit, upper_limit = limits
        if (
            not np.isfinite(lower_limit)
            or not np.isfinite(upper_limit)
            or lower_limit >= upper_limit
        ):
            raise ValueError("ylim must contain two increasing finite values")
        axis.set_ylim(lower_limit, upper_limit)

    if fname is not None:
        figure.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")

    return figure
