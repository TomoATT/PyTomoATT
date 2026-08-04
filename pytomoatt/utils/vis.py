"""Visualization helpers for source--receiver data."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

if TYPE_CHECKING:
    from pytomoatt.src_rec import SrcRec


_PLOT_FONTSIZE = 14


def _apply_axis_font_sizes(axis) -> None:
    """Apply consistent label and tick font sizes to an axis."""
    axis.xaxis.label.set_size(_PLOT_FONTSIZE)
    axis.yaxis.label.set_size(_PLOT_FONTSIZE)
    axis.tick_params(axis="both", labelsize=_PLOT_FONTSIZE)


def _apply_colorbar_font_sizes(colorbar) -> None:
    """Apply consistent label and tick font sizes to a colorbar."""
    colorbar.ax.xaxis.label.set_size(_PLOT_FONTSIZE)
    colorbar.ax.yaxis.label.set_size(_PLOT_FONTSIZE)
    colorbar.ax.tick_params(labelsize=_PLOT_FONTSIZE)


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
        receiver_color_values = None
    else:
        _check_columns(receivers, {"stlo", "stla"}, "receivers")
        receiver_values = receivers[["stlo", "stla"]].to_numpy(dtype=float)
        finite_receivers = np.isfinite(receiver_values).all(axis=1)
        receiver_lon, receiver_lat = receiver_values[finite_receivers].T
        receiver_color_values = None
        if color_by == "weight" and "weight" in receivers:
            receiver_color_values = np.asarray(
                receivers.loc[finite_receivers, "weight"], dtype=float
            )
            if not np.isfinite(receiver_color_values).all():
                raise ValueError("receivers contains non-finite weights")

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
        cmap = "jet"
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
    source_colorbar_y = 0.68 if receiver_color_values is not None else 0.52
    colorbar_axis = colorbar_host.inset_axes(
        (0.05, source_colorbar_y, 0.9, 0.12)
    )

    scatter_options = {
        "c": color_values,
        "cmap": cmap,
        "s": 8,
        "label": "Sources",
    }
    scatter_options.update(kwargs)
    if color_by == "weight" and "norm" not in scatter_options:
        vmin = scatter_options.pop("vmin", 0.0)
        vmax = scatter_options.pop("vmax", 1.0)
        shared_norm = Normalize(vmin=vmin, vmax=vmax)
        shared_norm.autoscale_None(color_values)
        scatter_options["norm"] = shared_norm

    source_scatter = map_axis.scatter(source_lon, source_lat, **scatter_options)
    section_scatter_options = scatter_options.copy()
    section_scatter_options["norm"] = source_scatter.norm
    section_scatter_options.pop("vmin", None)
    section_scatter_options.pop("vmax", None)
    latitude_depth_axis.scatter(
        source_depth, source_lat, **section_scatter_options
    )
    longitude_depth_axis.scatter(
        source_lon, source_depth, **section_scatter_options
    )

    receiver_scatter = None
    if receiver_lon.size:
        receiver_scatter_options = {
            "edgecolors": "white",
            "linewidths": 0.5,
            "label": "Receivers",
            "marker": "v",
            "s": 55,
        }
        if receiver_color_values is None:
            receiver_scatter_options["c"] = "tab:red"
        else:
            receiver_norm = Normalize(vmin=source_scatter.norm.vmin, vmax=source_scatter.norm.vmax)
            receiver_norm.autoscale_None(receiver_color_values)
            receiver_scatter_options.update({
                "c": receiver_color_values,
                "cmap": source_scatter.cmap,
                "norm": receiver_norm,
            })
        receiver_scatter = map_axis.scatter(
            receiver_lon,
            receiver_lat,
            **receiver_scatter_options,
        )

    map_axis.set(
        xlabel="Longitude",
        ylabel="Latitude",
        xlim=lon_limits,
        ylim=lat_limits,
    )
    map_axis.set_aspect("equal", adjustable="box")

    def _align_latitude_depth_axis(axis, renderer):
        lower_section_position = longitude_depth_axis.get_position(
            original=True
        )
        map_position = map_axis.get_position()
        # Reuse the lower section's automatically calculated padding so the
        # right and lower gaps are equal in physical units for any figure size.
        section_gap_inches = (
            map_position.y0 - lower_section_position.y1
        ) * figure.get_figheight()
        horizontal_gap = section_gap_inches / figure.get_figwidth()
        depth_length_inches = (
            lower_section_position.height * figure.get_figheight()
        )
        section_width = depth_length_inches / figure.get_figwidth()
        section_x0 = map_position.x1 + horizontal_gap
        return Bbox.from_extents(
            section_x0,
            map_position.y0,
            section_x0 + section_width,
            map_position.y1,
        )

    def _align_longitude_depth_axis(axis, renderer):
        section_position = axis.get_position(original=True)
        map_position = map_axis.get_position()
        return Bbox.from_extents(
            map_position.x0,
            section_position.y0,
            map_position.x1,
            section_position.y1,
        )

    def _align_colorbar_host(axis, renderer):
        right_position = _align_latitude_depth_axis(
            latitude_depth_axis, renderer
        )
        lower_position = _align_longitude_depth_axis(
            longitude_depth_axis, renderer
        )
        return Bbox.from_extents(
            right_position.x0,
            lower_position.y0,
            right_position.x1,
            lower_position.y1,
        )

    latitude_depth_axis.set_axes_locator(_align_latitude_depth_axis)
    longitude_depth_axis.set_axes_locator(_align_longitude_depth_axis)
    colorbar_host.set_axes_locator(_align_colorbar_host)
    map_axis.legend(fontsize=_PLOT_FONTSIZE)

    latitude_depth_axis.set(
        xlabel="Depth (km)",
        ylabel="Latitude",
        xlim=depth_limits,
        ylim=lat_limits,
    )
    latitude_depth_axis.yaxis.tick_right()
    latitude_depth_axis.yaxis.set_label_position("right")
    longitude_depth_axis.set(
        xlabel="Longitude",
        ylabel="Depth (km)",
        xlim=lon_limits,
        ylim=depth_limits,
    )
    longitude_depth_axis.invert_yaxis()
    for axis in (map_axis, latitude_depth_axis, longitude_depth_axis):
        _apply_axis_font_sizes(axis)

    colorbar = figure.colorbar(
        source_scatter,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(colorbar_label)
    _apply_colorbar_font_sizes(colorbar)

    if receiver_color_values is not None and receiver_scatter is not None:
        receiver_colorbar_axis = colorbar_host.inset_axes(
            (0.05, 0.22, 0.9, 0.12)
        )
        receiver_colorbar = figure.colorbar(
            receiver_scatter,
            cax=receiver_colorbar_axis,
            orientation="horizontal",
        )
        receiver_colorbar.set_label("Receiver weight")
        _apply_colorbar_font_sizes(receiver_colorbar)

    if fname is not None:
        figure.savefig(fname, dpi=300, bbox_inches="tight",facecolor="white",edgecolor="white")

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
    distance: Literal["dist_deg", "dist_km", "dist_3d_km"] = "dist_3d_km",
    color=None,
    fname: str | PathLike[str] | None = None,
    fig: Figure | None = None,
    ylim="adaptive",
    **kwargs,
) -> Figure:
    """Plot absolute travel time against source--receiver distance.

    Parameters
    ----------
    src_rec
        A :class:`~pytomoatt.src_rec.SrcRec` instance whose ``rec_points``
        contains the selected distance column and ``tt``.
    distance
        Distance column used for the x-axis: ``"dist_3d_km"`` (default) for
        three-dimensional source--receiver distance in kilometres,
        ``"dist_deg"`` for epicentral distance in degrees, or ``"dist_km"``
        for epicentral distance in kilometres.
    color
        Any Matplotlib-compatible color specification for the points. When
        ``None`` (default), Matplotlib selects the next color from the current
        axis color cycle.
    fname
        Optional output path. The format is inferred by Matplotlib from the
        filename extension.
    fig
        Optional existing Matplotlib figure. Points are added to its current
        axis; an axis is created when the figure has none.
    ylim
        Y-axis scaling strategy. ``"adaptive"`` (default) uses travel times
        at the minimum and maximum selected distances, ``"auto"`` or
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
    if distance not in {"dist_deg", "dist_km", "dist_3d_km"}:
        raise ValueError(
            "distance must be 'dist_deg', 'dist_km', or 'dist_3d_km'"
        )

    records = src_rec.rec_points
    _check_columns(records, {distance, "tt"}, "rec_points")
    if records.empty:
        raise ValueError("Cannot plot travel times without receiver records")

    values = records[[distance, "tt"]].to_numpy(dtype=float)
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

    scatter_options = {"s": 4}
    if color is not None:
        scatter_options["color"] = color
    scatter_options.update(kwargs)
    axis.scatter(values[finite, 0], values[finite, 1], **scatter_options)
    distance_label = {
        "dist_deg": "Epicentral distance (degree)",
        "dist_km": "Epicentral distance (km)",
        "dist_3d_km": "3-D source-receiver distance (km)",
    }[distance]
    axis.set(
        xlabel=distance_label,
        ylabel="Travel time (s)",
    )
    _apply_axis_font_sizes(axis)

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
        figure.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="white")

    return figure
