"""Utilities to display a Rich progress indicator while blocking calls run.

This module focuses on Plotly figure rendering (``fig.show()``) but is generic
enough to wrap *any* blocking callable.

Features
--------
- `show_with_progress(fig, ...)` — drop‑in replacement for `fig.show()`.
- `run_with_progress(callable, ...)` — wrap any blocking call.
- `with_progress(...)` — decorator to add a spinner/bar to functions.
- `ProgressRunner` — context manager for `with` style usage.
- `patch_plotly_show(...)` — monkey‑patch Plotly to always show an indicator.

Notes
-----
- Threaded design: the target callable runs in a background thread while the
  main thread renders an indicator (spinner or pulsing bar).
- Works well in terminals and most notebook environments that support Rich.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Literal

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    BarColumn,
)

T = TypeVar("T")
UIKind = Literal["spinner", "bar"]

# ---------- Core engine ----------


def _run_in_thread(blocking_fn: Callable[[], T]) -> Tuple[threading.Thread, Dict[str, Any]]:
    """Run ``blocking_fn`` in a background thread.

    Parameters
    ----------
    blocking_fn : Callable[[], T]
        Zero-argument callable to execute in a worker thread. Any positional
        arguments should be bound via a closure or ``functools.partial``.

    Returns
    -------
    tuple[threading.Thread, dict]
        A tuple ``(thread, holder)`` where ``thread`` is the running worker
        thread and ``holder`` is a dict with keys:
        - ``"res"`` : T | None — result (when successful).
        - ``"exc"`` : BaseException | None — captured exception, if any.

    Notes
    -----
    - Exceptions are not raised in the worker; they are stored for the caller
      to re-raise after the indicator completes.
    """
    result_holder: Dict[str, Any] = {"res": None, "exc": None}

    def worker() -> None:
        try:
            result_holder["res"] = blocking_fn()
        except BaseException as e:  # noqa: BLE001 — bubble up in main thread later
            result_holder["exc"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t, result_holder


def _wait_with_indicator(
    thread: threading.Thread,
    *,
    description: str,
    ui: UIKind = "spinner",
    poll_interval: float = 0.1,
    bar_cycle_time: float = 2.0,
    bar_steps: int = 100,
    refresh_interval: float = 0.05,
) -> None:
    """Display a Rich indicator until ``thread`` finishes.

    Parameters
    ----------
    thread : threading.Thread
        A started thread to wait on.
    description : str
        Text displayed alongside the indicator.
    ui : {"spinner", "bar"}, optional
        Indicator style. ``"spinner"`` shows a spinner + elapsed time.
        ``"bar"`` shows an indeterminate pulsing bar, by default ``"spinner"``.
    poll_interval : float, optional
        Sleep duration (seconds) between liveness checks used by the spinner,
        by default ``0.1``. (Ignored when ``ui='bar'``; see ``refresh_interval``.)
    bar_cycle_time : float, optional
        Approximate time (seconds) for one full bar sweep, by default ``2.0``.
    bar_steps : int, optional
        Resolution of the bar ("width" of the progress), by default ``100``.
    refresh_interval : float, optional
        UI update cadence (seconds) for the bar, by default ``0.05``.

    Notes
    -----
    - Spinner waits using ``poll_interval``; bar advances by ``step`` every
      ``refresh_interval`` and resets at completion to appear indeterminate.
    """
    if ui == "spinner":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            _ = progress.add_task(description, total=None)
            while thread.is_alive():
                time.sleep(poll_interval)
            thread.join()
        return

    # ui == "bar"
    step_per_tick: float = max(1.0, bar_steps * (refresh_interval / max(bar_cycle_time, 1e-6)))
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task_id = progress.add_task(description, total=bar_steps)
        while thread.is_alive():
            progress.advance(task_id, step_per_tick)
            # keep it pulsing
            if progress.tasks[0].completed >= bar_steps:
                progress.reset(task_id)
            time.sleep(refresh_interval)
        thread.join()


# ---------- Public helpers ----------


def run_with_progress(
    blocking_fn: Callable[[], T],
    *,
    description: str = "Working…",
    ui: UIKind = "spinner",
    poll_interval: float = 0.1,
    bar_cycle_time: float = 2.0,
    bar_steps: int = 100,
    refresh_interval: float = 0.05,
) -> T:
    """Run a callable while showing a spinner or pulsing bar.

    Parameters
    ----------
    blocking_fn : Callable[[], T]
        Zero-argument callable to execute. Wrap parameters using a closure or
        ``functools.partial`` if needed.
    description : str, optional
        Text shown next to the indicator, by default ``"Working…"``.
    ui : {"spinner", "bar"}, optional
        Indicator style, by default ``"spinner"``.
    poll_interval : float, optional
        Spinner liveness check sleep (seconds), by default ``0.1``.
    bar_cycle_time : float, optional
        Approximate time (seconds) for one full sweep of the bar, by default ``2.0``.
    bar_steps : int, optional
        Resolution of the bar ("width"), by default ``100``.
    refresh_interval : float, optional
        UI update cadence (seconds) for the bar, by default ``0.05``.

    Returns
    -------
    T
        The result returned by ``blocking_fn``.

    Raises
    ------
    BaseException
        Re-raises any exception raised by ``blocking_fn``.

    Examples
    --------
    >>> import time
    >>> run_with_progress(lambda: time.sleep(1.0), description="Sleeping…", ui="bar")
    """
    t, holder = _run_in_thread(blocking_fn)
    _wait_with_indicator(
        t,
        description=description,
        ui=ui,
        poll_interval=poll_interval,
        bar_cycle_time=bar_cycle_time,
        bar_steps=bar_steps,
        refresh_interval=refresh_interval,
    )
    if holder["exc"] is not None:
        raise holder["exc"]
    return holder["res"]  # type: ignore[return-value]


def with_progress(
    description: str = "Working…",
    *,
    ui: UIKind = "spinner",
    poll_interval: float = 0.1,
    bar_cycle_time: float = 2.0,
    bar_steps: int = 100,
    refresh_interval: float = 0.05,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that shows an indicator while the wrapped function runs.

    Parameters
    ----------
    description : str, optional
        Text shown next to the indicator, by default ``"Working…"``.
    ui : {"spinner", "bar"}, optional
        Indicator style, by default ``"spinner"``.
    poll_interval : float, optional
        Spinner liveness check sleep (seconds), by default ``0.1``.
    bar_cycle_time : float, optional
        Approximate time (seconds) for one full sweep of the bar, by default ``2.0``.
    bar_steps : int, optional
        Resolution of the bar ("width"), by default ``100``.
    refresh_interval : float, optional
        UI update cadence (seconds) for the bar, by default ``0.05``.

    Returns
    -------
    Callable
        A decorator that wraps a function and returns the wrapped callable.

    Examples
    --------
    >>> @with_progress("Generating & opening…", ui="bar")
    ... def render():
    ...     fig = make_figure()
    ...     fig.show()
    ...
    ... render()
    """

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        def wrapped(*args: Any, **kwargs: Any) -> T:
            return run_with_progress(
                lambda: fn(*args, **kwargs),
                description=description,
                ui=ui,
                poll_interval=poll_interval,
                bar_cycle_time=bar_cycle_time,
                bar_steps=bar_steps,
                refresh_interval=refresh_interval,
            )

        return wrapped

    return deco


class ProgressRunner:
    """Context manager showing an indicator during a critical section.

    Useful for explicit scoping around blocking operations, e.g.::

        with ProgressRunner("Opening plot…", ui="bar"):
            fig.show()

    Parameters
    ----------
    description : str, optional
        Text shown next to the indicator, by default ``"Working…"``.
    ui : {"spinner", "bar"}, optional
        Indicator style, by default ``"spinner"``.
    poll_interval : float, optional
        Spinner liveness check sleep (seconds), by default ``0.1``.
    bar_cycle_time : float, optional
        Approximate time (seconds) for one full sweep of the bar, by default ``2.0``.
    bar_steps : int, optional
        Resolution of the bar ("width"), by default ``100``.
    refresh_interval : float, optional
        UI update cadence (seconds) for the bar, by default ``0.05``.
    """

    def __init__(
        self,
        description: str = "Working…",
        *,
        ui: UIKind = "spinner",
        poll_interval: float = 0.1,
        bar_cycle_time: float = 2.0,
        bar_steps: int = 100,
        refresh_interval: float = 0.05,
    ) -> None:
        self.description: str = description
        self.ui: UIKind = ui
        self.poll_interval: float = poll_interval
        self.bar_cycle_time: float = bar_cycle_time
        self.bar_steps: int = bar_steps
        self.refresh_interval: float = refresh_interval
        self._progress: Optional[Progress] = None
        self._task_id: Optional[int] = None

    def __enter__(self) -> "ProgressRunner":
        if self.ui == "spinner":
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                transient=True,
            )
            self._progress.__enter__()
            self._task_id = self._progress.add_task(self.description, total=None)
        else:
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                transient=True,
            )
            self._progress.__enter__()
            self._task_id = self._progress.add_task(self.description, total=self.bar_steps)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        assert self._progress is not None
        self._progress.__exit__(exc_type, exc, tb)


# ---------- Plotly-specific sugar ----------


def show_with_progress(
    fig: Any,
    *,
    description: str = "Opening Plotly figure…",
    renderer: Optional[str] = None,
    ui: UIKind = "spinner",
    poll_interval: float = 0.1,
    bar_cycle_time: float = 2.0,
    bar_steps: int = 100,
    refresh_interval: float = 0.05,
) -> Any:
    """Drop-in replacement for ``fig.show()`` with a spinner or pulsing bar.

    Parameters
    ----------
    fig : Any
        A Plotly figure (e.g., ``plotly.graph_objects.Figure``). Typed as ``Any``
        to avoid importing Plotly at module import time.
    description : str, optional
        Text shown next to the indicator, by default ``"Opening Plotly figure…"``.
    renderer : str, optional
        Optional Plotly renderer name (e.g., ``"browser"``). If ``None``, the
        default renderer is used.
    ui : {"spinner", "bar"}, optional
        Indicator style, by default ``"spinner"``.
    poll_interval : float, optional
        Spinner liveness check sleep (seconds), by default ``0.1``.
    bar_cycle_time : float, optional
        Approximate time (seconds) for one full sweep of the bar, by default ``2.0``.
    bar_steps : int, optional
        Resolution of the bar ("width"), by default ``100``.
    refresh_interval : float, optional
        UI update cadence (seconds) for the bar, by default ``0.05``.

    Returns
    -------
    Any
        Whatever ``fig.show()`` returns (often ``None``).

    Raises
    ------
    BaseException
        Re-raises any exception from the underlying ``fig.show()`` call.

    Examples
    --------
    >>> show_with_progress(fig, ui="bar")
    >>> show_with_progress(fig, renderer="browser", ui="spinner")
    """

    def _call() -> Any:
        if renderer is None:
            return fig.show()
        return fig.show(renderer=renderer)

    return run_with_progress(
        _call,
        description=description,
        ui=ui,
        poll_interval=poll_interval,
        bar_cycle_time=bar_cycle_time,
        bar_steps=bar_steps,
        refresh_interval=refresh_interval,
    )


def patch_plotly_show(
    *,
    description: str = "Opening Plotly figure…",
    ui: UIKind = "spinner",
    poll_interval: float = 0.1,
    bar_cycle_time: float = 2.0,
    bar_steps: int = 100,
    refresh_interval: float = 0.05,
) -> Callable[..., Any]:
    """Monkey‑patch Plotly's ``Figure.show`` to always display an indicator.

    Call this once near program start. After patching, any subsequent
    ``fig.show()`` will render a Rich spinner or bar until the call returns.

    Parameters
    ----------
    description : str, optional
        Text shown next to the indicator, by default ``"Opening Plotly figure…"``.
    ui : {"spinner", "bar"}, optional
        Indicator style, by default ``"spinner"``.
    poll_interval : float, optional
        Spinner liveness check sleep (seconds), by default ``0.1``.
    bar_cycle_time : float, optional
        Approximate time (seconds) for one full sweep of the bar, by default ``2.0``.
    bar_steps : int, optional
        Resolution of the bar ("width"), by default ``100``.
    refresh_interval : float, optional
        UI update cadence (seconds) for the bar, by default ``0.05``.

    Returns
    -------
    Callable[..., Any]
        The original (unpatched) ``Figure.show`` method, which you can keep if
        you wish to restore the original behavior later.

    Raises
    ------
    RuntimeError
        If Plotly's ``BaseFigure`` cannot be imported.

    Examples
    --------
    >>> original = patch_plotly_show(description="Opening…", ui="bar")
    >>> # ... later, to restore:
    >>> # from plotly.basedatatypes import BaseFigure
    >>> # BaseFigure.show = original
    """
    try:
        # Works for plotly.graph_objects.Figure (inherits BaseFigure)
        from plotly.basedatatypes import BaseFigure  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Could not import plotly.basedatatypes.BaseFigure") from exc

    original_show: Callable[..., Any] = BaseFigure.show  # type: ignore[attr-defined]

    def patched_show(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN001 - external signature
        return run_with_progress(
            lambda: original_show(self, *args, **kwargs),
            description=description,
            ui=ui,
            poll_interval=poll_interval,
            bar_cycle_time=bar_cycle_time,
            bar_steps=bar_steps,
            refresh_interval=refresh_interval,
        )

    BaseFigure.show = patched_show  # type: ignore[attr-defined]
    return original_show  # in case you want to restore later

