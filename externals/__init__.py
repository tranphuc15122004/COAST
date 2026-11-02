"""External solvers, imported only when their symbols are requested."""

__all__ = [
    "lkh_solve",
    "ort_solve",
    "RollingHorizonPolicy",
    "solve_rolling_horizon",
]


def __getattr__(name):
    if name == "lkh_solve":
        from ._lkh import lkh_solve

        return lkh_solve
    if name == "ort_solve":
        from ._ort import ort_solve

        return ort_solve
    if name in {"RollingHorizonPolicy", "solve_rolling_horizon"}:
        from ._ort_rolling import RollingHorizonPolicy, solve_rolling_horizon

        return {
            "RollingHorizonPolicy": RollingHorizonPolicy,
            "solve_rolling_horizon": solve_rolling_horizon,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
