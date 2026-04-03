"""Shared plotting utilities for all three papers."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_rolling_coefficient(rolling_df, title="Time-varying coefficient", ylabel="Coefficient", figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sig = rolling_df["pvalue"] < 0.05
    ax.plot(rolling_df["center_year"], rolling_df["coefficient"], "b-o", markersize=4)
    ax.scatter(rolling_df.loc[sig, "center_year"], rolling_df.loc[sig, "coefficient"], color="red", s=30, zorder=5, label="p < 0.05")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Center year of window")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_irf(irf_df, title="Impulse Response Function", ylabel="Response", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(irf_df["horizon"], irf_df["coefficient"], "b-o", markersize=5)
    ax.fill_between(irf_df["horizon"], irf_df["ci_lower"], irf_df["ci_upper"], alpha=0.2, color="blue")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_survival_curves(surv_df, group_col="cluster", duration_col="duration", event_col="event", title="Kaplan-Meier survival curves", figsize=(10, 6)):
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(figsize=figsize)
    kmf = KaplanMeierFitter()
    for group, grp in surv_df.groupby(group_col):
        kmf.fit(grp[duration_col], grp[event_col], label=str(group))
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Duration (years from extensification peak)")
    ax.set_ylabel("Probability of remaining in Malthusian regime")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_counterfactual(cf_df, title="Counterfactual simulation", ylabel="Response variable", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cf_df["year"], cf_df["actual"], "b-", label="Actual", linewidth=2)
    ax.plot(cf_df["year"], cf_df["counterfactual"], "r--", label="Counterfactual", linewidth=2)
    ax.fill_between(cf_df["year"], cf_df["actual"], cf_df["counterfactual"], alpha=0.15, color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
