"""Repository-Aware Python Bug Triage Assistant — EDA dashboard.

Streamlit rewrite of notebooks/02_eda.ipynb: same dataset, same analyses,
same conclusions, made interactive.
"""

import os
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sqlalchemy import create_engine

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Repository-Aware Python Bug Triage Assistant",
    page_icon="🐞",
    layout="wide",
)

DB_URL = os.getenv(
    "BUGSINPY_DB_URL",
    "postgresql+psycopg2://krishnapriyagitalaxmi@localhost:5432/bugsinpy",
)

# Validated categorical slots (assigned in fixed order, never cycled).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_MUTED = "#0b0b0b", "#52514e"
GRID = "#dedcd6"

NUMERIC_FEATURES = [
    "patch_length",
    "num_files_changed",
    "lines_added",
    "lines_removed",
    "num_hunks",
    "number_of_declared_functions_in_patch",
    "number_of_declared_classes_in_patch",
]

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "font.size": 9,
})


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def to_list(value):
    """Postgres arrays arrive either as native lists or as '{a,b}' text."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None or value == "{}" or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).strip("{}").split(",") if part.strip()]


@st.cache_data(show_spinner="Loading BugsInPy dataset…")
def load_data() -> pd.DataFrame:
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM bug_dataset", engine)

    # keras bug 12 ships an empty patch, so every patch feature is 0 — drop it.
    df = df[~((df["bug_id"].astype(str) == "12") & (df["project"] == "keras"))].copy()

    for column in ["exception_types", "modified_files", "declared_classes_in_patch",
                   "declared_functions_in_patch"]:
        df[column] = df[column].apply(to_list)
    df["modified_modules"] = df["modified_modules"].apply(
        lambda x: ["root" if m == "." else m for m in to_list(x)]
    )

    df["lines_changed"] = df["lines_added"] + df["lines_removed"]
    return df


try:
    data = load_data()
except Exception as exc:  # noqa: BLE001 — surface connection errors in the UI
    st.error(
        "Could not read `bug_dataset` from PostgreSQL. "
        "Set `BUGSINPY_DB_URL` if your connection differs.\n\n"
        f"```\n{exc}\n```"
    )
    st.stop()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def show(fig, **kwargs):
    """Render a figure and release it so long sessions don't leak memory."""
    fig.tight_layout()
    st.pyplot(fig, **kwargs)
    plt.close(fig)


def hbar(series, title, xlabel, color=BLUE, height_per_row=0.32, fmt="{:,.0f}"):
    """Horizontal bar chart with direct value labels, largest at the top."""
    series = series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(2.5, height_per_row * len(series))))
    bars = ax.barh(series.index.astype(str), series.values, color=color, height=0.68)
    ax.bar_label(bars, labels=[fmt.format(v) for v in series.values],
                 padding=3, fontsize=8, color=INK_MUTED)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, series.max() * 1.14)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_color(GRID)
    ax.xaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig


def counter_series(frame, column):
    return pd.Series(Counter(item for lst in frame[column] for item in lst), dtype="int64")


def percentile_table(frame, columns):
    return (
        frame[columns]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .drop(index="count")
        .round(1)
    )


# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------

st.sidebar.header("Filters")

all_projects = sorted(data.project.unique())
selected_projects = st.sidebar.multiselect("Repositories", all_projects, default=all_projects)

all_versions = sorted(data.python_version.dropna().unique())
selected_versions = st.sidebar.multiselect("Python versions", all_versions, default=all_versions)

comment_choice = st.sidebar.radio(
    "Patch contains comments", ["All", "With comments", "Without comments"], horizontal=False
)

df = data[data.project.isin(selected_projects) & data.python_version.isin(selected_versions)]
if comment_choice == "With comments":
    df = df[df.has_comment]
elif comment_choice == "Without comments":
    df = df[~df.has_comment]

st.sidebar.caption(f"{len(df):,} of {len(data):,} bug fixes selected")
st.sidebar.divider()
st.sidebar.caption(
    "Source: BugsInPy, loaded from PostgreSQL table `bug_dataset`. "
    "keras bug 12 is excluded (empty patch)."
)

st.title("🐞 Repository-Aware Python Bug Triage Assistant")
st.caption("Interactive EDA over the BugsInPy dataset of real Python bug fixes")

if df.empty:
    st.warning("No bug fixes match the current filters.")
    st.stop()


# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Bug fixes", f"{len(df):,}")
k2.metric("Repositories", df.project.nunique())
k3.metric("Median patch length", f"{df.patch_length.median():.0f} lines")
k4.metric("Median lines changed", f"{df.lines_changed.median():.0f}")
k5.metric("Patches with comments", f"{df.has_comment.mean() * 100:.0f}%")

overview, numeric, project_tab, semantic, explorer, features = st.tabs(
    ["Overview", "Patch size", "By repository", "Semantics", "Bug explorer", "Features"]
)


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

with overview:
    left, right = st.columns([3, 2])

    with left:
        show(hbar(df.project.value_counts(), "Bug fixes per repository", "Bug fixes"))

    with right:
        show(hbar(df.python_version.value_counts(), "Python versions", "Bug fixes",
                  color=AQUA))

        comment_counts = df.has_comment.value_counts()
        fig, ax = plt.subplots(figsize=(7, 1.5))
        left_edge = 0
        gap = len(df) * 0.004  # keeps a surface-coloured seam between segments
        for label, color in [(True, BLUE), (False, ORANGE)]:
            value = int(comment_counts.get(label, 0))
            if not value:
                continue
            ax.barh([0], [value - gap], left=[left_edge], color=color, height=0.55)
            ax.text(left_edge + value / 2, 0,
                    f"{'with' if label else 'without'} comments\n{value / len(df) * 100:.0f}%",
                    ha="center", va="center", color=SURFACE, fontsize=9, weight="bold")
            left_edge += value
        ax.set_xlim(0, len(df))
        ax.set_title("Do bug-fix patches contain comments?")
        ax.axis("off")
        show(fig)

    with st.expander("Key insights", expanded=True):
        st.markdown(
            """
            - The dataset is **imbalanced**: pandas contributes the most bug fixes (169),
              while several repositories contribute fewer than 10. A model trained on it
              may partly learn *"this looks like a pandas bug."*
            - **Python 3.8.3 dominates**, with the rest spread over 3.6/3.7/3.8 releases.
              Version-specific analysis of rare releases (e.g. 3.7.7) is not meaningful.
            - Comments accompany roughly **57%** of patches — close enough to a coin flip
              that `has_comment` alone is unlikely to be a strong model feature.
            """
        )


# --------------------------------------------------------------------------
# Patch size / numerical analysis
# --------------------------------------------------------------------------

with numeric:
    clip = st.checkbox("Clip the long right tail", value=True,
                       help="Bounds the x-axis so the bulk of the distribution is readable.")

    left, right = st.columns(2)
    for column, label, color, limit, target in [
        ("patch_length", "Patch length (lines of diff)", BLUE, 450, left),
        ("lines_changed", "Lines changed (added + removed)", ORANGE, 240, right),
    ]:
        with target:
            values = df[column]
            hidden = 0
            if clip:
                hidden = int((values > limit).sum())
                values = values[values <= limit]
            fig, ax = plt.subplots(figsize=(7, 3.6))
            # bin over the visible range, otherwise the tail swallows every bin
            ax.hist(values, bins=30, range=(0, limit) if clip else None,
                    color=color, edgecolor=SURFACE, linewidth=0.8)
            median = df[column].median()
            ax.axvline(median, color=INK, linewidth=1.2, linestyle="--")
            ax.text(median + limit * 0.015, ax.get_ylim()[1] * 0.92,
                    f"median {median:.0f}", color=INK, fontsize=9)
            ax.set_title(f"Distribution of {label.lower()}")
            ax.set_xlabel(label)
            ax.set_ylabel("Bug fixes")
            ax.yaxis.grid(True, linewidth=0.6)
            ax.set_axisbelow(True)
            show(fig)
            if hidden:
                st.caption(f"{hidden} bug fixes above {limit} are not shown "
                           f"(max {df[column].max():,}). Uncheck to see the full range.")

    st.markdown("**Percentiles**")
    st.dataframe(
        percentile_table(df, ["patch_length", "lines_changed", "num_files_changed", "num_hunks"]),
        width="stretch",
    )

    st.markdown("**Correlation between numerical features**")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(df[NUMERIC_FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=2, linecolor=SURFACE,
                cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    show(fig)

    with st.expander("Key insights", expanded=True):
        st.markdown(
            """
            - Both distributions are strongly **right-skewed**: the median patch is ~23 lines
              and the median churn only ~7 lines, while the mean churn is ~31 — a handful of
              very large fixes stretch the tail.
            - `patch_length` is near-collinear with `lines_added` (0.99), `lines_removed` (0.98)
              and declared functions (0.98) — these carry largely the same signal.
            - `num_files_changed` correlates only weakly with patch size (~0.31): **a large fix
              does not necessarily span many files.**
            - `num_hunks` tracks `num_files_changed` (0.61) and declared classes (0.84), but
              barely tracks `lines_removed` (0.08).
            """
        )


# --------------------------------------------------------------------------
# Repository comparison
# --------------------------------------------------------------------------

with project_tab:
    metric = st.radio("Measure", ["patch_length", "lines_changed"], horizontal=True,
                      format_func=lambda c: "Patch length" if c == "patch_length" else "Lines changed")
    scale = st.radio("Outliers", ["Hide outliers", "Log scale (show all)"], horizontal=True)

    order = df.groupby("project")[metric].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.boxplot(data=df, x="project", y=metric, order=order, ax=ax,
                showfliers=scale.startswith("Log"), color=BLUE, width=0.6,
                linewidth=1, fliersize=3,
                boxprops={"alpha": 0.85}, medianprops={"color": SURFACE, "linewidth": 1.6})
    if scale.startswith("Log"):
        ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"{'Patch length' if metric == 'patch_length' else 'Lines changed'} "
                 f"by repository (ordered by median)")
    ax.set_xlabel("")
    ax.yaxis.grid(True, linewidth=0.6)
    ax.set_axisbelow(True)
    show(fig)

    summary = (
        df.groupby("project")
        .agg(bug_fixes=("bug_id", "size"),
             median_patch_length=("patch_length", "median"),
             median_lines_changed=("lines_changed", "median"),
             median_files_changed=("num_files_changed", "median"),
             comment_rate=("has_comment", "mean"))
        .sort_values("median_patch_length", ascending=False)
    )
    summary["comment_rate"] = (summary["comment_rate"] * 100).round(0)
    st.dataframe(summary, width="stretch",
                 column_config={"comment_rate": st.column_config.NumberColumn(
                     "comment rate (%)", format="%d%%")})

    with st.expander("Key insights", expanded=True):
        st.markdown(
            """
            - Median patch length for most repositories sits between **15 and 35 lines** —
              bug fixes are generally localized rather than sweeping refactors.
            - **PySnooper** has the highest median *patch length*, but **Black** has the highest
              median *code churn*: a larger git patch does not imply more modified code.
            - Black, Keras, Pandas, HTTPie and PySnooper show wide interquartile ranges
              (highly variable fix sizes); spaCy, Luigi and Scrapy are compact and consistent.
            - Pandas holds the most extreme outliers (patches over 2,500 lines), which is why
              the linear view hides fliers and the log view is offered separately.
            """
        )


# --------------------------------------------------------------------------
# Semantic features
# --------------------------------------------------------------------------

with semantic:
    top_n = st.slider("Show top N", 5, 25, 15, step=5)

    left, right = st.columns(2)
    with left:
        exceptions = counter_series(df, "exception_types")
        if exceptions.empty:
            st.info("No exception types in the current selection.")
        else:
            show(hbar(exceptions.sort_values(ascending=False).head(top_n),
                      f"Top {top_n} exception types in bug fixes", "Bug fixes"))
    with right:
        modules = counter_series(df, "modified_modules")
        if modules.empty:
            st.info("No modified modules in the current selection.")
        else:
            show(hbar(modules.sort_values(ascending=False).head(top_n),
                      f"Top {top_n} modified modules", "Bug fixes", color=AQUA))

    st.markdown("**Which exception types are most common in each repository?**")
    exploded = (df.explode("exception_types")
                  .dropna(subset=["exception_types"])
                  .reset_index(drop=True))
    if exploded.empty:
        st.info("No exception types in the current selection.")
    else:
        top10 = exploded.exception_types.value_counts().head(10).index
        top_rows = exploded[exploded.exception_types.isin(top10)]
        cross = pd.crosstab(top_rows["project"], top_rows["exception_types"])
        fig, ax = plt.subplots(figsize=(11, 0.42 * len(cross) + 2.5))
        sns.heatmap(cross, annot=True, fmt="d", cmap="Blues", linewidths=2,
                    linecolor=SURFACE, cbar_kws={"label": "Bug fixes"}, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        show(fig)

    with st.expander("Key insights", expanded=True):
        st.markdown(
            """
            - `ValueError` leads (72 patches), followed by `TypeError`, `KeyError` and
              `NotImplementedError` — a small set of exceptions covers a large share of fixes,
              with a long tail of one-off types.
            - The feature also captures **repository-specific semantics**: `ExtractorError`,
              `InvalidIndexError`, `AbstractMethodError`.
            - Exceptions are **not uniformly distributed** across repositories — `KeyError` and
              `OverflowError` are almost entirely pandas. Combining repository + exception type
              should narrow the search space for repository-aware retrieval.
            - Modified modules point at subsystem-level hot spots (`pandas/core`,
              `pandas/core/indexes`, `keras/engine`) rather than individual files.
            """
        )


# --------------------------------------------------------------------------
# Bug explorer
# --------------------------------------------------------------------------

with explorer:
    col1, col2 = st.columns([1, 2])
    repo = col1.selectbox("Repository", sorted(df.project.unique()))
    repo_bugs = df[df.project == repo].sort_values("patch_length", ascending=False)

    st.dataframe(
        repo_bugs[["bug_id", "python_version", "patch_length", "lines_added", "lines_removed",
                   "num_files_changed", "num_hunks", "has_comment"]],
        width="stretch", hide_index=True,
    )

    bug_id = col2.selectbox("Bug", repo_bugs.bug_id.tolist(),
                            format_func=lambda b: f"{repo} #{b}")
    bug = repo_bugs[repo_bugs.bug_id == bug_id].iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patch length", int(bug.patch_length))
    m2.metric("Lines changed", int(bug.lines_changed))
    m3.metric("Files changed", int(bug.num_files_changed))
    m4.metric("Hunks", int(bug.num_hunks))

    meta1, meta2 = st.columns(2)
    meta1.markdown(
        f"**Python version:** `{bug.python_version}`  \n"
        f"**Buggy commit:** `{bug.buggy_commit_id}`  \n"
        f"**Fixed commit:** `{bug.fixed_commit_id}`"
    )
    meta2.markdown(
        f"**Exceptions:** {', '.join(f'`{e}`' for e in bug.exception_types) or '—'}  \n"
        f"**Modules:** {', '.join(f'`{m}`' for m in bug.modified_modules) or '—'}  \n"
        f"**Test file:** `{bug.test_file}`"
    )

    with st.expander("Patch (unified diff)"):
        st.code(bug.patch, language="diff")


# --------------------------------------------------------------------------
# Feature dictionary
# --------------------------------------------------------------------------

with features:
    st.markdown(
        """
        ### Original dataset features

        | Feature | Type | Description |
        |---|---|---|
        | `project` | Categorical | Python repository containing the bug (pandas, keras, scrapy, …). |
        | `bug_id` | Categorical | Unique bug identifier within a project. |
        | `python_version` | Categorical | Python version used to reproduce the bug. |
        | `buggy_commit_id` | String | Commit hash of the buggy version. |
        | `fixed_commit_id` | String | Commit hash of the fixed version. |
        | `test_file` | String | Test file(s) used to reproduce or validate the fix. |
        | `pythonpath` | String | Optional module search path (populated for a small subset). |
        | `patch` | Text | Unified git diff of the bug fix. |

        ### Engineered features

        | Feature | Type | Extraction logic |
        |---|---|---|
        | `modified_files` | List | From `diff --git` headers. |
        | `num_files_changed` | Integer | `len(modified_files)` |
        | `lines_added` / `lines_removed` | Integer | Lines starting with `+` / `-` (excluding `+++`/`---`). |
        | `lines_changed` | Integer | `lines_added + lines_removed` |
        | `num_hunks` | Integer | Occurrences of `@@`. |
        | `patch_length` | Integer | `len(patch.splitlines())`, including metadata and context. |
        | `has_comment` | Boolean | Regex for `#`, `//`, `/*`, `*/`. |
        | `declared_classes_in_patch` / `declared_functions_in_patch` | List | Regex on `class` / `def`. |
        | `number_of_declared_classes_in_patch` / `..._functions_...` | Integer | Length of the lists above. |
        | `exception_types` | List | Exception names referenced in the patch. |
        | `modified_modules` | List | Directory-level module of each modified file (`.` → `root`). |

        > Declaration counts cover anything appearing in the patch, including git context lines,
        > so they describe the **structural scope** of a patch rather than the exact number of
        > modified definitions.

        ### Data notes
        - `pythonpath` is null for most rows — optional execution metadata, left unchanged.
        - keras bug 12 has an empty patch (all patch features are 0) and is excluded everywhere.
        """
    )

    st.markdown("### Raw data (current filters)")
    st.dataframe(df.drop(columns=["patch"]), width="stretch", hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        df.drop(columns=["patch"]).to_csv(index=False).encode(),
        file_name="bugsinpy_filtered.csv",
        mime="text/csv",
    )
