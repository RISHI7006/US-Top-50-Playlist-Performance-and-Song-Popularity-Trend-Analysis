"""Atlantic Playlist Performance Explorer
=========================================
A modern Streamlit web application for exploring the US Top-50 playlist
analytics originally developed in the Atlantic playlist notebook.

Core modules
------------
1. Playlist timeline explorer
2. Song ranking trend charts
3. Artist dominance leaderboard
4. Popularity vs rank scatter plots
5. Explicit vs non-explicit performance panels

Plus a background ML / Deep-Learning insight tab.

Run:  streamlit run app.py
"""
from __future__ import annotations


import os
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Atlantic Playlist Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject a modern dark-accented theme
st.markdown(
    """
    <style>
    :root {
        --bg: #0f1117;
        --card: #181a22;
        --accent: #1DB954;
        --accent2: #1ed760;
        --text: #e8eaf0;
        --muted: #9ca3b4;
    }
    .stApp { background: var(--bg); }
    .stSidebar .stSlider > div > div { background: var(--accent); }
    h1, h2, h3 { letter-spacing: -.02em; }
    .metric-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem; font-weight: 700; color: var(--accent2);
    }
    .metric-card .label {
        font-size: .8rem; color: var(--muted); margin-top: .3rem;
    }
    .tab-desc { color: var(--muted); font-size: .9rem; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / "atlantic_clean.csv"
    generator = BASE_DIR / "generate_data.py"

    if not csv_path.exists():
        if not generator.exists():
            st.error(
                "atlantic_clean.csv is missing and generate_data.py was not found. "
                "Upload the CSV with the app or include generate_data.py in the deployment."
            )
            st.stop()
        with st.spinner("Generating playlist dataset (first run only)…"):
            result = subprocess.run(
                [os.sys.executable, str(generator)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not csv_path.exists():
                st.error("Could not generate the playlist dataset.")
                if result.stderr:
                    st.code(result.stderr)
                st.stop()

    df = pd.read_csv(csv_path)

    required_columns = {
        "date", "position", "song", "artist", "popularity",
        "duration_ms", "is_explicit", "album_type", "total_tracks"
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        st.error(f"CSV is missing required columns: {', '.join(missing)}")
        st.stop()

    # Accept the expected DD-MM-YYYY format, while still handling ISO dates.
    raw_dates = df["date"].astype(str).str.strip()
    parsed_dates = pd.to_datetime(raw_dates, format="%d-%m-%Y", errors="coerce")
    fallback_dates = pd.to_datetime(raw_dates, errors="coerce")
    df["date"] = parsed_dates.fillna(fallback_dates)

    # Convert numeric columns safely.
    for col in ["position", "popularity", "duration_ms", "total_tracks"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Robust boolean parsing: avoid bool("False") == True.
    df["is_explicit"] = (
        df["is_explicit"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True, "1": True, "yes": True, "y": True,
            "false": False, "0": False, "no": False, "n": False
        })
        .fillna(False)
        .astype(bool)
    )

    df["duration_min"] = df["duration_ms"] / 60000

    # Remove unusable rows so Streamlit never receives NaT dates.
    df = df.dropna(
        subset=["date", "position", "song", "artist", "popularity"]
    ).copy()

    if df.empty:
        st.error("The dataset contains no valid rows/dates after cleaning.")
        st.stop()

    df["position"] = df["position"].astype(int)
    df = df.sort_values(["date", "position"]).reset_index(drop=True)

    # Row-level features
    df = df.sort_values(["song", "artist", "date"]).reset_index(drop=True)
    df["popularity_trend_score"] = (
        df.groupby(["song", "artist"])["popularity"]
          .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    df["prev_position"] = df.groupby(["song", "artist"])["position"].shift(1)
    df["rank_delta"] = df["prev_position"] - df["position"]
    df = df.sort_values(["date", "position"]).reset_index(drop=True)
    return df


@st.cache_data
def song_level_features(df: pd.DataFrame) -> pd.DataFrame:
    sf = df.groupby(["song", "artist"]).agg(
        days_on_chart=("date", "nunique"),
        avg_rank=("position", "mean"),
        best_rank=("position", "min"),
        rank_volatility=("position", "std"),
        avg_popularity=("popularity", "mean"),
        avg_duration_min=("duration_min", "mean"),
        total_tracks=("total_tracks", "first"),
        album_type=("album_type", "first"),
        is_explicit=("is_explicit", "first"),
        first_seen=("date", "min"),
        last_seen=("date", "max"),
    ).reset_index()
    sf["rank_volatility"] = sf["rank_volatility"].fillna(0)
    return sf


df = load_data()
song_df = song_level_features(df)

# ---------------------------------------------------------------------------
# Sidebar — global filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Filters")

min_timestamp = df["date"].min()
max_timestamp = df["date"].max()

if pd.isna(min_timestamp) or pd.isna(max_timestamp):
    st.error("No valid dates are available in the dataset.")
    st.stop()

min_date = min_timestamp.date()
max_date = max_timestamp.date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Streamlit can return one date or two dates depending on the widget state.
if isinstance(date_range, (list, tuple)):
    if len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date, end_date = date_range[0], date_range[1]
else:
    start_date = end_date = date_range

d0 = pd.Timestamp(start_date)
d1 = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

all_artists = sorted(df["artist"].unique())
selected_artists = st.sidebar.multiselect(
    "Artist filter", all_artists, default=[]
)

all_songs = sorted(df["song"].unique())
selected_songs = st.sidebar.multiselect(
    "Song filter", all_songs, default=[]
)

rank_min, rank_max = st.sidebar.slider(
    "Rank range", 1, 50, (1, 50)
)

album_choice = st.sidebar.radio(
    "Album type", ["All", "album", "single", "compilation"], index=0
)

# Apply filters
mask = pd.Series(True, index=df.index)
mask &= (df["date"] >= d0) & (df["date"] <= d1)
if selected_artists:
    mask &= df["artist"].isin(selected_artists)
if selected_songs:
    mask &= df["song"].isin(selected_songs)
mask &= (df["position"] >= rank_min) & (df["position"] <= rank_max)
if album_choice != "All":
    mask &= (df["album_type"] == album_choice)

fdf = df[mask].copy()


# ---------------------------------------------------------------------------
# Sidebar — PDF download
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Export Report")

if st.sidebar.button("Generate PDF Report", use_container_width=True):
    try:
        from pdf_report import generate_pdf
    except ImportError:
        st.sidebar.error("PDF export is unavailable because pdf_report.py is missing.")
        st.stop()

    parts = []
    parts.append(f"Date: {start_date} to {end_date}")
    if selected_artists:
        parts.append(f"Artists: {', '.join(selected_artists[:5])}"
                     + ("…" if len(selected_artists) > 5 else ""))
    if selected_songs:
        parts.append(f"Songs: {len(selected_songs)} selected")
    parts.append(f"Rank: {rank_min}–{rank_max}")
    parts.append(f"Album: {album_choice}")
    filters_desc = " | ".join(parts)

    with st.sidebar.spinner("Building PDF…"):
        pdf_bytes = generate_pdf(fdf, filters_desc)
    st.sidebar.success("Report ready!")
    st.sidebar.download_button(
        label="⬇️ Download PDF",
        data=pdf_bytes,
        file_name=f"atlantic_playlist_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎵 Atlantic Playlist Performance Explorer")
st.markdown(
    f"<p class='tab-desc'>US Top-50 daily snapshots · "
    f"{len(fdf):,} filtered rows · {fdf['date'].nunique()} days · "
    f"{fdf['song'].nunique()} unique songs · {fdf['artist'].nunique()} artists</p>",
    unsafe_allow_html=True,
)

# Quick metric cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='metric-card'><div class='value'>{fdf['song'].nunique()}</div>"
                "<div class='label'>Unique Songs</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><div class='value'>{fdf['artist'].nunique()}</div>"
                "<div class='label'>Unique Artists</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><div class='value'>{fdf['popularity'].mean():.1f}</div>"
                "<div class='label'>Avg Popularity</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><div class='value'>{fdf['position'].mean():.1f}</div>"
                "<div class='label'>Avg Rank</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📅 Timeline Explorer",
    "📈 Ranking Trends",
    "🏆 Artist Leaderboard",
    "🔵 Popularity vs Rank",
    "🚫 Explicit vs Clean",
    "🤖 ML / Deep Learning",
])

# ---- Tab 1: Timeline Explorer ----------------------------------------------
with tab1:
    st.subheader("Playlist Timeline Explorer")
    st.markdown("<p class='tab-desc'>Pick any date in the filtered range to see the full "
                "Top-50 snapshot for that day.</p>", unsafe_allow_html=True)

    available_dates = sorted(fdf["date"].dt.date.unique()) if len(fdf) else []
    if not available_dates:
        st.warning("No data matches the current filters.")
    else:
        chosen = st.select_slider("Select a date", options=available_dates,
                                  value=available_dates[len(available_dates) // 2])
        day_df = fdf[fdf["date"].dt.date == chosen].sort_values("position")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.dataframe(
                day_df[["position", "song", "artist", "popularity",
                        "album_type", "is_explicit", "duration_min"]],
                use_container_width=True,
                height=520,
                hide_index=True,
            )
        with col_b:
            st.metric("Songs on this day", len(day_df))
            st.metric("Avg popularity", f"{day_df['popularity'].mean():.0f}")
            st.metric("Explicit tracks", int(day_df["is_explicit"].sum()))

            fig = go.Figure(go.Bar(
                x=day_df["popularity"].values[::-1],
                y=[f"#{p} {s[:18]}" for p, s in zip(day_df["position"], day_df["song"])][::-1],
                orientation="h",
                marker=dict(color=day_df["popularity"].values[::-1],
                            colorscale="Greens"),
            ))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Popularity", showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 2: Ranking Trends -------------------------------------------------
with tab2:
    st.subheader("Song Ranking Trend Charts")
    st.markdown("<p class='tab-desc'>Track how individual songs move up or down the chart "
                "over time.</p>", unsafe_allow_html=True)

    if fdf.empty:
        st.warning("No data for current filters.")
    else:
        top_songs = (
            fdf.groupby(["song", "artist"])["date"]
            .nunique()
            .nlargest(12)
            .reset_index()
        )
        song_opts = [f"{r.song} — {r.artist}" for r in top_songs.itertuples()]
        picked = st.multiselect("Choose songs to plot", song_opts, default=song_opts[:4])

        if picked:
            fig = go.Figure()
            for label in picked:
                s, a = label.split(" — ")
                sub = fdf[(fdf["song"] == s) & (fdf["artist"] == a)].sort_values("date")
                if not sub.empty:
                    fig.add_trace(go.Scatter(
                        x=sub["date"], y=sub["position"],
                        mode="lines+markers", name=label[:30],
                        line=dict(width=2),
                    ))
            fig.update_yaxes(autorange="reversed", title="Position (1 = top)")
            fig.update_xaxes(title="Date")
            fig.update_layout(height=450, legend_font_size=11,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        # Rank movement distribution
        st.markdown("#### Day-over-Day Rank Movement")
        deltas = fdf["rank_delta"].dropna()
        if len(deltas):
            fig2 = px.histogram(deltas, nbins=60, labels={"value": "Rank Δ (+=up)"},
                                color_discrete_sequence=["#1DB954"])
            fig2.update_layout(height=300, showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)


# ---- Tab 3: Artist Leaderboard ---------------------------------------------
with tab3:
    st.subheader("Artist Dominance Leaderboard")
    st.markdown("<p class='tab-desc'>Artists ranked by total chart presence within the "
                "filtered window.</p>", unsafe_allow_html=True)

    if fdf.empty:
        st.warning("No data for current filters.")
    else:
        artist_agg = (
            fdf.groupby("artist")
            .agg(appearances=("date", "count"),
                 days=("date", "nunique"),
                 songs=("song", "nunique"),
                 avg_popularity=("popularity", "mean"),
                 best_rank=("position", "min"))
            .reset_index()
            .sort_values("appearances", ascending=False)
            .head(25)
        )
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.dataframe(
                artist_agg.style.format({
                    "avg_popularity": "{:.1f}",
                    "best_rank": "{:.0f}",
                }),
                use_container_width=True,
                hide_index=True,
                height=500,
            )
        with col_r:
            fig = px.bar(
                artist_agg.head(15),
                x="appearances", y="artist", orientation="h",
                color="avg_popularity", color_continuous_scale="Greens",
                labels={"appearances": "Chart slots", "artist": ""},
            )
            fig.update_layout(height=500, yaxis=dict(autorange="reversed"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ---- Tab 4: Popularity vs Rank scatter -------------------------------------
with tab4:
    st.subheader("Popularity vs Rank Scatter Plots")
    st.markdown("<p class='tab-desc'>Each point is a song-day.  See the relationship "
                "between playlist position and Spotify popularity.</p>", unsafe_allow_html=True)

    if fdf.empty:
        st.warning("No data for current filters.")
    else:
        sample = fdf.sample(min(4000, len(fdf)), random_state=1)
        fig = px.scatter(
            sample, x="position", y="popularity",
            color="is_explicit", color_discrete_map={True: "#E1306C", False: "#1DB954"},
            opacity=0.6, hover_data=["song", "artist", "date"],
            labels={"position": "Playlist Rank", "popularity": "Popularity"},
        )
        fig.update_layout(height=480,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        # Trend line (aggregate)
        agg = fdf.groupby("position")["popularity"].mean().reset_index()
        fig2 = go.Figure(go.Scatter(
            x=agg["position"], y=agg["popularity"],
            mode="lines+markers", line=dict(color="#1ed760", width=3),
            name="Avg popularity per rank",
        ))
        fig2.update_layout(height=280, xaxis_title="Rank", yaxis_title="Avg Popularity",
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)


# ---- Tab 5: Explicit vs Non-Explicit ---------------------------------------
with tab5:
    st.subheader("Explicit vs Non-Explicit Performance")
    st.markdown("<p class='tab-desc'>Compare chart performance between explicit and clean "
                "tracks.</p>", unsafe_allow_html=True)

    if fdf.empty:
        st.warning("No data for current filters.")
    else:
        exp = fdf[fdf["is_explicit"]]
        clean = fdf[~fdf["is_explicit"]]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Explicit tracks", f"{len(exp):,}",
                      delta=f"{exp['popularity'].mean():.1f} avg pop" if len(exp) else "")
        with m2:
            st.metric("Clean tracks", f"{len(clean):,}",
                      delta=f"{clean['popularity'].mean():.1f} avg pop" if len(clean) else "")
        with m3:
            diff = (exp["popularity"].mean() - clean["popularity"].mean()) if len(exp) and len(clean) else 0
            st.metric("Popularity gap (exp − clean)", f"{diff:+.1f}")

        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            if len(exp):
                fig.add_trace(go.Box(y=exp["popularity"], name="Explicit",
                                     marker_color="#E1306C"))
            if len(clean):
                fig.add_trace(go.Box(y=clean["popularity"], name="Clean",
                                     marker_color="#1DB954"))
            fig.update_layout(height=380, yaxis_title="Popularity",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            # Album type split by explicit
            pivot = fdf.groupby(["album_type", "is_explicit"]).size().reset_index(name="count")
            pivot["is_explicit"] = pivot["is_explicit"].map({True: "Explicit", False: "Clean"})
            fig2 = px.bar(pivot, x="album_type", y="count", color="is_explicit",
                          barmode="group", color_discrete_map={"Explicit": "#E1306C", "Clean": "#1DB954"},
                          labels={"album_type": "Album type", "count": "Count"})
            fig2.update_layout(height=380, showlegend=True,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)

        # Rank distribution comparison
        st.markdown("#### Average Rank by Content Type")
        rank_cmp = fdf.groupby("is_explicit")["position"].mean().reset_index()
        rank_cmp["is_explicit"] = rank_cmp["is_explicit"].map({True: "Explicit", False: "Clean"})
        fig3 = px.bar(rank_cmp, x="is_explicit", y="position",
                      color="is_explicit", color_discrete_map={"Explicit": "#E1306C", "Clean": "#1DB954"},
                      labels={"position": "Avg Rank (lower = better)"})
        fig3.update_layout(height=260, showlegend=False,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)


# ---- Tab 6: ML / Deep Learning ---------------------------------------------
with tab6:
    st.subheader("🤖 Machine Learning & Deep Learning Insights")
    st.markdown("<p class='tab-desc'>Two background models predict song popularity from "
                "playlist features.  Training runs once and is cached.</p>", unsafe_allow_html=True)

    try:
        from ml_models import prepare_features, train_random_forest, train_neural_network
    except ImportError as e:
        st.warning(f"ML models are unavailable: {e}")
        st.info("The dashboard tabs still work. Add ml_models.py and its dependencies to enable this tab.")
        st.stop()

    ml_source = fdf if len(fdf) > 200 else df

    try:
        X, y = prepare_features(ml_source)
    except Exception as e:
        st.error(f"Could not prepare ML features: {e}")
        st.stop()

    @st.cache_resource
    def get_models(X_arr, y_arr):
        rf = train_random_forest(X_arr, y_arr)
        try:
            dl = train_neural_network(X_arr, y_arr)
        except Exception:
            dl = None
        return rf, dl

    with st.spinner("Training models (cached after first run)…"):
        try:
            rf_result, dl_result = get_models(X, y)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("#### 🌲 Core ML — Random Forest")
        st.metric("R² score", f"{rf_result['r2']:.3f}")
        st.metric("RMSE", f"{rf_result['rmse']:.2f}")
        st.markdown("**Feature Importances**")
        imp_df = pd.DataFrame(
            sorted(rf_result["importances"].items(), key=lambda kv: -kv[1]),
            columns=["Feature", "Importance"],
        )
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=["#1DB954"])
        fig_imp.update_layout(height=280, yaxis=dict(autorange="reversed"),
                              showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=10))
        st.plotly_chart(fig_imp, use_container_width=True)

    with mc2:
        st.markdown("#### 🧠 Deep Learning — PyTorch MLP")
        if dl_result is not None:
            st.metric("R² score", f"{dl_result['r2']:.3f}")
            st.metric("RMSE", f"{dl_result['rmse']:.2f}")
        else:
            st.warning("PyTorch is not installed — deep learning model skipped.")

    # Prediction vs actual scatter for both models
    st.markdown("#### Predicted vs Actual Popularity")
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=rf_result["y_test"], y=rf_result["y_pred"],
        mode="markers", name="Random Forest",
        marker=dict(color="#1DB954", size=4, opacity=0.5),
    ))
    if dl_result is not None:
        fig_pv.add_trace(go.Scatter(
            x=dl_result["y_test"], y=dl_result["y_pred"],
            mode="markers", name="Neural Network",
            marker=dict(color="#E1306C", size=4, opacity=0.5),
        ))
    lim = [0, 100]
    fig_pv.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                line=dict(dash="dash", color="gray"),
                                showlegend=False))
    fig_pv.update_layout(height=420, xaxis_title="Actual", yaxis_title="Predicted",
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(t=10))
    st.plotly_chart(fig_pv, use_container_width=True)
