import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 2rem 1rem; }

    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 600;
        color: #f9fafb;
        line-height: 1;
    }
    .metric-sub {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }

    .fare-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.5rem;
    }
    .fare-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #2a2d3e;
        font-size: 14px;
    }
    .fare-row:last-child { border-bottom: none; font-weight: 600; font-size: 15px; }
    .fare-label { color: #9ca3af; }
    .fare-amount { color: #f9fafb; font-family: 'DM Mono', monospace; }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 12px;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 20px;
    }
    .pill-green  { background: #052e16; color: #4ade80; }
    .pill-amber  { background: #431407; color: #fb923c; }
    .pill-red    { background: #450a0a; color: #f87171; }

    .section-title {
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 1rem;
    }

    div[data-testid="stSlider"] > div { padding: 0; }
    div[data-testid="stSlider"] label { font-size: 13px !important; color: #9ca3af !important; }

    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 14px;
        padding: 0.6rem 1.5rem;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #1d4ed8; }

    h1 { font-size: 22px !important; font-weight: 600 !important; color: #f9fafb !important; }
    h2 { font-size: 16px !important; font-weight: 500 !important; color: #e5e7eb !important; }
    h3 { font-size: 14px !important; font-weight: 500 !important; color: #d1d5db !important; }
    p  { color: #9ca3af !important; font-size: 14px !important; }

    .stTabs [data-baseweb="tab-list"] { background: #1a1d2e; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #6b7280; border-radius: 7px; font-size: 13px; font-weight: 500; padding: 6px 16px; }
    .stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }

    .stSidebar { background: #0d0f1a !important; }
    .stSidebar [data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── MODEL TRAINING ──────────────────────────────────────────────────────────

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 8000

    df = pd.DataFrame({
        'riders':   np.random.randint(1, 100, n),
        'drivers':  np.random.randint(1, 100, n),
        'distance': np.random.uniform(1, 30, n).round(1),
        'hour':     np.random.randint(0, 24, n),
        'weather':  np.random.randint(1, 6, n),
        'day_type': np.random.randint(0, 2, n),   # 0=weekday, 1=weekend
    })

    def make_price(row):
        ratio   = row['riders'] / max(row['drivers'], 1)
        surge   = min(max(ratio * 0.8, 0.8), 3.0)
        base    = 50
        dist_c  = row['distance'] * 8
        peak    = 25 if (7 <= row['hour'] <= 9 or 17 <= row['hour'] <= 20) else (15 if (row['hour'] >= 22 or row['hour'] <= 5) else 0)
        wm      = {1: 0.9, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.3}[row['weather']]
        weekend = 1.1 if row['day_type'] == 1 else 1.0
        return round((base + dist_c + peak) * surge * wm * weekend + np.random.normal(0, 3), 2)

    df['price'] = df.apply(make_price, axis=1)

    X = df[['riders','drivers','distance','hour','weather','day_type']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    preds  = model.predict(X_test)
    mae    = mean_absolute_error(y_test, preds)
    r2     = r2_score(y_test, preds)

    return model, mae, r2, df


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def hour_label(h):
    if h == 0:  return "12 AM"
    if h < 12:  return f"{h} AM"
    if h == 12: return "12 PM"
    return f"{h-12} PM"

def weather_label(w):
    return {1:"Sunny", 2:"Cloudy", 3:"Normal", 4:"Windy", 5:"Heavy Rain"}[w]

def weather_icon(w):
    return {1:"☀️", 2:"⛅", 3:"🌤️", 4:"🌬️", 5:"🌧️"}[w]

def get_surge(riders, drivers):
    ratio = riders / max(drivers, 1)
    return min(max(ratio * 0.8, 0.8), 3.0)

def get_peak_bonus(hour):
    if 7 <= hour <= 9 or 17 <= hour <= 20: return 25
    if hour >= 22 or hour <= 5: return 15
    return 0

def demand_status(riders, drivers):
    ratio = riders / max(drivers, 1)
    if ratio > 1.5: return "High", "pill-red",  "🔴"
    if ratio > 1.0: return "Medium", "pill-amber", "🟡"
    return "Low", "pill-green", "🟢"

def time_slot(hour):
    if 7 <= hour <= 9 or 17 <= hour <= 20: return "Peak Hours", "pill-amber"
    if hour >= 22 or hour <= 5:             return "Late Night",  "pill-amber"
    return "Off-Peak", "pill-green"


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🚗 Ride Conditions")
    st.markdown("---")

    riders   = st.slider("Riders requesting",  1, 100, 40, help="How many people want a ride right now")
    drivers  = st.slider("Drivers available",  1, 100, 50, help="How many drivers are free nearby")
    distance = st.slider("Distance (km)",      1,  30,  8, help="Trip distance in kilometres")
    hour     = st.slider("Hour of day",        0,  23,  9, format="%d", help="Current hour (0 = midnight)")
    weather  = st.slider("Weather",            1,   5,  3, help="1 = Sunny  →  5 = Heavy Rain")
    day_type = st.radio("Day type", ["Weekday", "Weekend"], horizontal=True)

    st.markdown("---")
    st.markdown(f"**Selected:** {hour_label(hour)} · {weather_icon(weather)} {weather_label(weather)}")
    st.markdown(f"**Day:** {'Weekend 🎉' if day_type == 'Weekend' else 'Weekday 💼'}")


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────

with st.spinner("Training XGBoost model on 8,000 samples..."):
    model, mae, r2, df = train_model()


# ─── PREDICT ─────────────────────────────────────────────────────────────────

day_int = 1 if day_type == "Weekend" else 0
input_df = pd.DataFrame([[riders, drivers, distance, hour, weather, day_int]],
                         columns=['riders','drivers','distance','hour','weather','day_type'])
predicted = round(float(model.predict(input_df)[0]), 0)

surge     = get_surge(riders, drivers)
base      = 50
dist_cost = round(distance * 8, 0)
peak      = get_peak_bonus(hour)
wm        = {1:0.9, 2:0.95, 3:1.0, 4:1.05, 5:1.3}[weather]
wknd      = 1.1 if day_int == 1 else 1.0

surge_contrib  = round((base + dist_cost + peak) * (surge - 1) * wm * wknd)
weather_contrib= round((base + dist_cost + peak) * surge * (wm - 1) * wknd)
wknd_contrib   = round((base + dist_cost + peak) * surge * wm * (wknd - 1))

d_status, d_class, d_icon = demand_status(riders, drivers)
t_slot, t_class = time_slot(hour)


# ─── HEADER ──────────────────────────────────────────────────────────────────

st.markdown("## 🚗 Dynamic Pricing Engine")
st.markdown("Real-time fare prediction using XGBoost — adjust conditions in the sidebar.")
st.markdown("---")


# ─── METRIC CARDS ────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Fare</div>
        <div class="metric-value">₹{int(predicted)}</div>
        <div class="metric-sub">XGBoost prediction</div>
    </div>""", unsafe_allow_html=True)

with c2:
    surge_color = "#f87171" if surge > 2 else "#fb923c" if surge > 1.2 else "#4ade80"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Surge Multiplier</div>
        <div class="metric-value" style="color:{surge_color}">{surge:.1f}x</div>
        <div class="metric-sub">{riders} riders / {drivers} drivers</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Demand Level</div>
        <div class="metric-value" style="font-size:20px; margin-top:6px;">
            <span class="pill {d_class}">{d_icon} {d_status}</span>
        </div>
        <div class="metric-sub">ratio: {riders/max(drivers,1):.2f}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Time Slot</div>
        <div class="metric-value" style="font-size:20px; margin-top:6px;">
            <span class="pill {t_class}">{t_slot}</span>
        </div>
        <div class="metric-sub">{hour_label(hour)} · {weather_icon(weather)} {weather_label(weather)}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── TABS ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["  Fare Breakdown  ", "  Price Factors  ", "  Model Insights  "])


# TAB 1 — FARE BREAKDOWN
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### Fare Components")
        rows = [
            ("Base fare",                               f"₹{base}"),
            (f"Distance ({distance} km × ₹8)",         f"₹{int(dist_cost)}"),
            (f"Peak hour bonus ({hour_label(hour)})",   f"₹{peak}"),
            (f"Surge ({surge:.2f}x)",                   f"₹{max(surge_contrib, 0)}"),
            (f"Weather ({weather_label(weather)})",     f"₹{max(weather_contrib, 0)}"),
            (f"Weekend bonus" if day_int else "Weekday (no bonus)", f"₹{max(wknd_contrib, 0)}"),
        ]
        html = '<div class="fare-card">'
        for label, amount in rows:
            html += f'<div class="fare-row"><span class="fare-label">{label}</span><span class="fare-amount">{amount}</span></div>'
        html += f'<div class="fare-row"><span>Total Fare</span><span class="fare-amount">₹{int(predicted)}</span></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    with col_right:
        st.markdown("#### Component Chart")
        labels  = ["Base", "Distance", "Peak", "Surge", "Weather", "Weekend"]
        values  = [base, int(dist_cost), peak, max(surge_contrib,0), max(weather_contrib,0), max(wknd_contrib,0)]
        colors  = ["#6b7280","#3b82f6","#f59e0b","#ef4444","#10b981","#8b5cf6"]

        fig = go.Figure(go.Bar(
            x=values, y=labels,
            orientation='h',
            marker_color=colors,
            marker_line_width=0,
            text=[f"₹{v}" for v in values],
            textposition='outside',
            textfont=dict(color="#9ca3af", size=12, family="DM Mono"),
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=60, t=10, b=10),
            height=260,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(color="#9ca3af", size=12), gridcolor='rgba(255,255,255,0.04)'),
            bargap=0.35,
        )
        st.plotly_chart(fig, use_container_width=True)


# TAB 2 — PRICE FACTORS
with tab2:
    st.markdown("#### How price changes with each factor")

    col_a, col_b = st.columns(2)

    with col_a:
        rider_range = list(range(5, 101, 5))
        prices_vs_riders = []
        for r in rider_range:
            s = get_surge(r, drivers)
            p = (base + dist_cost + peak) * s * wm * wknd
            prices_vs_riders.append(round(p))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=rider_range, y=prices_vs_riders,
            mode='lines+markers',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=5, color='#3b82f6'),
            fill='tozeroy',
            fillcolor='rgba(59,130,246,0.08)',
            name="fare"
        ))
        fig2.add_vline(x=riders, line_color="#f59e0b", line_dash="dot", line_width=1.5,
                       annotation_text=f"  current: {riders}", annotation_font_color="#f59e0b", annotation_font_size=11)
        fig2.update_layout(
            title=dict(text="Fare vs. Rider Count", font=dict(color="#9ca3af", size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=220, margin=dict(l=0,r=0,t=35,b=0),
            xaxis=dict(title="riders", tickfont=dict(color="#6b7280",size=11), gridcolor='rgba(255,255,255,0.04)', title_font=dict(color="#6b7280")),
            yaxis=dict(tickprefix="₹", tickfont=dict(color="#6b7280",size=11), gridcolor='rgba(255,255,255,0.04)'),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        dist_range = list(range(1, 31))
        prices_vs_dist = []
        for d in dist_range:
            dc = d * 8
            p  = (base + dc + peak) * surge * wm * wknd
            prices_vs_dist.append(round(p))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=dist_range, y=prices_vs_dist,
            mode='lines+markers',
            line=dict(color='#10b981', width=2),
            marker=dict(size=5, color='#10b981'),
            fill='tozeroy',
            fillcolor='rgba(16,185,129,0.08)',
        ))
        fig3.add_vline(x=distance, line_color="#f59e0b", line_dash="dot", line_width=1.5,
                       annotation_text=f"  current: {distance}km", annotation_font_color="#f59e0b", annotation_font_size=11)
        fig3.update_layout(
            title=dict(text="Fare vs. Distance", font=dict(color="#9ca3af", size=13)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=220, margin=dict(l=0,r=0,t=35,b=0),
            xaxis=dict(title="km", tickfont=dict(color="#6b7280",size=11), gridcolor='rgba(255,255,255,0.04)', title_font=dict(color="#6b7280")),
            yaxis=dict(tickprefix="₹", tickfont=dict(color="#6b7280",size=11), gridcolor='rgba(255,255,255,0.04)'),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Fare heatmap: riders × drivers")
    heat_riders  = list(range(5, 101, 5))
    heat_drivers = list(range(5, 101, 5))
    z = []
    for dr in heat_drivers:
        row_vals = []
        for ri in heat_riders:
            s = get_surge(ri, dr)
            p = (base + dist_cost + peak) * s * wm * wknd
            row_vals.append(round(p))
        z.append(row_vals)

    fig4 = go.Figure(go.Heatmap(
        z=z,
        x=[str(r) for r in heat_riders],
        y=[str(d) for d in heat_drivers],
        colorscale=[[0,'#1a3a2a'],[0.3,'#10b981'],[0.7,'#f59e0b'],[1,'#ef4444']],
        colorbar=dict(tickprefix="₹", tickfont=dict(color="#9ca3af", size=11)),
    ))
    fig4.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=300, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(title="riders", tickfont=dict(color="#6b7280",size=10), title_font=dict(color="#6b7280")),
        yaxis=dict(title="drivers available", tickfont=dict(color="#6b7280",size=10), title_font=dict(color="#6b7280")),
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Green = cheap, Yellow = moderate, Red = expensive surge pricing")


# TAB 3 — MODEL INSIGHTS
with tab3:
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Mean Absolute Error", f"₹{mae:.2f}", help="Average prediction error on test set")
    col_m2.metric("R² Score", f"{r2:.4f}", help="1.0 = perfect, higher is better")
    col_m3.metric("Training samples", "8,000", help="Synthetic dataset size")

    st.markdown("#### Feature importance")
    features    = ['riders','drivers','distance','hour','weather','day_type']
    importances = model.feature_importances_
    feat_df     = pd.DataFrame({'feature': features, 'importance': importances})
    feat_df     = feat_df.sort_values('importance', ascending=True)

    fig5 = go.Figure(go.Bar(
        x=feat_df['importance'],
        y=feat_df['feature'],
        orientation='h',
        marker_color='#3b82f6',
        marker_line_width=0,
        text=[f"{v:.3f}" for v in feat_df['importance']],
        textposition='outside',
        textfont=dict(color="#9ca3af", size=11, family="DM Mono"),
    ))
    fig5.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=260, margin=dict(l=0,r=60,t=10,b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(color="#9ca3af",size=13), gridcolor='rgba(255,255,255,0.04)'),
        bargap=0.35
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("Higher importance = the model relies more on that feature to make predictions.")

    st.markdown("#### Prediction vs. actual (test set sample)")
    X_samp = df.sample(300, random_state=1)
    y_pred_samp = model.predict(X_samp[['riders','drivers','distance','hour','weather','day_type']])

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=X_samp['price'], y=y_pred_samp,
        mode='markers',
        marker=dict(color='#3b82f6', opacity=0.5, size=5),
        name="predictions"
    ))
    mn, mx = df['price'].min(), df['price'].max()
    fig6.add_trace(go.Scatter(
        x=[mn,mx], y=[mn,mx],
        mode='lines',
        line=dict(color='#ef4444', dash='dot', width=1.5),
        name="perfect line"
    ))
    fig6.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=280, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(title="actual price (₹)", tickfont=dict(color="#6b7280",size=11), gridcolor='rgba(255,255,255,0.04)', title_font=dict(color="#6b7280")),
        yaxis=dict(title="predicted price (₹)", tickfont=dict(color="#6b7280",size=11), gridcolor='rgba(255,255,255,0.04)', title_font=dict(color="#6b7280")),
        legend=dict(font=dict(color="#9ca3af"), bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig6, use_container_width=True)
    st.caption("Points close to the red dotted line = accurate predictions.")


# ─── FOOTER ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:12px; color:#374151;'>Dynamic Pricing Engine · XGBoost + Streamlit · Built for resume portfolio</p>",
    unsafe_allow_html=True
)

# to run  -- python -m streamlit run app.py