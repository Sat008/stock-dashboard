# chart_tools.py

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Fibonacci Retracement
# ---------------------------
def plot_fibonacci_retracement(df, high, low):
    fib_levels = {
        '0.0%': high,
        '23.6%': high - 0.236*(high-low),
        '38.2%': high - 0.382*(high-low),
        '50.0%': high - 0.5*(high-low),
        '61.8%': high - 0.618*(high-low),
        '78.6%': high - 0.786*(high-low),
        '100%': low,
    }
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'], label="Close")
    for label, level in fib_levels.items():
        ax.hlines(level, df.index.min(), df.index.max(), linestyles="--", label=label)
        ax.text(df.index.max(), level, f" {label} {level:.2f}", va="center")
    ax.legend()
    return fig


# ---------------------------
# Fibonacci Projection (Extension)
# ---------------------------
def plot_fibonacci_projection(df, pointA, pointB, pointC):
    # Extension from AB projected from C
    ab = pointB - pointA
    proj_levels = {
        '100%': pointC + ab,
        '161.8%': pointC + 1.618*ab,
        '261.8%': pointC + 2.618*ab,
    }
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'])
    for label, level in proj_levels.items():
        ax.hlines(level, df.index.min(), df.index.max(), linestyles="--", label=label)
        ax.text(df.index.max(), level, f" {label} {level:.2f}", va="center")
    ax.legend()
    return fig


# ---------------------------
# Fibonacci Fan
# ---------------------------
def plot_fibonacci_fan(df, start_price, end_price, start_date, end_date):
    ratios = [0.382, 0.5, 0.618]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'])
    dx = (end_date - start_date).days
    dy = end_price - start_price
    for r in ratios:
        slope = dy * r / dx
        line = [start_price + slope*(i) for i in range(dx+1)]
        ax.plot(df.index[:dx+1], line, "--", label=f"Fan {int(r*100)}%")
    ax.legend()
    return fig


# ---------------------------
# Elliott Wave (simplified)
# ---------------------------
def plot_elliott_wave(df, points):
    """
    points: list of 5 prices for waves 1–5
    """
    if len(points) != 5:
        raise ValueError("Elliott wave requires 5 points")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'])
    ax.plot(df.index[:5], points, marker="o", color="red")
    for i, p in enumerate(points, 1):
        ax.text(df.index[i-1], p, f"W{i}")
    return fig


# ---------------------------
# Gartley Pattern (simplified)
# ---------------------------
def plot_gartley_pattern(df, points):
    """
    points: list of 5 prices [X, A, B, C, D]
    """
    if len(points) != 5:
        raise ValueError("Gartley pattern requires 5 points")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'])
    ax.plot(df.index[:5], points, marker="o", color="blue")
    labels = ["X","A","B","C","D"]
    for lbl, p, idx in zip(labels, points, range(len(points))):
        ax.text(df.index[idx], p, lbl)
    return fig


# ---------------------------
# Trend Lines
# ---------------------------
def plot_trend_lines(df, lines):
    """
    lines: list of tuples [(x1,y1,x2,y2), ...]
    """
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df.index, df['Close'])
    for (x1,y1,x2,y2) in lines:
        ax.plot([df.index[x1], df.index[x2]], [y1,y2], "r--")
    return fig
