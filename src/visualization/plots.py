"""
plots.py

Visualization functions for exploratory data analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPORT_DIR = Path("reports/figures")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename: str):
    """
    Save current matplotlib figure.
    """

    plt.tight_layout()
    plt.savefig(REPORT_DIR / filename, dpi=300)
    plt.close()


def plot_sales_trend(df: pd.DataFrame):
    """
    Plot daily sales trend.
    """

    plt.figure(figsize=(12, 6))

    sales = df.groupby("date")["sales"].sum()

    plt.plot(sales.index, sales.values)

    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Sales")

    save_plot("sales_trend.png")


def plot_monthly_sales(df: pd.DataFrame):
    """
    Plot average monthly sales.
    """

    monthly = (
        df.groupby(df["date"].dt.month)["sales"]
        .mean()
        .sort_index()
    )

    plt.figure(figsize=(10, 5))

    plt.bar(monthly.index, monthly.values)

    plt.title("Average Monthly Sales")
    plt.xlabel("Month")
    plt.ylabel("Average Sales")

    save_plot("monthly_sales.png")


def plot_yearly_sales(df: pd.DataFrame):
    """
    Plot yearly sales.
    """

    yearly = (
        df.groupby(df["date"].dt.year)["sales"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    plt.plot(yearly.index, yearly.values, marker="o")

    plt.title("Yearly Sales")
    plt.xlabel("Year")
    plt.ylabel("Total Sales")

    save_plot("yearly_sales.png")


def plot_sales_distribution(df: pd.DataFrame):
    """
    Histogram of sales.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(df["sales"], bins=30)

    plt.title("Sales Distribution")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")

    save_plot("sales_distribution.png")


def plot_boxplot(df: pd.DataFrame):
    """
    Boxplot of sales.
    """

    plt.figure(figsize=(6, 5))

    plt.boxplot(df["sales"])

    plt.title("Sales Boxplot")

    save_plot("sales_boxplot.png")


def plot_heatmap(df: pd.DataFrame):
    """
    Correlation heatmap.
    """

    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="Blues"
    )

    plt.title("Correlation Heatmap")

    save_plot("correlation_heatmap.png")


def generate_all_plots(df: pd.DataFrame):
    """
    Generate every EDA visualization.
    """

    print("\nGenerating visualizations...")

    plot_sales_trend(df)
    plot_monthly_sales(df)
    plot_yearly_sales(df)
    plot_sales_distribution(df)
    plot_boxplot(df)
    plot_heatmap(df)

    print("All plots saved successfully.")