"""
Business Insights generator engine.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_insights(forecast_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate business insights and actionable recommendations from forecast data.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame containing forecast predictions in the "Forecast" column,
        and optionally calendar columns like "date", "day_of_week", "month", etc.

    Returns
    -------
    dict
        A dictionary containing stats, trends, weekend analysis, and recommendations.
    """
    if "Forecast" not in forecast_df.columns:
        raise ValueError("Forecast column not found in the input DataFrame.")

    forecast_values = forecast_df["Forecast"].values
    if len(forecast_values) == 0:
        return {}

    total_sales = float(np.sum(forecast_values))
    avg_sales = float(np.mean(forecast_values))
    max_sales = float(np.max(forecast_values))
    min_sales = float(np.min(forecast_values))

    # Try to find the peak date/index
    peak_date_str = ""
    if "date" in forecast_df.columns:
        peak_idx = np.argmax(forecast_values)
        peak_date_str = str(forecast_df.loc[peak_idx, "date"])
    else:
        peak_idx = np.argmax(forecast_values)
        peak_date_str = f"Day {peak_idx + 1}"

    # 1. Trend Analysis (Linear Regression on forecast values)
    x = np.arange(len(forecast_values)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, forecast_values)
    slope = lr.coef_[0]

    # Calculate percentage change from start to end of trend line
    trend_start = lr.intercept_
    trend_end = trend_start + slope * (len(forecast_values) - 1)
    if trend_start != 0:
        pct_change = ((trend_end - trend_start) / trend_start) * 100
    else:
        pct_change = 0.0

    if pct_change > 2.0:
        trend_direction = "Increasing"
        trend_icon = "📈"
        trend_summary = f"Demand is projected to grow by {pct_change:.1f}% over the forecast horizon."
    elif pct_change < -2.0:
        trend_direction = "Decreasing"
        trend_icon = "📉"
        trend_summary = f"Demand is projected to contract by {abs(pct_change):.1f}% over the forecast horizon."
    else:
        trend_direction = "Stable"
        trend_icon = "➡️"
        trend_summary = "Demand is expected to remain stable with minor seasonal fluctuations."

    # 2. Weekend vs Weekday analysis
    weekend_avg = None
    weekday_avg = None
    weekend_ratio = 1.0

    # Determine weekends
    is_weekend_mask = None
    if "is_weekend" in forecast_df.columns:
        is_weekend_mask = forecast_df["is_weekend"] == 1
    elif "day_of_week" in forecast_df.columns:
        is_weekend_mask = forecast_df["day_of_week"].isin([5, 6])
    elif "date" in forecast_df.columns:
        dates = pd.to_datetime(forecast_df["date"])
        is_weekend_mask = dates.dt.dayofweek.isin([5, 6])

    if is_weekend_mask is not None and is_weekend_mask.any() and (~is_weekend_mask).any():
        weekend_avg = float(np.mean(forecast_values[is_weekend_mask]))
        weekday_avg = float(np.mean(forecast_values[~is_weekend_mask]))
        if weekday_avg != 0:
            weekend_ratio = weekend_avg / weekday_avg

    # 3. Actionable Recommendations
    recommendations = []

    # Inventory recommendation
    if trend_direction == "Increasing":
        rec_pct = max(5.0, min(20.0, pct_change))
        recommendations.append({
            "title": "Increase Safety Stock",
            "description": f"Due to an upward trend ({trend_icon} growing by {pct_change:.1f}%), we recommend increasing inventory levels by {rec_pct:.1f}% to prevent stockouts during peak periods.",
            "type": "warning"
        })
    elif trend_direction == "Decreasing":
        rec_pct = max(5.0, min(20.0, abs(pct_change)))
        recommendations.append({
            "title": "Reduce Holding Levels",
            "description": f"Demand is trending downward ({trend_icon} falling by {abs(pct_change):.1f}%). Adjust purchase orders downward by {rec_pct:.1f}% to minimize storage holding costs and prevent overstocking.",
            "type": "info"
        })
    else:
        recommendations.append({
            "title": "Maintain Baseline Safety Stock",
            "description": "Demand remains stable. Keep current inventory rules but verify stock limits before holidays or major seasonal events.",
            "type": "success"
        })

    # Weekend / Staffing recommendation
    if weekend_avg is not None and weekend_ratio > 1.1:
        diff_pct = (weekend_ratio - 1.0) * 100
        recommendations.append({
            "title": "Optimize Weekend Staffing",
            "description": f"Weekend sales are projected to be {diff_pct:.1f}% higher than weekdays on average. Allocate extra front-of-house staff and increase replenishment frequency on Friday evening/Saturday morning.",
            "type": "warning"
        })
    elif weekend_avg is not None and weekend_ratio < 0.9:
        diff_pct = (1.0 - weekend_ratio) * 100
        recommendations.append({
            "title": "Weekday Operations Focus",
            "description": f"Weekday sales outperform weekends by {diff_pct:.1f}% on average. Shift promotions, vendor deliveries, and staff shifts to mid-week peak times to maximize conversion.",
            "type": "info"
        })

    # Promotion recommendation
    if weekday_avg is not None and weekday_avg > 0:
        low_sales_days = []
        # Find day of week averages
        dow_col = None
        if "day_of_week" in forecast_df.columns:
            dow_col = "day_of_week"
        elif "date" in forecast_df.columns:
            forecast_df = forecast_df.copy()
            forecast_df["day_of_week"] = pd.to_datetime(forecast_df["date"]).dt.dayofweek
            dow_col = "day_of_week"

        if dow_col is not None:
            dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_averages = forecast_df.groupby(dow_col)["Forecast"].mean()
            lowest_dows = dow_averages.nsmallest(2).index.tolist()
            low_days_str = " and ".join([dow_names[d] for d in lowest_dows])
            recommendations.append({
                "title": "Targeted Mid-week Promotions",
                "description": f"Historical low-demand periods occur on {low_days_str}. Consider launching short flash-sales or customer loyalty multipliers on these days to level out weekly sales volatility.",
                "type": "success"
            })

    # Peak anomaly warning
    if max_sales > 1.5 * avg_sales:
        recommendations.append({
            "title": "Prepare for Peak Demand Spike",
            "description": f"A major spike of {max_sales:.1f} units (expected around {peak_date_str}) is {((max_sales - avg_sales) / avg_sales * 100):.1f}% above the average. Pre-position buffer stock for key items to avoid delivery delays.",
            "type": "warning"
        })

    return {
        "summary": {
            "total_sales": total_sales,
            "avg_sales": avg_sales,
            "max_sales": max_sales,
            "min_sales": min_sales,
            "peak_date": peak_date_str,
            "trend_direction": trend_direction,
            "trend_summary": trend_summary,
            "trend_icon": trend_icon,
            "pct_change": pct_change,
        },
        "weekend_analysis": {
            "weekend_avg": weekend_avg,
            "weekday_avg": weekday_avg,
            "weekend_ratio": weekend_ratio,
        },
        "recommendations": recommendations,
    }
