"""
Unit tests for the business insights generator.
"""

import pandas as pd
import pytest

from utils.insights import generate_insights


def test_generate_insights_columns_check():
    df = pd.DataFrame({"sales": [10, 20, 30]})
    with pytest.raises(ValueError):
        generate_insights(df)


def test_generate_insights_empty():
    df = pd.DataFrame({"Forecast": []})
    res = generate_insights(df)
    assert res == {}


def test_generate_insights_upward_trend():
    # Linear upward trend
    forecasts = [100 + i * 5 for i in range(30)]  # 100 to 245
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "Forecast": forecasts,
        "is_weekend": [0, 0, 0, 0, 0, 1, 1] * 4 + [0, 0]  # Weekends on Saturday/Sunday
    })

    res = generate_insights(df)

    assert "summary" in res
    assert res["summary"]["trend_direction"] == "Increasing"
    assert res["summary"]["peak_date"].startswith("2025-01-30")  # Last day (highest sales)
    assert len(res["recommendations"]) > 0

    # Ensure safety stock recommendation is present
    inventory_rec = [r for r in res["recommendations"] if "Safety Stock" in r["title"]]
    assert len(inventory_rec) == 1
    assert "inventory" in inventory_rec[0]["description"].lower()


def test_generate_insights_downward_trend():
    # Linear downward trend
    forecasts = [200 - i * 5 for i in range(30)]  # 200 to 55
    dates = pd.date_range(start="2025-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "Forecast": forecasts,
        "day_of_week": dates.dayofweek
    })

    res = generate_insights(df)

    assert res["summary"]["trend_direction"] == "Decreasing"
    
    # Ensure holding levels recommendation is present
    holding_rec = [r for r in res["recommendations"] if "Holding Levels" in r["title"]]
    assert len(holding_rec) == 1
