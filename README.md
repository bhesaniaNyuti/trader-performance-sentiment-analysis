# 📊 Trader Performance vs Market Sentiment Analysis

## 🔍 Overview

This project analyzes how market sentiment (Fear vs Greed) influences trader behavior and performance using historical trading data. The goal is to identify patterns and derive actionable trading strategies.

---

## 📁 Dataset

* **Market Sentiment Dataset**: Fear/Greed classification by date
* **Trader Data**: Trade-level data including PnL, size, side, and timestamps

---

## ⚙️ Setup & How to Run

### 1. Clone Repository

```bash
git clone https://github.com/bhesaniaNyuti/trader-performance-sentiment-analysis
cd trader-performance-sentiment-analysis
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run Analysis

```bash
python round0.py
```

### 4. (Optional) Run Dashboard

```bash
streamlit run app.py
```

---

## 📊 Output (Charts & Tables)

### Key Visualizations:

* PnL Distribution across Fear vs Greed
* Long vs Short Ratio by Sentiment
* Segment-wise Performance (High vs Low Position Size)

### Key Tables:

* Performance metrics (mean PnL, win rate)
* Drawdown comparison
* Trader segmentation results

---

## 🧠 Methodology

1. **Data Cleaning & Preparation**

   * Standardized column names
   * Converted timestamps to daily level
   * Removed duplicate sentiment entries
   * Mapped sentiment data to trades using date alignment

2. **Feature Engineering**

   * Win/Loss indicator
   * Position size (USD-based)
   * Daily PnL per trader
   * Trade frequency and behavior metrics

3. **Analysis**

   * Compared performance (PnL, win rate, drawdown) across sentiment regimes
   * Evaluated behavioral changes (trade size, frequency, long/short bias)
   * Segmented traders based on position size and activity

---

## 🔥 Key Insights

* **High Drawdowns During Greed**
  Traders experience significant drawdowns (~ -11,202), indicating overexposure during optimistic market phases.

* **Balanced Long/Short Behavior**
  BUY (~49%) and SELL (~51%) ratios remain nearly equal, suggesting performance depends on execution rather than direction.

* **High Risk–High Return Pattern**
  Large position trades generate significantly higher PnL (~93 vs ~4), but also contribute to higher volatility and risk.

---

## 💡 Strategy Recommendations

* **Strategy 1: Risk Control in Greed**
  Reduce position sizes and enforce strict stop-loss rules during Greed phases to minimize drawdowns.

* **Strategy 2: Selective High Exposure**
  Allocate larger positions only to high-confidence trades, as profitability is concentrated in fewer high-risk trades.

* **Additional Rule:**
  Avoid overtrading during Greed periods and focus on quality setups.

---

## 🚀 Bonus Work

* Built a **Random Forest model** to predict trade profitability
* Performed **K-Means clustering** to identify trader archetypes
* Designed a **Streamlit dashboard** for interactive exploration

---

## 📌 Conclusion

The analysis highlights a strong relationship between market sentiment, trader behavior, and performance. Effective risk management and adaptive strategies are essential for navigating sentiment-driven markets.

---

