import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "final_data.csv"
MIN_TRAIN_ROWS = 10


def _mode_or_unknown(series: pd.Series) -> str:
	mode = series.mode()
	if mode.empty:
		return "Unknown"
	return str(mode.iloc[0])


def _safe_importance(model, n_features: int) -> np.ndarray:
	if hasattr(model, "feature_importances_"):
		return np.asarray(model.feature_importances_)
	return np.zeros(n_features)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
	header = pd.read_csv(path, nrows=0)
	available = set(header.columns)
	required = ["account", "classification", "closed_pnl", "size_usd", "side"]
	optional = ["date", "timestamp"]
	usecols = [c for c in required + optional if c in available]

	df = pd.read_csv(path, usecols=usecols)

	if "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce")
	elif "timestamp" in df.columns:
		df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("D")
	else:
		raise ValueError("Missing both 'date' and 'timestamp' columns in dataset.")

	df["classification"] = df["classification"].fillna("Unknown").astype(str)
	df["closed_pnl"] = pd.to_numeric(df["closed_pnl"], errors="coerce").fillna(0.0)
	df["size_usd"] = pd.to_numeric(df["size_usd"], errors="coerce").fillna(0.0)
	df["side_num"] = (
		df["side"].astype(str).str.upper().str.contains("BUY").astype(float)
		if "side" in df.columns
		else 0.5
	)

	df = df.dropna(subset=["account", "date"]).copy()
	return df


@st.cache_data(show_spinner=False)
def build_daily_features(path: str) -> pd.DataFrame:
	df = load_data(path)

	daily = (
		df.groupby(["account", "date"], as_index=False)
		.agg(
			day_pnl=("closed_pnl", "sum"),
			trades=("closed_pnl", "size"),
			avg_position=("size_usd", "mean"),
			pnl_volatility=("closed_pnl", "std"),
			buy_ratio=("side_num", "mean"),
			sentiment=("classification", _mode_or_unknown),
		)
		.sort_values(["account", "date"])
	)

	daily["pnl_volatility"] = daily["pnl_volatility"].fillna(0.0)
	daily["sentiment_score"] = (
		daily["sentiment"].str.upper().map({"FEAR": 0.0, "GREED": 1.0}).fillna(0.5)
	)

	rank = daily["day_pnl"].rank(method="first")
	daily["profit_bucket"] = pd.qcut(
		rank,
		q=3,
		labels=["Loss", "Neutral", "Profit"],
	)

	behavior_cols = [
		"day_pnl",
		"trades",
		"avg_position",
		"pnl_volatility",
		"buy_ratio",
		"sentiment_score",
	]

	for col in behavior_cols:
		daily[f"prev_{col}"] = daily.groupby("account")[col].shift(1)

	daily["target_bucket_next"] = daily.groupby("account")["profit_bucket"].shift(-1)
	daily["target_vol_next"] = daily.groupby("account")["pnl_volatility"].shift(-1)

	return daily


@st.cache_resource(show_spinner=True)
def train_models(path: str):
	daily = build_daily_features(path)
	features = [
		"prev_day_pnl",
		"prev_trades",
		"prev_avg_position",
		"prev_pnl_volatility",
		"prev_buy_ratio",
		"prev_sentiment_score",
	]
	model_notes = []

	cls_df = daily.dropna(subset=features + ["target_bucket_next"]).copy()
	if len(cls_df) >= MIN_TRAIN_ROWS and cls_df["target_bucket_next"].nunique() > 1:
		X_cls = cls_df[features]
		y_cls = cls_df["target_bucket_next"].astype(str)

		stratify = y_cls if y_cls.value_counts().min() >= 2 else None
		Xc_train, Xc_test, yc_train, yc_test = train_test_split(
			X_cls,
			y_cls,
			test_size=0.2,
			random_state=42,
			stratify=stratify,
		)

		classifier = RandomForestClassifier(
			n_estimators=140,
			max_depth=10,
			random_state=42,
			n_jobs=-1,
			class_weight="balanced_subsample",
		)
		classifier.fit(Xc_train, yc_train)
		yc_pred = classifier.predict(Xc_test)
		cls_accuracy = accuracy_score(yc_test, yc_pred)
	else:
		if cls_df.empty:
			fill_bucket = "Neutral"
		else:
			fill_bucket = str(cls_df["target_bucket_next"].astype(str).mode().iloc[0])
		classifier = DummyClassifier(strategy="constant", constant=fill_bucket)
		classifier.fit(np.zeros((1, len(features))), [fill_bucket])
		cls_accuracy = 1.0 if len(cls_df) <= 1 else np.nan
		model_notes.append("Classification model fallback: not enough labeled rows/classes for train-test split.")

	reg_df = daily.dropna(subset=features + ["target_vol_next"]).copy()
	if len(reg_df) >= MIN_TRAIN_ROWS:
		X_reg = reg_df[features]
		y_reg = reg_df["target_vol_next"]

		Xr_train, Xr_test, yr_train, yr_test = train_test_split(
			X_reg,
			y_reg,
			test_size=0.2,
			random_state=42,
		)

		regressor = RandomForestRegressor(
			n_estimators=120,
			max_depth=10,
			random_state=42,
			n_jobs=-1,
		)
		regressor.fit(Xr_train, yr_train)
		yr_pred = regressor.predict(Xr_test)
		reg_mae = mean_absolute_error(yr_test, yr_pred)
	else:
		if reg_df.empty:
			fill_vol = 0.0
		else:
			fill_vol = float(reg_df["target_vol_next"].mean())
		regressor = DummyRegressor(strategy="constant", constant=fill_vol)
		regressor.fit(np.zeros((1, len(features))), [fill_vol])
		reg_mae = 0.0 if len(reg_df) <= 1 else np.nan
		model_notes.append("Volatility model fallback: not enough rows for stable train-test split.")

	return {
		"daily": daily,
		"features": features,
		"classifier": classifier,
		"regressor": regressor,
		"cls_accuracy": cls_accuracy,
		"reg_mae": reg_mae,
		"notes": model_notes,
	}


@st.cache_data(show_spinner=False)
def trader_profiles(path: str) -> pd.DataFrame:
	df = load_data(path)
	profile = (
		df.groupby("account", as_index=False)
		.agg(
			avg_position=("size_usd", "mean"),
			trades=("closed_pnl", "size"),
			total_pnl=("closed_pnl", "sum"),
			pnl_volatility=("closed_pnl", "std"),
			greed_ratio=("classification", lambda s: (s.str.upper() == "GREED").mean()),
			buy_ratio=("side_num", "mean"),
		)
		.fillna(0.0)
	)
	profile["win_rate"] = (
		df.assign(win=df["closed_pnl"] > 0)
		.groupby("account")["win"]
		.mean()
		.reindex(profile["account"])
		.fillna(0.0)
		.to_numpy()
	)
	return profile


@st.cache_data(show_spinner=False)
def cluster_traders(path: str, n_clusters: int) -> pd.DataFrame:
	profile = trader_profiles(path).copy()
	if profile.empty:
		profile["cluster"] = pd.Series(dtype=int)
		return profile

	n_clusters = int(max(1, min(n_clusters, len(profile))))
	cols = [
		"avg_position",
		"trades",
		"total_pnl",
		"pnl_volatility",
		"greed_ratio",
		"buy_ratio",
		"win_rate",
	]

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(profile[cols])

	model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
	profile["cluster"] = model.fit_predict(X_scaled)
	return profile


def main():
	st.set_page_config(page_title="Trader Analytics", layout="wide")
	st.title("Trader Analytics: Prediction + Archetypes")
	st.caption("Next-day profitability/volatility modeling with sentiment and behavior features.")

	try:
		artifacts = train_models(DATA_PATH)
	except Exception as err:
		st.error(f"Failed to build analytics pipeline: {err}")
		return

	daily = artifacts["daily"]
	features = artifacts["features"]
	classifier = artifacts["classifier"]
	regressor = artifacts["regressor"]

	tab1, tab2, tab3 = st.tabs(["Overview", "Predictive Model", "Trader Clusters"])

	with tab1:
		c1, c2, c3 = st.columns(3)
		c1.metric("Daily Rows", f"{len(daily):,}")
		c2.metric("Accounts", f"{daily['account'].nunique():,}")
		c3.metric("Classifier Accuracy", f"{artifacts['cls_accuracy']:.2%}")

		sentiment_mix = daily["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
		st.write("Sentiment mix")
		st.bar_chart(sentiment_mix.set_index("sentiment"))

		pnl_view = (
			daily.groupby("date", as_index=False)["day_pnl"].mean().sort_values("date").tail(200)
		)
		st.write("Average daily PnL (latest 200 dates)")
		st.line_chart(pnl_view.set_index("date")["day_pnl"])

	with tab2:
		st.subheader("Next-day prediction")
		if artifacts["notes"]:
			for note in artifacts["notes"]:
				st.warning(note)
		if np.isnan(artifacts["cls_accuracy"]):
			st.write("Profitability bucket accuracy: N/A (fallback model)")
		else:
			st.write(f"Profitability bucket accuracy: {artifacts['cls_accuracy']:.2%}")
		if np.isnan(artifacts["reg_mae"]):
			st.write("Volatility MAE: N/A (fallback model)")
		else:
			st.write(f"Volatility MAE: {artifacts['reg_mae']:.4f}")

		bucket_importance = _safe_importance(classifier, len(features))
		vol_importance = _safe_importance(regressor, len(features))

		importance_df = pd.DataFrame(
			{
				"feature": features,
				"bucket_importance": bucket_importance,
				"vol_importance": vol_importance,
			}
		).sort_values("bucket_importance", ascending=False)
		st.write("Feature importance")
		st.dataframe(importance_df, use_container_width=True)

		st.subheader("Quick what-if")
		sample = daily[features].dropna().tail(1)
		if not sample.empty:
			baseline = sample.iloc[0].to_dict()
		else:
			baseline = {f: 0.0 for f in features}

		col_a, col_b, col_c = st.columns(3)
		prev_day_pnl = col_a.number_input("Prev day PnL", value=float(baseline["prev_day_pnl"]))
		prev_trades = col_b.number_input("Prev trades", value=float(baseline["prev_trades"]), min_value=0.0)
		prev_avg_position = col_c.number_input(
			"Prev avg position", value=float(baseline["prev_avg_position"]), min_value=0.0
		)

		col_d, col_e, col_f = st.columns(3)
		prev_pnl_volatility = col_d.number_input(
			"Prev pnl volatility",
			value=float(baseline["prev_pnl_volatility"]),
			min_value=0.0,
		)
		prev_buy_ratio = col_e.slider(
			"Prev buy ratio",
			min_value=0.0,
			max_value=1.0,
			value=float(np.clip(baseline["prev_buy_ratio"], 0.0, 1.0)),
		)
		prev_sentiment_score = col_f.slider(
			"Prev sentiment score (Fear=0, Greed=1)",
			min_value=0.0,
			max_value=1.0,
			value=float(np.clip(baseline["prev_sentiment_score"], 0.0, 1.0)),
		)

		x_input = pd.DataFrame(
			[
				{
					"prev_day_pnl": prev_day_pnl,
					"prev_trades": prev_trades,
					"prev_avg_position": prev_avg_position,
					"prev_pnl_volatility": prev_pnl_volatility,
					"prev_buy_ratio": prev_buy_ratio,
					"prev_sentiment_score": prev_sentiment_score,
				}
			]
		)

		pred_bucket = classifier.predict(x_input)[0]
		pred_vol = regressor.predict(x_input)[0]
		st.success(f"Predicted next-day bucket: {pred_bucket}")
		st.info(f"Predicted next-day volatility: {pred_vol:.4f}")

	with tab3:
		st.subheader("Behavioral archetypes")
		profiles = trader_profiles(DATA_PATH)
		if profiles.empty:
			st.info("No trader profiles available for clustering.")
			return

		n_accounts = int(profiles["account"].nunique())
		max_clusters = min(8, n_accounts)
		if max_clusters <= 1:
			st.info("Need at least 2 accounts for clustering.")
			st.dataframe(profiles, use_container_width=True)
			return

		default_clusters = min(3, max_clusters)
		n_clusters = st.slider(
			"Number of clusters",
			min_value=2,
			max_value=max_clusters,
			value=default_clusters,
		)

		clustered = cluster_traders(DATA_PATH, n_clusters)
		cluster_summary = (
			clustered.groupby("cluster", as_index=False)
			.agg(
				traders=("account", "size"),
				avg_position=("avg_position", "mean"),
				total_pnl=("total_pnl", "mean"),
				win_rate=("win_rate", "mean"),
				pnl_volatility=("pnl_volatility", "mean"),
			)
			.sort_values("cluster")
		)

		st.dataframe(cluster_summary, use_container_width=True)
		st.write("Average total PnL by cluster")
		st.bar_chart(cluster_summary.set_index("cluster")["total_pnl"])


if __name__ == "__main__":
	main()