"""从最早时间提取半年数据用于 LSTM 训练。"""

import datetime

import polars as pl

p1 = pl.read_parquet("data/converted/tushare_news_2021_today_part1.parquet")
p2 = pl.read_parquet("data/converted/tushare_news_2021_today_part2.parquet")
df = pl.concat([p1, p2])

dt_col = pl.col("datetime").str.to_datetime()
df = df.with_columns(dt_col.alias("dt"))

# 提取 Python datetime 对象
min_dt: datetime.datetime = df.select(pl.col("dt").min()).to_series()[0]
max_dt: datetime.datetime = df.select(pl.col("dt").max()).to_series()[0]
end_dt = min_dt + datetime.timedelta(weeks=100)

subset = df.filter(pl.col("dt") >= min_dt, pl.col("dt") < end_dt)

print(f"原始总条数: {len(df):,}")
print(f"最早: {min_dt.date()}")
print(f"最晚: {max_dt.date()}")
print(f"提取范围: {min_dt.date()} ~ {end_dt.date()}")
print(f"提取条数: {len(subset):,}")

out = "data/converted/tushare_news_lstm_training.parquet"
subset.drop("dt").write_parquet(out)
print(f"已保存到 {out}")
