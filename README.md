# Alpaca + yfinance DP3

Short project README and quick start for the Alpaca + yfinance data project used in the DS3022 final project.

**What this repo contains**
- `historical/` — scripts to fetch historical price data (yfinance), clean it and write canonical tables into DuckDB.
- `stream/` — producer/consumer streaming example using Alpaca (producer) and Kafka (consumer) with local persistence into DuckDB.
- `visualizations/` — exported PNGs and report figures (moved here from the sample project for delivery).
- `requirements.txt` — Python dependencies used by the scripts.

Purpose: provide a repeatable, local pipeline to collect market data (stream and or historical), persist it in DuckDB analytical stores, and produce reproducible visualizations and analysis.

Quick start (macOS / zsh)
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create an environment file with credentials (do NOT commit your keys):
```bash
cp .env.example .env
# edit .env and set ALPACA_API_KEY, ALPACA_API_SECRET, KAFKA_BOOTSTRAP_SERVERS, etc.
```

Running the historical pipeline (offline reproducible):
```bash
cd historical
python hist_fetch_data.py    # fetches via yfinance and writes to historical_market_data.duckdb
python hist_clean_and_visualize.py   # cleans and generates visuals (writes cleaned tables to DuckDB)
```

Running the streaming demo (requires Alpaca credentials and a Kafka broker):
```bash
cd stream
# start Kafka locally or point to a broker; then:
python producer.py   # connects to Alpaca (paper/live), sends bars to Kafka
python consumer.py   # consumes from Kafka and writes to DuckDB
```

Where data lives
- Canonical analytical store: `historical_market_data.duckdb` (project root). Tables used by scripts include `bars`, `bars_cleaned`, and `fetch_metadata`.
- Visual artifacts for reports are in `visualizations/`.

Notes & best practices
- Do NOT commit `.env` or local DuckDB files. A `.gitignore` is included; if you moved files between repos, double-check for accidentally committed secrets and rotate keys if needed.
- If you store very large PNGs (>100MB) use Git LFS.

Troubleshooting
- If you see Alpaca auth errors: check `ALPACA_API_KEY`/`ALPACA_API_SECRET` in `.env`, confirm whether you're using the paper or live endpoint, and confirm subscription / data access on Alpaca.
- If DuckDB reads/writes fail: ensure the current process has write permissions to the repo root and that no other process has locked the `.duckdb` file.

Contributing
- This repo is a delivery/transport of the data pipeline from the DS3022 project. If you want changes (e.g., change where visuals are written), modify the scripts in `historical/` and `stream/` and follow the Quick Start steps above.

Contact / Credits
- Original project and author: Thomas Cusick (repo migrated from the DS3022 project).
