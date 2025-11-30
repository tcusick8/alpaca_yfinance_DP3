#!/usr/bin/env python3
"""
Kafka → DuckDB Consumer
Consumes live Alpaca bar data and stores it in DuckDB.
"""

import os
import json
import logging
import duckdb
from kafka import KafkaConsumer

# ------------------------------------------
# Configuration
# ------------------------------------------
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC = "alpaca-bars"
DUCKDB_PATH = "data/market_data.duckdb"

# ------------------------------------------
# Logging
# ------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("consumer")

# ------------------------------------------
# Consumer
# ------------------------------------------
class DuckDBConsumer:
    def __init__(self):
        os.makedirs("data", exist_ok=True)

        # Kafka consumer
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
        )

        # DuckDB
        self.conn = duckdb.connect(DUCKDB_PATH)
        self._init_table()

    def _init_table(self):
        """Create bars table if not exists."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                symbol TEXT,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                received_at TIMESTAMP
            );
        """)

    def run(self):
        """Consume messages forever."""
        logger.info("Consumer started — waiting for messages...")

        for msg in self.consumer:
            bar = msg.value

            # Insert into DuckDB
            self.conn.execute(
                """
                INSERT INTO bars
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    bar["symbol"],
                    bar["timestamp"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    bar["volume"],
                    bar["received_at"],
                ],
            )

            logger.info(f"Stored: {bar['symbol']} @ {bar['close']}")

def main():
    DuckDBConsumer().run()

if __name__ == "__main__":
    main()


