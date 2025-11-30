#!/usr/bin/env python3
"""
Alpaca WebSocket → Kafka Producer
Streams real-time 1-minute bar data for selected stocks into Kafka.
"""

import os
import json
import logging
import asyncio
import time
import requests
from datetime import datetime
from kafka import KafkaProducer
from alpaca.data.live import StockDataStream

# ------------------------------------------
# Configuration
# ------------------------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "alpaca-bars"

# Choose a small but meaningful set of symbols
SYMBOLS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]

# ------------------------------------------
# Logging Setup
# ------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("producer")

# ------------------------------------------
# Producer Class
# ------------------------------------------
class AlpacaKafkaProducer:
    def __init__(self):
        # Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: (k.encode("utf-8") if k else None),
        )

        # Alpaca stream
        self.stream = StockDataStream(ALPACA_API_KEY, ALPACA_API_SECRET)

        # reconnect/backoff configuration
        self.max_retries = 10
        self.initial_backoff = 1  # seconds

    async def handle_bar(self, bar):
        """
        Called every time Alpaca sends a new bar.
        We convert the bar object into a dict and send it to Kafka.
        """
        msg = {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp.isoformat(),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "received_at": datetime.utcnow().isoformat(),
        }

        # Send to Kafka
        self.producer.send(KAFKA_TOPIC, key=bar.symbol, value=msg)
        logger.info(f"Sent: {bar.symbol} @ {bar.close}")

    async def start(self):
        """Subscribe to symbols + begin WebSocket stream."""
        logger.info(f"Subscribing to: {', '.join(SYMBOLS)}")

        # Quick pre-check: validate Alpaca credentials via REST before starting long-lived websocket
        ok, msg = self._check_alpaca_credentials()
        if not ok:
            logger.error("Alpaca credential check failed: %s", msg)
            logger.error("Aborting stream start. Verify ALPACA_API_KEY / ALPACA_API_SECRET and data subscription.")
            return

        for symbol in SYMBOLS:
            self.stream.subscribe_bars(self.handle_bar, symbol)

        # Controlled reconnect loop with exponential backoff to avoid rapid reconnects and HTTP 429
        attempts = 0
        backoff = self.initial_backoff
        while attempts < self.max_retries:
            try:
                await self.stream._run_forever()
                # if _run_forever returns normally, break out
                break
            except ValueError as e:
                # alpaca library raises ValueError('auth failed') on bad auth
                msg = str(e)
                if 'auth failed' in msg.lower():
                    logger.error("Alpaca auth failed: %s", msg)
                    logger.error("Verify API keys and that your account has the required data subscription.")
                    return
                else:
                    logger.warning("ValueError during stream: %s; retrying in %s seconds", msg, backoff)
            except Exception as e:
                logger.warning("Error during stream runtime: %s; retrying in %s seconds", e, backoff)

            attempts += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

        if attempts >= self.max_retries:
            logger.error("Exceeded max reconnect attempts (%s). Exiting.", self.max_retries)

    def _check_alpaca_credentials(self):
        """Synchronous check of Alpaca REST account endpoint to validate credentials.

        Returns (ok: bool, message: str).
        This helps detect bad API keys or permission/subscription issues before opening the websocket.
        """
        key = ALPACA_API_KEY
        secret = ALPACA_API_SECRET
        if not key or not secret:
            return False, 'Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment.'

        headers = {
            'APCA-API-KEY-ID': key,
            'APCA-API-SECRET-KEY': secret,
        }

        # Try both live and paper account endpoints before failing
        endpoints = [
            'https://api.alpaca.markets/v2/account',
            'https://paper-api.alpaca.markets/v2/account',
        ]

        failures = []
        for url in endpoints:
            try:
                r = requests.get(url, headers=headers, timeout=10)
            except Exception as exc:
                failures.append((url, f'network error: {exc}'))
                continue

            if r.status_code == 200:
                return True, f'OK ({url})'

            # collect non-200 responses and continue trying other endpoints
            failures.append((url, f'HTTP {r.status_code}'))

        # If we get here, neither endpoint returned 200
        msg = '; '.join([f'{u} -> {s}' for u, s in failures])
        return False, f'Credential check failed for endpoints: {msg}'


# ------------------------------------------
# Entrypoint
# ------------------------------------------
async def main():
    producer = AlpacaKafkaProducer()
    await producer.start()

if __name__ == "__main__":
    asyncio.run(main())


