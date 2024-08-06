import asyncio
import json
import os
from websockets import connect
import numpy as np

websocket_uri = "wss://fstream.binance.com/ws/!forceOrder@arr"
filename = "binance.csv"

# Check if the CSV file exists, otherwise, create it and write the header.
if not os.path.isfile(filename):
    with open(filename, "w") as f:
        f.write(",".join(["symbol", "side", "order_type", "time_in_force",
                           "original_quantity", "price", "average_price",
                           "order_status", "order_last_filled_quantity",
                           "order_filled_accumulated_quantity",
                           "order_trade_time"]) + "\n")

# Global list to store prices for support/resistance calculations
price_history = []

def calculate_fibonacci_levels(high, low):
    """
    Calculate Fibonacci retracement levels based on given high and low.
    """
    difference = high - low
    level_0 = low
    level_1 = low + difference * 0.236
    level_2 = low + difference * 0.382
    level_3 = low + difference * 0.618
    level_4 = high
    return {
        "level_0": level_0,
        "level_1": level_1,
        "level_2": level_2,
        "level_3": level_3,
        "level_4": level_4,
    }

def determine_market_mood(current_price, previous_price):
    """
    Determine market mood based on price changes.
    """
    return "UP" if current_price > previous_price else "DOWN"

async def binance_liquidations(uri, filename):
    async for websocket in connect(uri):
        try:
            while True:
                msg = await websocket.recv()
                print(msg)  # Print raw message for debugging
                data = json.loads(msg)["o"]

                # Process the liquidation data
                symbol = data["s"]
                side = data["S"]
                order_type = data["o"]
                time_in_force = data["f"]
                original_quantity = data["q"]
                price = data["p"]
                average_price = data["ap"]
                order_status = data["X"]
                order_last_filled_quantity = data["l"]
                order_filled_accumulated_quantity = data["z"]
                order_trade_time = data["T"]

                # Save to CSV
                msg_to_write = [symbol, side, order_type, time_in_force,
                                original_quantity, price, average_price,
                                order_status, order_last_filled_quantity,
                                order_filled_accumulated_quantity,
                                order_trade_time]
                with open(filename, "a") as f:
                    f.write(",".join(map(str, msg_to_write)) + "\n")
                
                # Track price for Fibonacci levels
                current_price = float(price)
                price_history.append(current_price)

                # Calculate Fibonacci levels once we have enough data
                if len(price_history) >= 2:  # Ensure we have at least two prices
                    high_price = np.max(price_history[-20:])  # Last 20 prices as a range
                    low_price = np.min(price_history[-20:])   # Last 20 prices as a range
                    fib_levels = calculate_fibonacci_levels(high_price, low_price)
                    print(f"Fibonacci levels: {fib_levels}")

                    # Determine market mood (assumed based on last two prices)
                    market_mood = determine_market_mood(current_price, price_history[-2])
                    print(f"Current Market Mood: {market_mood}")

        except Exception as e:
            print(e)
            continue

# Run the WebSocket listener
asyncio.run(binance_liquidations(websocket_uri, filename))