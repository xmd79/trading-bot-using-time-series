import time
from binance.client import Client as BinanceClient

# Load credentials from file
with open("credentials.txt", "r") as f:
    api_key = f.readline().strip()
    api_secret = f.readline().strip()

# Instantiate Binance client
client = BinanceClient(api_key, api_secret)

def get_price(pair):
    try:
        return float(client.get_symbol_ticker(symbol=pair)['price'])
    except Exception as e:
        print(f"Error fetching price for {pair}: {e}")
        return None

def check_arbitrage_opportunity():
    while True:
        try:
            # Fetch prices
            btc_usdc_price = get_price('BTCUSDC')
            eth_btc_price = get_price('ETHBTC')
            eth_usdc_price = get_price('ETHUSDC')

            # Continue if all prices are successfully fetched
            if btc_usdc_price and eth_btc_price and eth_usdc_price:
                # Calculate effective prices
                btc_eth_price = 1 / eth_btc_price  # Convert ETH/BTC to BTC/ETH
                effective_btc_usdc_via_eth = btc_eth_price * eth_usdc_price

                # Calculate percentage profits
                profit_direct = btc_usdc_price - effective_btc_usdc_via_eth
                profit_direct_percentage = (profit_direct / effective_btc_usdc_via_eth) * 100

                profit_reverse = effective_btc_usdc_via_eth - btc_usdc_price
                profit_reverse_percentage = (profit_reverse / btc_usdc_price) * 100

                # Check for arbitrage opportunity
                if btc_usdc_price > effective_btc_usdc_via_eth:
                    print(f"Arbitrage Opportunity: BTC -> ETH -> USDC")
                    print(f"BTC/USDC Price: {btc_usdc_price}")
                    print(f"Effective BTC/USDC via ETH: {effective_btc_usdc_via_eth}")
                    print(f"Potential Profit: {profit_direct_percentage:.2f}%")

                elif effective_btc_usdc_via_eth > btc_usdc_price:
                    print(f"Arbitrage Opportunity: USDC -> ETH -> BTC")
                    print(f"BTC/USDC Price: {btc_usdc_price}")
                    print(f"Effective BTC/USDC via ETH: {effective_btc_usdc_via_eth}")
                    print(f"Potential Profit: {profit_reverse_percentage:.2f}%")

            # Wait before checking again
            time.sleep(1)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

# Run the arbitrage checker
check_arbitrage_opportunity()
