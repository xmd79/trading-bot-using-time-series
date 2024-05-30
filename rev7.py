            prev_reversal_index = i
            break

    cycle_start_time = t[prev_reversal_index]
    cycle_end_time = t[current_time_index]
    cycle_duration = cycle_end_time - cycle_start_time

    current_time = t[current_time_index]
    next_reversal_time = cycle_start_time + 1800  # 30 minutes
    time_to_next_reversal = next_reversal_time - current_time

    cycle_direction = "Up" if wave[prev_reversal_index] < wave[current_time_index] else "Down"
    incoming = "Dip" if cycle_direction == "Down" else "Top"

    middle_threshold = (np.max(wave) + np.min(wave)) / 2

    current_close_sine_value = wave[-1]

    distance_to_min = np.abs(current_close_sine_value - np.min(wave))
    distance_to_max = np.abs(current_close_sine_value - np.max(wave))

    total_distance = distance_to_min + distance_to_max
    percentage_to_min = (distance_to_min / total_distance) * 100
    percentage_to_max = (distance_to_max / total_distance) * 100

    min_real_price = np.min(wave)
    max_real_price = np.max(wave)

    print("Current Cycle between Reversals:")
    print(f"Start Time: {cycle_start_time}, End Time: {cycle_end_time}, Duration: {cycle_duration} seconds")
    print(f"Current Time: {current_time}, Time to Next Reversal: {time_to_next_reversal} seconds")
    print(f"Incoming Reversal: {incoming}")
    print(f"Min Real Price: {min_real_price}, Max Real Price: {max_real_price}")
    print(f"Cycle Direction: {cycle_direction}")

    print("Analysis of Current Cycle:")
    print(f"Current Close Real Price: {close_real_price}")
    print(f"Current Close Value on Sine: {current_close_sine_value}")
    print(f"Distance from Current Close Value to Min: {distance_to_min}")
    print(f"Distance from Current Close Value to Max: {distance_to_max}")
    print(f"Percentage to Min: {percentage_to_min}%")
    print(f"Percentage to Max: {percentage_to_max}%")
    print(f"Middle Threshold for Stationary Sine: {middle_threshold}")

    last_reversal_value = wave[prev_reversal_index]
    last_reversal_type = "Top" if last_reversal_value == np.min(wave) else "Dip"
    print(f"Last Reversal Type: {last_reversal_type}")

def scan_assets(pair):
    filter1(pair)

print('Scanning all available assets on main timeframe...')
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(scan_assets, trading_pairs)

print('Filtered dips (30m):', filtered_pairs_dips)
print('Filtered tops (30m):', filtered_pairs_tops)
print('Intermediate pairs (30m):', intermediate_pairs)

if len(filtered_pairs_dips) > 0:
    print('Rescanning dips on lower timeframes...')
    selected_pair_dips = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(filter2, filtered_pairs_dips)

    print('Dips after 5m filter:', selected_pair_dips)

    if len(selected_pair_dips) > 1:
        print('Multiple dips found on 5m timeframe. Analyzing to select the lowest...')
        lowest_dip = None
        lowest_value = float('inf')

        for pair in selected_pair_dips:
            interval = '5m'
            df = get_klines(pair, interval)
            close = df['Close'].values

            frequency = 1 / len(close)
            phase_shift = 0
            angle = 0
            duration = len(close) / 12  # Assumed duration
            sampling_rate = len(close)  # Assumed sampling rate

            t, wave = generate_stationary_wave_with_harmonics(frequency, phase_shift, angle, duration, sampling_rate)

            min_wave_value = np.min(wave)

            if min_wave_value < lowest_value:
                lowest_value = min_wave_value
                lowest_dip = pair

            # Get projected turning points
            projected_high_dates, projected_low_dates, fib_levels_high, fib_levels_low = project_future_turning_points(df, datetime.now().timestamp(), {'date': pd.Timestamp('2025-01-16 04:53:23.520000'), 'price': 150}, {'date': pd.Timestamp('2025-01-11 18:32:27.840000'), 'price': 120}, 1)

            # Print the forecasted dates
            print("Forecasted high date:", projected_high_dates)
            print("Forecasted low date:", projected_low_dates)

            # Print real price value for forecasted dates
            for date, price in zip(projected_high_dates, fib_levels_high):
                print(f"Real price value for forecasted high date {date}: {price}")

            for date, price in zip(projected_low_dates, fib_levels_low):
                print(f"Real price value for forecasted low date {date}: {price}")

        print(f'Lowest dip on 5m timeframe is {lowest_dip} with wave value {lowest_value}')
        print(f'Current asset vs USDT: {lowest_dip}')
        interval = '5m'
        df = get_klines(lowest_dip, interval)
        close_real_price = df['Close'].values[-1]
        analyze_wave(t, wave, frequency, sampling_rate, df)

    else:
        print(f'Selected dip on 5m timeframe: {selected_pair_dips[0]}')
        print(f'Current asset vs USDT: {selected_pair_dips[0]}')
        interval = '5m'
        df = get_klines(selected_pair_dips[0], interval)
        close_real_price = df['Close'].values[-1]
        analyze_wave(t, wave, frequency, sampling_rate, df)

else:
    print('No dips found in main timeframe.')
