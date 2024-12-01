from datetime import datetime

def get_lilith_aspects_with_segments():
    # Define influences and their associated elements and states for each hour
    aspects = {
        0: ("Renewal, Rest", "Fear of the Unknown", "Water", ["Wet"], "Fixed"),
        1: ("Introspection, Dreaming", "Nightmares, Anxiety", "Water", ["Wet"], "Fixed"),
        2: ("Creativity, Inspiration", "Overwhelm", "Air", ["Wet", "Hot"], "Mutable"),
        3: ("Intuition, Seer", "Paranoia", "Air", ["Wet", "Hot"], "Mutable"),
        4: ("Embracing Shadows", "Resentment", "Earth", ["Cold", "Dry"], "Fixed"),
        5: ("Healing, Rebirth", "Stagnation", "Earth", ["Cold", "Dry"], "Fixed"),
        6: ("Morning Clarity", "Confusion", "Fire", ["Hot"], "Mutable"),
        7: ("Empowerment, Motivation", "Aggression", "Fire", ["Hot"], "Mutable"),
        8: ("Nurturing, Compassion", "Overprotection", "Earth", ["Cold", "Dry"], "Fixed"),
        9: ("Social Bonding", "Isolation", "Water", ["Wet"], "Fixed"),
        10: ("Manifestation", "Disillusionment", "Air", ["Wet", "Hot"], "Mutable"),
        11: ("Purpose, Drive", "Aimlessness", "Fire", ["Hot"], "Mutable"),
        12: ("Fulfillment", "Burnout", "Earth", ["Cold", "Dry"], "Fixed"),
        13: ("Growth, Learning", "Resistance to Change", "Air", ["Wet", "Hot"], "Mutable"),
        14: ("Transformation", "Conflicts", "Fire", ["Hot"], "Mutable"),
        15: ("Clarity, Insight", "Doubt", "Air", ["Wet", "Hot"], "Mutable"),
        16: ("Connection", "Disconnection", "Water", ["Wet"], "Fixed"),
        17: ("Leadership, Power", "Tyranny", "Fire", ["Hot"], "Mutable"),
        18: ("Celebration, Joy", "Guilt", "Earth", ["Cold", "Dry"], "Fixed"),
        19: ("Reflection, Gratitude", "Regret", "Water", ["Wet"], "Fixed"),
        20: ("Relaxation, Peace", "Anxiety Night", "Water", ["Wet"], "Fixed"),
        21: ("Closure, Letting Go", "Holding On", "Earth", ["Cold", "Dry"], "Fixed"),
        22: ("Mystery, Exploration", "Fear of the Dark", "Air", ["Wet", "Hot"], "Mutable"),
        23: ("Resting, Surrender", "Resistance to Change", "Water", ["Wet"], "Fixed"),
    }

    # Daily attributes for each day of the week including various features
    daily_attributes = {
        0: {
            "name": "Sunday",
            "ruler": "Sun",
            "numerology": (1, "Unity, Leadership, Initiation"),
            "kabbalistic_sphere": "Tiferet (Beauty)",
            "esoteric_meaning": "Radiance, self-expression, vitality, and inner light.",
            "color": "Gold or yellow",
            "element": "Fire",
            "angle_range": "0°–51.43°",
        },
        1: {
            "name": "Monday",
            "ruler": "Moon",
            "numerology": (2, "Duality, Reflection, Balance"),
            "kabbalistic_sphere": "Yesod (Foundation)",
            "esoteric_meaning": "Emotional depth, cycles of life, intuition, and adaptability.",
            "color": "Silver or pale blue",
            "element": "Water",
            "angle_range": "51.43°–102.86°",
        },
        2: {
            "name": "Tuesday",
            "ruler": "Mars",
            "numerology": (3, "Creativity, Action, Expansion"),
            "kabbalistic_sphere": "Gevurah (Strength)",
            "esoteric_meaning": "Passion, determination, assertiveness, and overcoming conflict.",
            "color": "Red",
            "element": "Fire",
            "angle_range": "102.86°–154.29°",
        },
        3: {
            "name": "Wednesday",
            "ruler": "Mercury",
            "numerology": (4, "Order, Stability, Communication"),
            "kabbalistic_sphere": "Hod (Splendor)",
            "esoteric_meaning": "Mental agility, learning, adaptability, and creativity.",
            "color": "Orange or yellow",
            "element": "Air",
            "angle_range": "154.29°–205.72°",
        },
        4: {
            "name": "Thursday",
            "ruler": "Jupiter",
            "numerology": (5, "Change, Growth, Abundance"),
            "kabbalistic_sphere": "Chesed (Mercy)",
            "esoteric_meaning": "Optimism, spiritual growth, abundance, and faith.",
            "color": "Royal blue or purple",
            "element": "Air",
            "angle_range": "205.72°–257.15°",
        },
        5: {
            "name": "Friday",
            "ruler": "Venus",
            "numerology": (6, "Harmony, Love, Connection"),
            "kabbalistic_sphere": "Netzach (Victory)",
            "esoteric_meaning": "Beauty, relationships, emotional harmony, and inspiration.",
            "color": "Green or pink",
            "element": "Water",
            "angle_range": "257.15°–308.58°",
        },
        6: {
            "name": "Saturday",
            "ruler": "Saturn",
            "numerology": (7, "Mysticism, Discipline, Completion"),
            "kabbalistic_sphere": "Binah (Understanding)",
            "esoteric_meaning": "Patience, responsibility, karma, and boundaries.",
            "color": "Black or dark gray",
            "element": "Earth",
            "angle_range": "308.58°–360°",
        },
    }

    # Get the current local time
    now = datetime.now()
    current_hour = now.hour  # Current hour in 24-hour format
    current_minute = now.minute  # Current minute
    current_day = (now.weekday() + 1) % 7  # Adjust to map Sunday=0 through Saturday=6

    # Get attributes for the current day
    current_day_info = daily_attributes[current_day]

    # Get the aspects for the current, previous, and next hour
    this_hour_aspect = aspects.get(current_hour, ("Unknown", "Unknown", "Unknown", [], "Unknown"))
    previous_hour_aspect = aspects.get((current_hour - 1) % 24, ("Unknown", "Unknown", "Unknown", [], "Unknown"))
    next_hour_aspect = aspects.get((current_hour + 1) % 24, ("Unknown", "Unknown", "Unknown", [], "Unknown"))

    # Calculate the segment number based on the current minute
    segment_size = 7.5  # Each segment is 7.5 minutes
    segment_number = int(current_minute // segment_size)

    # Determine the influence based on the segment
    if segment_number < 4:
        positive_aspect = previous_hour_aspect[0] if segment_number < 2 else this_hour_aspect[0]
        negative_aspect = previous_hour_aspect[1] if segment_number < 2 else this_hour_aspect[1]
    else:
        positive_aspect = this_hour_aspect[0] if segment_number < 6 else next_hour_aspect[0]
        negative_aspect = this_hour_aspect[1] if segment_number < 6 else next_hour_aspect[1]

    # Calculate distances and percentages for the current hour
    total_minutes = 60
    distance_to_start_current_hour = current_minute
    distance_to_end_current_hour = total_minutes - current_minute

    # Calculate percentages to start and end of the current hour
    percent_to_start_current_hour = (distance_to_start_current_hour / total_minutes) * 100
    percent_to_end_current_hour = (distance_to_end_current_hour / total_minutes) * 100

    # Calculate percentages to start and end of the current day (24 hours)
    total_hours_in_day = 24
    distance_to_start_current_day = current_hour * 60 + current_minute
    distance_to_end_current_day = (total_hours_in_day * 60) - distance_to_start_current_day

    percent_to_start_current_day = (distance_to_start_current_day / (total_hours_in_day * 60)) * 100
    percent_to_end_current_day = (distance_to_end_current_day / (total_hours_in_day * 60)) * 100

    # Calculate Symmetry and Distance for Inner and Outer Cycle
    S_entire = 0
    E_entire = 24

    # Current position is in hours
    P_current = current_hour + (current_minute / 60)

    # Calculate Inner Cycle Properties
    Range_inner = E_entire - S_entire
    Inner_Percentage = (P_current - S_entire) / Range_inner * 100

    Dist_to_S_inner = P_current - S_entire
    Dist_to_E_inner = E_entire - P_current

    # Calculate symmetrical percentages
    Dist_to_S_inner_perc = (Dist_to_S_inner / Range_inner) * 100
    Dist_to_E_inner_perc = (Dist_to_E_inner / Range_inner) * 100

    # Energy Symmetry Calculation: Create a scale of energy based on the current hour's aspect
    energy_levels = {
        "Fire": 80,
        "Air": 70,
        "Earth": 50,
        "Water": 60,
    }

    current_energy = energy_levels[this_hour_aspect[2]]

    # Suggest activities based on the current hour's aspect
    activity_suggestions = {
        "Fire": "Engage in physical exercise or creative projects.",
        "Air": "Focus on socializing or brainstorming new ideas.",
        "Earth": "Spend time in nature or focus on routine tasks.",
        "Water": "Practice relaxation techniques or engage in artistic activities."
    }

    suggestion = activity_suggestions.get(this_hour_aspect[2], "No specific suggestion available.")

    # Get element details for the current hour's aspect
    element_details = {
        "Earth": {
            "Qualities": "Cold and Dry",
            "Explanation": "Earth is stable, heavy, and solid, embodying qualities of fixedness and materiality.",
            "Planet": "Saturn",
            "Frequency Range (Hz)": (50, 70),
            "Energy": "Grounding and Stability",
            "Vibrational Resonance": 128,
            "Numerological Resonance": 4,
            "Magnetic Resonance": "Low",
            "Angular Momentum": "Stability in Motion"
        },
        "Water": {
            "Qualities": "Cold and Wet",
            "Explanation": "Water is fluid, nurturing, and cohesive, associated with emotional depth and adaptability.",
            "Planet": "Moon",
            "Frequency Range (Hz)": (70, 100),
            "Energy": "Emotional Balance",
            "Vibrational Resonance": 256,
            "Numerological Resonance": 2,
            "Magnetic Resonance": "Moderate",
            "Angular Momentum": "Flow and Adaptation"
        },
        "Fire": {
            "Qualities": "Hot and Dry",
            "Explanation": "Fire is transformative, energetic, and dynamic, symbolizing activity and passion.",
            "Planet": "Mars",
            "Frequency Range (Hz)": (100, 150),
            "Energy": "Action and Creativity",
            "Vibrational Resonance": 432,
            "Numerological Resonance": 1,
            "Magnetic Resonance": "High",
            "Angular Momentum": "Dynamic Transformative Energy"
        },
        "Air": {
            "Qualities": "Hot and Wet",
            "Explanation": "Air is light, expansive, and communicative, embodying intellect and movement.",
            "Planet": "Jupiter",
            "Frequency Range (Hz)": (150, 200),
            "Energy": "Intellect and Social Connection",
            "Vibrational Resonance": 512,
            "Numerological Resonance": 3,
            "Magnetic Resonance": "Variable",
            "Angular Momentum": "Fluid Movement"
        },
    }

    # Return all relevant details, including element details for current hour's aspect
    return {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "current_hour": current_hour,
        "current_minute": current_minute,
        "segment_number": segment_number,
        "positive_aspect": positive_aspect,
        "negative_aspect": negative_aspect,
        "current_hour_aspect": this_hour_aspect,
        "percent_to_start_current_hour": percent_to_start_current_hour,
        "percent_to_end_current_hour": percent_to_end_current_hour,
        "this_hour_aspect": this_hour_aspect,
        "previous_hour_aspect": previous_hour_aspect,
        "next_hour_aspect": next_hour_aspect,
        "percent_to_start_current_day": percent_to_start_current_day,
        "percent_to_end_current_day": percent_to_end_current_day,
        "Inner_Percentage": Inner_Percentage,
        "Dist_to_S_inner": Dist_to_S_inner,
        "Dist_to_E_inner": Dist_to_E_inner,
        "Dist_to_S_inner_perc": Dist_to_S_inner_perc,
        "Dist_to_E_inner_perc": Dist_to_E_inner_perc,
        "element": this_hour_aspect[2],
        "states": this_hour_aspect[3],
        "current_energy": current_energy,
        "suggestion": suggestion,
        "daily_info": current_day_info,  # Day-specific information
        "element_details": element_details[this_hour_aspect[2]],  # Element details for current hour's aspect
    }

# Example usage
if __name__ == "__main__":
    lilith_info = get_lilith_aspects_with_segments()

    # Print Current Hour Information
    print(f"--- Current Hour Information ---")
    print(f"Current Time: {lilith_info['current_time']}")
    print(f"Current Hour: {lilith_info['current_hour']}")
    print(f"Current Minute: {lilith_info['current_minute']}")
    print(f"Current Segment: {lilith_info['segment_number'] + 1}/8")
    print(f"Positive Aspect: {lilith_info['positive_aspect']}")
    print(f"Negative Aspect: {lilith_info['negative_aspect']}")
    print(f"Current Hour Aspect: {lilith_info['current_hour_aspect']}")
    print(f"Element: {lilith_info['element']}")
    print(f"States: {', '.join(lilith_info['states'])}")
    print(f"Current Energy Level: {lilith_info['current_energy']}")
    print(f"Activity Suggestion: {lilith_info['suggestion']}")
    print(f"Percentage to Start of Current Hour: {lilith_info['percent_to_start_current_hour']:.12f}%")
    print(f"Percentage to End of Current Hour: {lilith_info['percent_to_end_current_hour']:.12f}%")

    # Print Daily Information
    daily_info = lilith_info['daily_info']
    print(f"\n--- Daily Information for {daily_info['name']} ---")
    print(f"Day: {daily_info['name']}")
    print(f"Planetary Ruler: {daily_info['ruler']}")
    print(f"Numerology: {daily_info['numerology'][0]} - {daily_info['numerology'][1]}")
    print(f"Kabbalistic Sphere: {daily_info['kabbalistic_sphere']}")
    print(f"Esoteric Meaning: {daily_info['esoteric_meaning']}")
    print(f"Color: {daily_info['color']}")
    print(f"Element: {daily_info['element']}")
    print(f"Angle Range: {daily_info['angle_range']}")

    # Print Aspects for Previous and Next Hours
    print(f"\nPrevious Hour Aspect: {lilith_info['previous_hour_aspect']}")
    print(f"Next Hour Aspect: {lilith_info['next_hour_aspect']}")

    # Print Start and End Percentages for Current Day
    print(f"\nPercentage from Start of Current Day: {lilith_info['percent_to_start_current_day']:.12f}%")
    print(f"Percentage to End of Current Day: {lilith_info['percent_to_end_current_day']:.12f}%")

    # Print Distances and Symmetries for Inner Cycle
    print(f"\nDistance to Start of Inner Cycle: {lilith_info['Dist_to_S_inner']:.12f} hours")
    print(f"Distance to End of Inner Cycle: {lilith_info['Dist_to_E_inner']:.12f} hours")

    # Print Element Details
    element_details = lilith_info['element_details']
    print(f"\n--- Current Hour Element Details ---")
    print(f"Element Qualities for {element_details['Qualities']}: {element_details['Explanation']}.")
    print(f"Planet: {element_details['Planet']}")
    print(f"Frequency Range: {element_details['Frequency Range (Hz)']} Hz")
    print(f"Energy: {element_details['Energy']}")
    print(f"Vibrational Resonance: {element_details['Vibrational Resonance']} Hz")
    print(f"Numerological Resonance: {element_details['Numerological Resonance']}")
    print(f"Magnetic Resonance: {element_details['Magnetic Resonance']}")
    print(f"Angular Momentum: {element_details['Angular Momentum']}")