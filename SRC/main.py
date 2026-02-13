import matplotlib.pyplot as plt
import numpy as np

from simulator import simulate
from metrics import summarise, prob_reach_goal
from market_data import fetch_stooq_close_prices, estimate_annual_return_vol

def main():

    scenarios = {
        "cautious":  {"r": 0.05, "vol": 0.10},
        "balanced":  {"r": 0.07, "vol": 0.15},
        "aggressive":{"r": 0.09, "vol": 0.20},
    }

    while True:
        print("\n--- Financial Risk Simulator ---")

        scenario = input("Choose scenario (cautious, balanced, aggressive): ").lower()
        if scenario not in scenarios:
            print("Invalid scenario. Try again.")
            continue

        years = int(input("Investment length (years): "))
        monthly = float(input("Monthly contribution: "))
        goal = float(input("Goal amount (optional, enter 0 to skip): "))

        params = scenarios[scenario]

        paths = simulate(
            years=years,
            start_balance=1000,
            monthly_contribution=monthly,
            expected_return_annual=params["r"],
            volatility_annual=params["vol"],
            inflation_annual=0.02,
            simulations=3000,
            seed=42
        )

        final_values = paths[:, -1]
        summary = summarise(final_values)

        print("\nResults:")
        print("Median outcome:", round(summary["median"], 2))
        print("10th percentile:", round(summary["p10"], 2))
        print("90th percentile:", round(summary["p90"], 2))

        spread = summary["p90"] - summary["p10"]
        print(f"Uncertainty range (90th - 10th percentile): £{spread:,.0f}")

        if goal > 0:
            p = prob_reach_goal(final_values, goal)
            print(f"Chance of reaching £{goal:,.0f}: {p * 100:.2f}%")

        # Plot sample paths
        for i in range(50):
            plt.plot(paths[i])
        plt.title(f"Wealth Paths ({scenario.title()} scenario)")
        plt.xlabel("Months")
        plt.ylabel("Balance (Real Terms)")
        plt.show()

        # Histogram
        plt.figure()
        plt.hist(final_values, bins=60)
        plt.axvline(summary["median"], linestyle="--", label="Median")
        if goal > 0:
            plt.axvline(goal, linestyle="--", label="Goal")
        plt.title("Distribution of Final Outcomes")
        plt.xlabel("Final Balance (Real Terms)")
        plt.ylabel("Count")
        plt.legend()
        plt.show()

        again = input("\nRun another simulation? (y/n): ").lower()
        if again != "y":
            break


if __name__ == "__main__":
    main()
