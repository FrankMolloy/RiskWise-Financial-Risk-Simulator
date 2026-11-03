import numpy as np
from numbers import Integral

class MonteCarloSimulator:
    def __init__(
        self,
        years: int,
        start_balance: float,
        monthly_contribution: float,
        expected_return_annual: float,
        volatility_annual: float,
        inflation_annual: float,
        simulations: int = 1000,
        seed: int = 42
    ):
        if isinstance(years, bool) or not isinstance(years, Integral) or years < 0:
            raise ValueError("years must be a non-negative integer")
        if isinstance(simulations, bool) or not isinstance(simulations, Integral) or simulations < 1:
            raise ValueError("simulations must be a positive integer")
        values = (start_balance, monthly_contribution, expected_return_annual,
                  volatility_annual, inflation_annual)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("simulation parameters must be finite")
        if start_balance < 0 or monthly_contribution < 0 or volatility_annual < 0:
            raise ValueError("balances, contributions and volatility must be non-negative")
        if inflation_annual <= -1:
            raise ValueError("inflation_annual must be greater than -1")
        self.years = years
        self.start_balance = start_balance
        self.monthly_contribution = monthly_contribution
        self.expected_return_annual = expected_return_annual
        self.volatility_annual = volatility_annual
        self.inflation_annual = inflation_annual
        self.simulations = simulations
        self.seed = seed

    def run(self) -> np.ndarray:
        """Return real balances shaped (simulations, 12 * years + 1).

        Contributions are fixed nominal amounts paid after each month's return.
        A local legacy RNG preserves seeded results without changing global state.
        """
        rng = np.random.RandomState(self.seed)

        months = self.years * 12

        # Convert annual return + volatility to monthly
        mean_monthly_return = self.expected_return_annual / 12
        monthly_volatility = self.volatility_annual / np.sqrt(12)

        # Generate random monthly returns
        random_returns = rng.normal(
            mean_monthly_return,
            monthly_volatility,
            (self.simulations, months)
        )

        # Initialise balance array
        balances = np.zeros((self.simulations, months + 1))
        balances[:, 0] = self.start_balance

        # Simulate
        for t in range(1, months + 1):
            balances[:, t] = (
                balances[:, t - 1] * (1 + random_returns[:, t - 1])
                + self.monthly_contribution
            )

        # Adjust for inflation
        monthly_inflation = (1 + self.inflation_annual) ** (1/12) - 1
        inflation_factor = (1 + monthly_inflation) ** np.arange(months + 1)

        balances_real = balances / inflation_factor

        return balances_real

def simulate(
    years: int,
    start_balance: float,
    monthly_contribution: float,
    expected_return_annual: float,
    volatility_annual: float,
    inflation_annual: float,
    simulations: int = 3000,
    seed: int = 42
) -> np.ndarray:
    sim = MonteCarloSimulator(
        years=years,
        start_balance=start_balance,
        monthly_contribution=monthly_contribution,
        expected_return_annual=expected_return_annual,
        volatility_annual=volatility_annual,
        inflation_annual=inflation_annual,
        simulations=simulations,
        seed=seed
)
    return sim.run()
