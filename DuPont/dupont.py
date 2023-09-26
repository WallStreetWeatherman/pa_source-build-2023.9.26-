class Company:
    def __init__(self, name, net_income, sales, average_assets, average_equity):
        self.name = name
        
        # Multiply values by 1,000,000 for internal computations
        self.net_income = net_income * 1_000_000
        self.sales = sales * 1_000_000
        self.average_assets = average_assets * 1_000_000
        self.average_equity = average_equity * 1_000_000

        # Validate the data upon initialization
        self._validate_data()

    def _validate_data(self):
        """Validate data to ensure no negative or zero values, and that sales > net_income."""
        if any(value <= 0 for value in [self.net_income, self.sales, self.average_assets, self.average_equity]):
            raise ValueError("Net income, sales, average assets, and average equity must be positive.")
        
        if self.net_income > self.sales:
            raise ValueError("Net income should not exceed sales.")

    def net_profit_margin(self):
        """Net Profit Margin = Net Income / Sales"""
        return self.net_income / self.sales

    def asset_turnover_ratio(self):
        """Asset Turnover Ratio = Sales / Average Assets"""
        return self.sales / self.average_assets

    def equity_multiplier(self):
        """Equity Multiplier = Average Assets / Average Equity"""
        return self.average_assets / self.average_equity

    def return_on_equity(self):
        """ROE = Net Profit Margin * Asset Turnover Ratio * Equity Multiplier"""
        return self.net_profit_margin() * self.asset_turnover_ratio() * self.equity_multiplier()

def display_dupont_analysis(company):
    try:
        print(f"DuPont Analysis for {company.name}:")
        print(f"Net Profit Margin: {company.net_profit_margin():.2f}")
        print(f"Asset Turnover Ratio: {company.asset_turnover_ratio():.2f}")
        print(f"Equity Multiplier: {company.equity_multiplier():.2f}")
        print(f"Return on Equity (ROE): {company.return_on_equity():.2f}\n")
    except Exception as e:
        print(f"An error occurred while processing the data for {company.name}. Error: {str(e)}\n")

# Try and Catch block to handle potential initialization errors
try:
    # Representing values in decimals representing millions
    company_A = Company("Company A", net_income=0.1, sales=0.5, average_assets=2.0, average_equity=1.0)
    display_dupont_analysis(company_A)
except Exception as e:
    print(f"An error occurred while initializing data for Company A. Error: {str(e)}\n")

try:
    # Representing values in decimals representing millions
    company_B = Company("Company B", net_income=0.12, sales=0.55, average_assets=2.1, average_equity=1.05)
    display_dupont_analysis(company_B)
except Exception as e:
    print(f"An error occurred while initializing data for Company B. Error: {str(e)}\n")
