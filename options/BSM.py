import math
from scipy.stats import norm

def black_scholes(S, X, T, r, sigma, option_type="call"):
    """Compute the Black-Scholes price for a European option."""
    
    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == "call":
        option_price = S * norm.cdf(d1) - X * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        option_price = X * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    return option_price

def main():
    # Hard-coded values
    S = 100.0       # Current stock price
    X = 110.0       # Strike price
    T = 1.0         # Time to expiration in years
    r = 0.05        # Risk-free rate (eg, 0.05 for 5%)
    sigma = 0.2     # Volatility
    option_type = "call"  # Option type ('call' or 'put')
    
    option_price = black_scholes(S, X, T, r, sigma, option_type)
    
    print(f"The Black-Scholes {option_type} option price is: ${option_price:.2f}")

if __name__ == "__main__":
    main()