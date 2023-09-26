import math
from scipy.stats import norm

def BjerksundStensland(S, K, r, q, T, sigma, option_type="call"):
    """Compute the Bjerksund-Stensland price for an American option."""
    
    # Helper functions
    def alpha(phi, S, K, T, r, q, sigma):
        return -(2*r*T - 2*q*T + sigma**2*T*(phi+1)/(2*(phi+1))) / (2*(r-q+sigma**2/(2*(phi+1))) * T)
    
    def beta(phi, S, K, T, r, q, sigma):
        return (2*(r-q)*T - sigma**2*T*(phi+1)/phi) / (2*(r-q+sigma**2/(2*phi)) * T)
    
    def h(phi, S, K, T, r, q, sigma):
        return -(r-q)*T - sigma**2*T/2*phi
    
    def I(phi, S, K, T, r, q, sigma):
        kappa = 2*r / (sigma**2 * (1-math.exp(-r*T)))
        lambda_ = (-h(phi, S, K, T, r, q, sigma) + alpha(phi, S, K, T, r, q, sigma) + kappa) / (beta(phi, S, K, T, r, q, sigma) - alpha(phi, S, K, T, r, q, sigma) + kappa)
        d1 = -(math.log(S/K) + (beta(phi, S, K, T, r, q, sigma) - alpha(phi, S, K, T, r, q, sigma))*sigma**2*T/2) / (sigma * math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        return K * (1 - norm.cdf(d1)) * (1 - math.exp((r-q)*T)) / kappa - lambda_ * S * (1 - norm.cdf(d2))
    
    # Bjerksund-Stensland model
    if option_type == "call":
        if r == q:
            return max(S-K, 0)
        else:
            b0 = max(K, (r/(r-q))*K)
            b1 = (1-math.exp((r-q)*T)) * b0 + (math.exp((r-q)*T) - 1) * K
            beta_infinity = (1-T*sigma**2/2 - (r-q)*T*(sigma**2*T+2)/(2*(sigma**2*T+4))) * (S/b1)**((r-q)/sigma**2)
            beta_zero = max(0, alpha(1, S, K, T, r, q, sigma))
            beta_star = max(beta_zero, beta_infinity)
            d1 = -(math.log(S/K) + (beta(1, S, K, T, r, q, sigma) - alpha(1, S, K, T, r, q, sigma))*sigma**2*T/2) / (sigma * math.sqrt(T))
            h_star = -(r-q)*T - 0.5*sigma**2*T - (sigma*math.sqrt(T)*(norm.cdf(d1) - 1))/2
            I_star = I(1, S, K, T, r, q, sigma)
            
            if S < (I_star):
                return b0 - S + (b1 - b0) * (math.exp((r-q)*T) - 1 - (r-q)*T - h_star) * (S/b1)**beta_star
            else:
                return max(S-K, 0)
    elif option_type == "put":
        return BjerksundStensland(K, S, r, q, T, sigma, "call") - S + K*math.exp(-r*T)

def main():
    # Hard-coded values
    S = 100.0       # Current stock price
    K = 110.0       # Strike price
    T = 1.0         # Time to expiration in years
    r = 0.05        # Risk-free rate (e.g., 0.05 for 5%)
    q = 0.02        # Dividend yield (e.g., 0.02 for 2%)
    sigma = 0.2     # Volatility
    option_type = "call"  # Option type ('call' or 'put')
    
    option_price = BjerksundStensland(S, K, r, q, T, sigma, option_type)
    
    print(f"The Bjerksund-Stensland {option_type} option price is: ${option_price:.2f}")

if __name__ == "__main__":
    main()
