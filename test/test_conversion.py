from ripple import lambdas_to_lambda_tildes, lambda_tildes_to_lambdas
import jax.numpy as jnp
import numpy as np

def test_conversion(N: int = 1000, m_l : float = 0.5, m_u : float = 3.0, lambda_l : float = 0, lambda_u : float = 5000):
    
    # Generate random lambda and delta lambda tilde pairs
    og_lambda1 = np.random.uniform(low = lambda_l, high = lambda_u, size = N)
    og_lambda2 = np.random.uniform(low = lambda_l, high = lambda_u, size = N)
    
    # Generate random masses
    m1 = np.random.uniform(low = m_l, high = m_u, size = N)
    m2 = np.random.uniform(low = m_l, high = m_u, size = N)
    
    # Convert lambdas to lambda tildes
    lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([og_lambda1, og_lambda2, m1, m2]))
    # Convert back to lambdas
    lambda_1, lambda_2 = lambda_tildes_to_lambdas(jnp.array([lambda_tilde, delta_lambda_tilde, m1, m2]))
    
    lambda_1, lambda_2 = np.asarray(lambda_1), np.asarray(lambda_2)
    
    # # Check that the conversion is accurate
    # assert np.allclose(lambda_1, og_lambda1)
    # assert np.allclose(lambda_2, og_lambda2)
    
    mse = np.mean((lambda_1 - og_lambda1)**2 + (lambda_2 - og_lambda2)**2)
    
    print(f"Mean squared error: {mse}")
    
    return None

if __name__ == "__main__":
    test_conversion()