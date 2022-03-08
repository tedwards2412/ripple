import numpy as np
import PhenomD


def test_phenomD():
    theta = np.array([20.0, 5.0, 0.9, -0.9])
    coeffs = PhenomD.get_coeffs(theta)
    print(coeffs)
    return None


if __name__ == "__main__":
    test_phenomD()
