from ripple import lambdas_to_lambda_tildes
import jax.numpy as jnp
import jax

print("hello?")

m_l, m_u = 0.5, 3.0
lambda_u = 5000

N = 2

m1_list = jnp.linspace(m_l, m_u, N)
m2_list = jnp.linspace(m_l, m_u, N)

l1_list = jnp.linspace(0, lambda_u, N)
l2_list = jnp.linspace(0, lambda_u, N)

lambda_tilde_list, delta_lambda_tilde_list = [], []

# print("jitting function")
# lambdas_to_lambda_tildes = jax.jit(lambdas_to_lambda_tildes)
# print("jitting done")

for m1 in m1_list:
    for m2 in m2_list:
        for l1 in l1_list:
            for l2 in l2_list:
                print(m1, m2, l1, l2)
                lambda_tilde, delta_lambda_tilde = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
                lambda_tilde_list.append(lambda_tilde)
                lambda_tilde_list.append(delta_lambda_tilde)
                
print("jnp.min(lambda_tilde_list)")
print(jnp.min(lambda_tilde_list))

print("jnp.max(lambda_tilde_list)")
print(jnp.max(lambda_tilde_list))

print("jnp.min(delta_lambda_tilde_list)")
print(jnp.min(delta_lambda_tilde_list))

print("jnp.max(delta_lambda_tilde_list)")
print(jnp.max(delta_lambda_tilde_list))