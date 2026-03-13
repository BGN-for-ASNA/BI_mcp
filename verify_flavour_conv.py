import sys
import os

# Add the parent directory of BI_mcp to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import tools from BI_mcp
from BI_mcp import tools

stan_model = """
data {
  int N;
}
model {
  y ~ normal(0, 1);
}
"""

bi_python_code = """
def model(x, y):
    alpha = m.dist.normal(0, 10, name='alpha', shape=(1,))
    mu = alpha + x
    m.dist.normal(mu, 1, obs=y)
"""

print("--- Testing Stan to BI R (verify bi.dist) ---")
r_result = tools.convert_stan_to_bi_r(stan_model)
print(r_result["bi_code"])

print("\n--- Testing BI Flavor Translation (Python -> R) ---")
conv_result = tools.convert_bi_flavor(bi_python_code, "python", "r")
print(conv_result["bi_code"])

print("\n--- Testing BI Flavor Translation (R -> Julia) ---")
bi_r_code = conv_result["bi_code"]
conv_result_julia = tools.convert_bi_flavor(bi_r_code, "r", "julia")
print(conv_result_julia["bi_code"])
