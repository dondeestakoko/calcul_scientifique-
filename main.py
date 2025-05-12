import numpy as np

def f(x):
    return x**2 - 8 * np.log(x)

right = 2
left = 1
precision = 10**(-3)

middle = (right + left) / 2  # Initialize `middle`

while right - left >= precision:
    middle = (right + left) / 2
    if f(middle) == 0:
        break
    elif f(left) * f(middle) < 0:
        right = middle
    else:  # Simplify the condition here
        left = middle

print(middle)
print(f(middle))
