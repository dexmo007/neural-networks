"""http://neuralnetworksanddeeplearning.com/chap1.html#exercise_513527"""
import random
import numpy as np


def generate_input(digit):
    inputs = [random.uniform(0, 0.01) for i in range(10)]
    inputs[digit] = random.uniform(0.99, 1)
    return inputs


weights = [
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
]

for i in range(10):
    binary = [0 if b < 0.99 else 1 for b in np.matmul(
        weights, generate_input(i))]
    binary.reverse()
    print(i, ''.join(str(b) for b in binary))
