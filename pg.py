
def normalize_weights(weights):
    """ensure the maximum absolute weight is 1, adapting all other weights respectively"""
    return [w / max(map(abs, weights)) for w in weights]


print(normalize_weights([6, 2, 2]))
print(normalize_weights([-6, 2, 2]))
print(normalize_weights([6, -2, 2]))
