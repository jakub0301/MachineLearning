"""Linear regression implementation"""

import numpy as np


# y = ax + b
def compute_error(b, a, points):
    totalError = 0
    for (x,y) in points:
        totalError += (y - (a * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, a_current, points, learingRate):
    b_gradient = 0
    a_gradient = 0
    N = float(len(points))
    for (x,y) in points:
        b_gradient += -(2 / N) * (y - ((a_current * x) + b_current))
        a_gradient += -(2 / N) * x * (y - ((a_current * x) + b_current))
    return (b_current - (learingRate * b_gradient),
            a_current - (learingRate * a_gradient))


def gradient_descent(
        points,
        starting_b,
        starting_a,
        learning_rate,
        num_iterations):
    b = starting_b
    a = starting_a
    for i in range(num_iterations):
        b, a = step_gradient(b, a, np.array(points), learning_rate)
    return (b, a)


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0
    initial_a = 0
    num_iterations = 1000
    print(f"Gradient descent at b = {initial_b}, a = {initial_a}, error = {compute_error(initial_b, initial_a, points)}")
    print("Running...")
    [b, a] = gradient_descent(
        points, initial_b, initial_a, learning_rate, num_iterations)
    print(f"After {num_iterations} iterations b = {b}, a = {a}, error = {compute_error(b, a, points)}")


if __name__ == "__main__":
    run()
