import numpy as np
import matplotlib.pyplot as plt


def generate_trajectory(length=100, start=0, end=1, lower_bound=0, upper_bound=1, polynomial=3, noise_scale=1, seed=100):
    # Generate a trajectory.
    trajectory = np.linspace(start, end, length, dtype=np.float64)
    trajectory **= polynomial

    # Create a random number generator with the given seed (for repeatability).
    rng = np.random.default_rng(seed)
    noise = (rng.random(length) * 2 - 1) * noise_scale

    # Normalization and scaling
    trajectory = trajectory - np.min(trajectory)
    trajectory /= np.max(trajectory)
    trajectory *= (upper_bound - lower_bound)
    trajectory += lower_bound

    # Adding the generated noise
    trajectory += noise
    return trajectory


def exponential_moving_average(signal, alpha=0.5):
    ema = [signal[0]]
    beta = 1 - alpha
    for i in range(1, len(signal)):
        ema.append(alpha * signal[i] + beta * ema[i - 1])

    return ema


def predict_trajectory(frames, frames_to_observe, frames_to_detect, starting_frame=0):
    # The exponential moving average damping coefficients:
    position_alpha = 0.4
    velocity_alpha = 0.4

    # Average the whole trajectory (The whole? Mainly for graphing later.)
    trajectory_averaged = exponential_moving_average(frames, position_alpha)
    # Get the relevant positions on which the do the prediction.
    positions = trajectory_averaged[starting_frame:
                                    starting_frame + frames_to_observe]

    # Find the velocities.
    velocities = [0]  # The first is 0, because we have no more information about the past. 
    for i in range(1, frames_to_observe):
        velocities.append(positions[i] - positions[i - 1])

    # Also average the velocities.
    velocities = exponential_moving_average(velocities, velocity_alpha)

    # Only use the last value and do a linear prediction.
    predicted_velocity = velocities[-1]
    starting_position = positions[-1]
    predicted_positions = []

    # Find, linearly, the position of the object in each of the future frames.
    for i in range(1, frames_to_detect + 1):
        predicted_positions.append(starting_position + predicted_velocity * i)

    # Pad the output with None values, so we can graph only the prediction.
    output = [None] * (starting_frame + frames_to_observe)
    output.extend(predicted_positions)
    output.extend([None] * (len(frames) - len(output)))

    # Return the predicted positions (padded) and the whole averaged trajectory.
    return output, trajectory_averaged


def main():
    # All the variables try to reasonably close to possible real values of this problem.
    # 78 points? Prediction runs 30 fps. The average human tends to walk at ~1.4 m/s.
    # Distance is sqrt((1.23 - (-1.23))^2 + (3 - 0.3)^2) ~= 3.65 m
    # Therefore, the time it takes to walk 3.65 is 3.65 m / 1.4 m/s ~= 2.61 s
    # Therefore, we will get 2.61 * 30 ~= 78.27 frames ~= 78 frames
    #
    # The noise scale is 0.05 in both axes. That means +-5 cm, that is an abs error of 10 cm.
    # This is realistic as shown by the measurements results.
    trajectory_length = 78
    trajectory_y = generate_trajectory(
        trajectory_length, -5, 5, 0.3, 3, 3, noise_scale=0.05, seed=11)
    trajectory_x = generate_trajectory(
        trajectory_length, -5, 5, -1.23, 1.23, 1, noise_scale=0.05, seed=10)

    # Based on how many observed frames predict how many future frames?
    # The prediction is linear!
    observe = 5
    predict = 10

    for i in range(trajectory_length - observe):
        predictions_y, trajectory_averaged_y = predict_trajectory(
            trajectory_y, observe, predict, i)
        predictions_x, trajectory_averaged_x = predict_trajectory(
            trajectory_x, observe, predict, i)

        plt.plot(trajectory_x, trajectory_y)
        plt.plot(trajectory_averaged_x, trajectory_averaged_y)
        plt.plot(predictions_x, predictions_y)

        plt.pause(0.1)
        # plt.show()
        plt.cla()


if __name__ == "__main__":
    main()
