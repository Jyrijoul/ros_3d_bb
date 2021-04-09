import numpy as np
import matplotlib.pyplot as plt


def generate_trajectory_(length, trajectory_scale=1, direction_scale=1, noise_scale=1, window_length=10, seed=12):

    starting_point = rng.random() * trajectory_scale
    direction = rng.random() * 2 - 1 * direction_scale
    trajectory_initial = np.arange(1, length + 1) * trajectory_scale
    trajectory = trajectory_initial * direction
    #trajectory **= 2

    averaged_noise = np.convolve(noise, np.ones(
        window_length), "valid") / window_length
    print(trajectory)
    return np.asarray(trajectory[(window_length - 1) // 2:length - (window_length - 1) // 2 - 1]) + averaged_noise


def generate_trajectory(length=100, start=0, end=1, lower_bound=0, upper_bound=1, polynomial=3, noise_scale=1, seed=100):
    trajectory = np.linspace(start, end, length, dtype=np.float64)
    trajectory **= polynomial

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
    position_alpha = 0.4
    velocity_alpha = 0.4
    trajectory_averaged = exponential_moving_average(frames, position_alpha)
    # observed = frames[starting_frame:starting_frame + frames_to_observe]
    # positions = exponential_moving_average(observed, 0.6)
    positions = trajectory_averaged[starting_frame:starting_frame + frames_to_observe]

    if len(positions) < frames_to_observe:
        positions = list(np.pad(positions, (frames_to_observe - len(positions))))

    velocities = [0]

    for i in range(1, frames_to_observe):
        velocities.append(positions[i] - positions[i - 1])

    velocities = exponential_moving_average(velocities, velocity_alpha)
    # print(velocities)
    predicted_velocity = velocities[-1]
    starting_position = positions[-1]
    predicted_positions = []

    for i in range(1, frames_to_detect + 1):
        predicted_positions.append(starting_position + predicted_velocity * i)
    
    output = [None] * (starting_frame + frames_to_observe)
    # positions.extend(predicted_positions)
    output.extend(predicted_positions)
    output.extend([None] * (len(frames) - len(output)))

    # return positions
    return output, trajectory_averaged


def main():
    # trajectory = generate_trajectory(100)
    trajectory_length = 54
    trajectory = generate_trajectory(trajectory_length, -5, 5, 0.3, 3, 4, noise_scale=0.1, seed=11)
    # trajectory_averaged = exponential_moving_average(trajectory, 0.5)
    # print(trajectory)

    observe = 5
    predict = 10

    for i in range(trajectory_length - observe):
        predictions, trajectory_averaged = predict_trajectory(trajectory, observe, predict, i)
        # print(predictions)
        # print(len(predictions))

        plt.plot(trajectory)
        plt.plot(trajectory_averaged)
        plt.plot(predictions)
        
        plt.pause(0.2)
        #plt.show()
        plt.cla()


if __name__ == "__main__":
    main()
