import cv2
import csv


class Error:
    def __init__(self, measurement, ground_truth):
        self.measurement = measurement
        self.ground_truth = ground_truth
        self.x_measured = measurement[0]
        self.z_measured = measurement[1]
        self.x_ground_truth = ground_truth[0]
        self.z_ground_truth = ground_truth[1]

        self.x_error = self.x_measured - self.x_ground_truth
        self.z_error = self.z_measured - self.z_ground_truth
        self.x_error_abs = abs(self.x_measured - self.x_ground_truth)
        self.z_error_abs = abs(self.z_measured - self.z_ground_truth)

    def to_list(self):
        return [self.x_measured, self.x_ground_truth, 
                self.z_measured, self.z_ground_truth, 
                self.x_error, self.x_error_abs, 
                self.z_error, self.z_error_abs]


def read_measurements(measurements_file):
    separator = "] "  # a pretty bad separator...
    with open(measurements_file, "r") as f:
        lines = []
        for line in f:
            line = line.strip().split(separator)
            line[0] = line[0] + "]"
            lines.append(line)
    return lines


def imread_measurement_time(
    measurement_time,
    path="/home/jyri/catkin_ws/src/ros_3d_bb/measurements/",
    prefix="measurements",
    imtype="color",
    filetype="png",
):
    return cv2.imread(
        path + prefix + "_" + measurement_time + "_" + imtype + "." + filetype, 1
    )


def get_user_choice(measurements_list, measurement_time):
    measurement_choices = "'0'"
    print("\nHere is the list of measurements taken at " + measurement_time + ":")
    print(measurements_list)

    for i in range(1, len(measurements_list)):
        measurement_choices += ", '" + str(i) + "'"

    selection = []
    while True:
        if len(measurements_list) == 1:
            selection = measurements_list
            break

        choice = input(
            "Which measurement best represents the ground truth?\nInput "
            + measurement_choices
            + " or 'all' in order to choose either one or all of them\n"
        )

        if choice == "all":
            selection = measurements_list
            break
        elif choice.isdecimal():
            num = int(choice)
            if 0 <= num < len(measurements_list):
                selection = [measurements_list[num]]
                break
        elif choice == "":
            return -1, -1

        print("Please enter a valid choice!")

    ground_truth = input(
        "Please enter the ground truth X and Z, for example: '0, 210', with regards to\n"
        + str(selection)
        + " <-- your choice of measurements:\n"
    )

    if ground_truth == "":
        return -1, -1
    else:
        ground_truth = eval(ground_truth)

    return selection, ground_truth


def remove_y_coordinate(measurements):
    new_measurements = []

    for measurement in measurements:
        new_measurements.append((measurement[0], measurement[2]))

    return new_measurements


def calculate_xz_error(measurements_map):
    errors = []
    for measurement, ground_truth in measurements_map.items():
        errors.append(Error(measurement, ground_truth))

    return errors


def errors_to_csv(filename, errors):
    # [self.x_measured, self.x_ground_truth, 
    # self.z_measured, self.z_ground_truth, 
    # self.x_error, self.x_error_abs, 
    # self.z_error, self.z_error_abs]
    header = ["X measured", "X true", "Z measured", "Z true", 
            "X error", "X absolute error", "Z error", "Z absolute error"]
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=';',quoting=csv.QUOTE_MINIMAL)

        writer.writerow(header)
        for error in errors:
            writer.writerow(error.to_list())

    print("Written successfully to " + filename + "!")
            

def main():
    print("Let's map the measurements to the ground truth!")

    # A map to represent the measurements
    measurement_to_ground_truth = {}

    lines = read_measurements(
        "/home/jyri/catkin_ws/src/ros_3d_bb/measurements/measurements.txt"
    )
    for line in lines:
        measurement_time = line[-1]
        cv2.imshow("Image", imread_measurement_time(measurement_time))
        cv2.waitKey(1)

        measurements_list = eval(line[0])
        measurements_list = remove_y_coordinate(measurements_list)
        selection, ground_truth = get_user_choice(measurements_list, measurement_time)
        if selection == -1:  # When the user wants to end the process, break.
            break
        print("User selected:", selection, ground_truth)
        for measurement in selection:
            measurement_to_ground_truth[measurement] = ground_truth

    print(measurement_to_ground_truth)
    errors = calculate_xz_error(measurement_to_ground_truth)
    
    for error in errors:
        print(error.to_list())

    errors_to_csv("data_with_outliers.csv", errors)

    print("Done! Exiting..,")


if __name__ == "__main__":
    main()
