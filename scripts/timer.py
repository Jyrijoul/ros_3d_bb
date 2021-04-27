import time


class Timer:
    def __init__(self, name, in_ms=True):
        self.name = name
        self.in_ms = in_ms
        self.times = [time.perf_counter()]
        self.total = 0
        self.timings = []

    def __str__(self):
        return self.name + str(self.times)

    def update(self):
        self.times.append(time.perf_counter())

    def stop(self, print_output=True):
        self.times.append(time.perf_counter())

        self.timings = []
        for i in range(1, len(self.times)):
            if not self.in_ms:
                self.timings.append(self.times[i] - self.times[i - 1])
            else:
                self.timings.append((self.times[i] - self.times[i - 1]) * 1000)

        if print_output:
            print("Times (" + self.name + "):")
            print(self.timings)

        self.total = sum(self.timings)
        return self.timings

    def average_times(timers, append_total=True):
        """Find the average timings, assuming every timer has an equal amount of timings."""
        nr_of_timers = len(timers)
        nr_of_times = len(timers[0].timings)
        averages = [0] * nr_of_times
        if append_total:
            averages.append(0)
        
        for timer in timers:
            for i in range(nr_of_times):
                averages[i] += timer.timings[i] / nr_of_timers

            if append_total:
                averages[i + 1] += timer.total / nr_of_timers

        return averages
