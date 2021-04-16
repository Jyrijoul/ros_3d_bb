class Predictor:
    def __init__(self, tracker, sensitivity=0.5):
        self.tracker = tracker
        self.sensitivity = sensitivity
        self.averages = {}
        self.predictions = {}

    def update(self):
        previous_averages = set(self.averages)
        current_averages = set()
        for obj in self.tracker.objects:
            if not obj.uid in self.averages:
                self.averages[obj.uid] = (obj.x, obj.y, obj.v_x, obj.v_y)
                current_averages.add(obj.uid)
            else:
                old_values = self.averages[obj.uid]
                new_values = (obj.x, obj.y, obj.v_x, obj.v_y)
                averages = []
                for i in range(len(old_values)):
                    # Calculate the exponential moving average.
                    averages.append(
                        new_values[i] * self.sensitivity + old_values[i] * (1 - self.sensitivity))

                self.averages[obj.uid] = tuple(averages)
                current_averages.add(obj.uid)

        expired_averages = previous_averages - current_averages
        for uid in expired_averages:
            del self.averages[uid]

    def predict(self, frames_to_predict):
        for obj in self.tracker.objects:
            averages = self.averages[obj.uid]

            current_x = obj.x
            current_y = obj.y
            average_v_x = averages[2]
            average_v_y = averages[3]
            
            # New position in n frames = old position + average velocity * n
            new_x = current_x + average_v_x * frames_to_predict
            new_y = current_y + average_v_y * frames_to_predict

            self.predictions[obj.uid] = (new_x, new_y, average_v_x, average_v_y)
