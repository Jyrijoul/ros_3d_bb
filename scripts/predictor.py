class Predictor:
    """A linear predictor based on the exponential moving average

    Call update() to sync with the tracker and predict() to make predictions.
    Use the attribute "predictions" to get a dictionary of predictions 
    (with an object's UID as the key).
    """

    def __init__(self, tracker, sensitivity=0.5):
        """
        Parameters
        ----------
        tracker : Tracker
            The Tracker object used
        sensitivity : float, optional
            The alpha value of the EMA, by default 0.5
        """
        
        self.tracker = tracker
        self.sensitivity = sensitivity
        self.averages = {}
        self.predictions = {}

    def update(self):
        """Syncs with tracker in order to make predictions.
        
        Call this function before using predict() to get up-to-date results!
        """

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
        """Makes linear predictions about the objects' positions.

        Parameters
        ----------
        frames_to_predict : float
            The predictor calculates the objects' positions
            this number of frames into the future.
            Can also be floating point values.
        
        """
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
