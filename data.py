class DataPoint:
     def __init__(self, inputs, quality):
        self.inputs = inputs      # list[float], length = 11
        self.quality = quality    # int

class DataSet:
    def __init__(self):
        self.samples = []  # start empty
        self.average_point = None
    
    def Compute_Average(self):
        if not self.samples:
            return None
        num_features = len(self.samples[0].inputs)
        avg_inputs = []
        
        for i in range(num_features):
            feature_sum = sum(s.inputs[i] for s in self.samples)
            avg_inputs.append(feature_sum / len(self.samples))
            
        avg_quality = sum(s.quality for s in self.samples) / len(self.samples)

        self.average_point = DataPoint(avg_inputs, avg_quality)    
        
    def Rescale(self, Value, Upper_bound, Lower_bound):
        return (Value - Lower_bound) / (Upper_bound - Lower_bound)

    def load_from_csv(self, file_path, start, end = None):
        import csv
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')  # UCI wine dataset uses ;
            for i, row in enumerate(reader):
                if (i < start):
                    continue
                if (end is not None and i >= end):
                    break
                inputs = [
                    self.Rescale(float(row['fixed acidity']), 15.9, 4.6),
                    self.Rescale(float(row['volatile acidity']), 1.58, 0.12),
                    self.Rescale(float(row['citric acid']), 1.66, 0),
                    self.Rescale(float(row['residual sugar']), 15.5, 0.9),
                    self.Rescale(float(row['chlorides']), .611, .012),
                    self.Rescale(float(row['free sulfur dioxide']), 72, 1),
                    self.Rescale(float(row['total sulfur dioxide']), 289, 6),
                    self.Rescale(float(row['density']), 1.0037, .9901),
                    self.Rescale(float(row['pH']), 4.01, 2.74),
                    self.Rescale(float(row['sulphates']), 2, .33),
                    self.Rescale(float(row['alcohol']), 14.9, 8.4)
                ]
                quality = int(row['quality'])
                self.samples.append(DataPoint(inputs, quality)) 
        self.Compute_Average()