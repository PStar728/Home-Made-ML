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
        
    def Normalize(self, Value, Upper_bound, Lower_bound):
        return min(1.0, max(0.0, (Value - Lower_bound) / (Upper_bound - Lower_bound)))

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
                    self.Normalize(float(row['fixed acidity']), 15.9, 4.6),             # P1
                    self.Normalize(float(row['fixed acidity']), 15.9, 4.6) ** 2,        # P1^2
                    self.Normalize(float(row['fixed acidity']), 15.9, 4.6) ** 0.5,      # SQRT(P1)
                    self.Normalize(float(row['volatile acidity']), 1.58, 0.12),         # P2
                    self.Normalize(float(row['volatile acidity']), 1.58, 0.12) ** 2,                           # P2^2
                    self.Normalize(float(row['volatile acidity']), 1.58, 0.12) ** 0.5,                         # SQRT(P2)
                    self.Normalize(float(row['citric acid']), 1.66, 0),                 # P3
                    self.Normalize(float(row['citric acid']), 1.66, 0) ** 2,            # P3^2
                    self.Normalize(float(row['citric acid']), 1.66, 0) ** 0.5,                                 # SQRT(P3)
                    self.Normalize(float(row['residual sugar']), 15.5, 0.9),            # P4
                    self.Normalize(float(row['residual sugar']), 15.5, 0.9) ** 2,       # P4^2
                    self.Normalize(float(row['residual sugar']), 15.5, 0.9) ** 0.5,                            # SQRT(P4)
                    self.Normalize(float(row['chlorides']), .611, .012),                # P5
                    self.Normalize(float(row['chlorides']), .611, .012) ** 2,           # P5^2
                    self.Normalize(float(row['chlorides']), .611, .012) ** 0.5,                                # SQRT(P5)
                    self.Normalize(float(row['free sulfur dioxide']), 72, 1),           # P6
                    self.Normalize(float(row['free sulfur dioxide']), 72, 1) ** 2,                             # P6^2
                    self.Normalize(float(row['free sulfur dioxide']), 72, 1) ** 0.5,    # SQRT(P6)
                    self.Normalize(float(row['total sulfur dioxide']), 289, 6),         # P7
                    self.Normalize(float(row['total sulfur dioxide']), 289, 6) ** 2,    # P7^2
                    self.Normalize(float(row['total sulfur dioxide']), 289, 6) ** 0.5,  # SQRT(P7)
                    self.Normalize(float(row['density']), 1.0037, .9901),               # P8
                    self.Normalize(float(row['density']), 1.0037, .9901) ** 2,                                 # P8^2
                    self.Normalize(float(row['density']), 1.0037, .9901) ** 0.5,        # SQRT(P8)
                    self.Normalize(float(row['pH']), 4.01, 2.74),                       # P9
                    self.Normalize(float(row['pH']), 4.01, 2.74) ** 2,                  # P9^2
                    self.Normalize(float(row['pH']), 4.01, 2.74) ** 0.5,                                       # SQRT(P9)
                    self.Normalize(float(row['sulphates']), 2, .33),                    # P10
                    self.Normalize(float(row['sulphates']), 2, .33) ** 2,               # P10^2
                    self.Normalize(float(row['sulphates']), 2, .33) ** 0.5,             # SQRT(P10)
                    self.Normalize(float(row['alcohol']), 14.9, 8.4),                   # P11
                    self.Normalize(float(row['alcohol']), 14.9, 8.4) ** 2,              # P11^2
                    self.Normalize(float(row['alcohol']), 14.9, 8.4) ** 0.5                                    # SQRT(P11)

                ]
                quality = int(row['quality'])
                self.samples.append(DataPoint(inputs, quality)) 
        self.Compute_Average()