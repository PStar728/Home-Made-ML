PARAMETERS = [
    ('fixed acidity',        15.9,   4.6,   ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P1
    ('volatile acidity',     1.58,   0.12,  ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P2
    ('citric acid',          1.66,   0,     ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P3
    ('residual sugar',       15.5,   0.9,   ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P4
    ('chlorides',            0.611,  0.012, ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P5
    ('free sulfur dioxide',  72,     1,     ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P6
    ('total sulfur dioxide', 289,    6,     ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P7
    ('density',              1.0037, 0.9901,['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P8
    ('pH',                   4.01,   2.74,  ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P9
    ('sulphates',            2,      0.33,  ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P10
    ('alcohol',              14.9,   8.4,   ['raw', 'square', 'sqrt', 'inverse', 'cube']),  # P11
]

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

    def get_parameters(self, rowName):
        parameters = []
        self.Normalize

    def load_from_csv(self, file_path, start, end=None):
        import csv
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for i, row in enumerate(reader):
                if i < start:
                    continue
                if end is not None and i >= end:
                    break

                inputs = []
                for col, upper, lower, transforms in PARAMETERS:
                    val = self.Normalize(float(row[col]), upper, lower)
                    if 'raw' in transforms: inputs.append(val)
                    if 'square' in transforms: inputs.append(val ** 2)
                    if 'sqrt' in transforms: inputs.append(val ** 0.5)
                    if 'inverse' in transforms: inputs.append(1 / val if val != 0 else 0)
                    if 'cube' in transforms: inputs.append(val ** 3)

                quality = int(row['quality'])
                self.samples.append(DataPoint(inputs, quality))
        self.Compute_Average()