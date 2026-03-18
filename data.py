import math

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
     def __init__(self, inputs, quality, rawInputs):
        self.inputs = inputs      # list[float], length = 11
        self.quality = quality    # int
        self.rawInputs = rawInputs

class DataSet:
    def __init__(self):
        self.samples = []  # start empty
        self.weirdness = []
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
        return None

    def Normalize(self, Value, Upper_bound, Lower_bound):
        return min(1.0, max(0.0, (Value - Lower_bound) / (Upper_bound - Lower_bound)))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def Compute_z_score(self):
        #print(f"samples count: {len(self.samples)}")
        n = len(self.samples)
        n_features = len(self.samples[0].rawInputs)

        # compute mean per feature
        means = []
        for i in range(n_features):
            means.append(sum(s.rawInputs[i] for s in self.samples) / n)

        # compute std per feature
        stds = []
        for i in range(n_features):
            variance = sum((s.rawInputs[i] - means[i]) ** 2 for s in self.samples) / n
            stds.append(variance ** 0.5)

        # compute weirdness per sample
        self.weirdness = []
        for s in self.samples:
            score = sum(((s.rawInputs[i] - means[i]) / stds[i]) ** 2
                        for i in range(n_features) if stds[i] != 0)
            self.weirdness.append(score)


    def load_from_csv(self, file_path, start, end, isTraining):
        import csv
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for i, row in enumerate(reader):
                if i < start:
                    continue
                if end is not None and i >= end:
                    break

                rawInputs = []
                inputs = []
                for col, upper, lower, transforms in PARAMETERS:
                    val = self.Normalize(float(row[col]), upper, lower)
                    if 'raw' in transforms:
                        inputs.append(val)
                        rawInputs.append(val)
                    if 'square' in transforms: inputs.append(val ** 2)
                    if 'sqrt' in transforms: inputs.append(val ** 0.5)
                    if 'inverse' in transforms:
                        sig_inv = self.sigmoid(1 / val) if val != 0 else 0
                        inputs.append(sig_inv)
                    if 'cube' in transforms: inputs.append(val ** 3)

                num_base = len(inputs)

                for i in range(num_base):
                    for j in range(i+1, num_base):
                        inputs.append(inputs[i] * inputs[j])

                quality = int(row['quality'])
                #print(f"samples countin load: {len(self.samples)}")

                self.samples.append(DataPoint(inputs, quality, rawInputs))
        #self.Compute_Average()
        self.Compute_z_score()
