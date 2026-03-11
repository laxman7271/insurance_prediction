import pickle
import numpy as np

class Insurance_Prediction:

    def __init__(self):

        with open("artifacts/scaler.pki", "rb") as f:
            self.scaler = pickle.load(f)

        with open("artifacts/model.pki", "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):

        input_data = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])

        scaled_input = self.scaler.transform(input_data)

        result = self.model.predict(scaled_input)[0]

        # prevent negative values
        result = max(result, 0)

        return result
