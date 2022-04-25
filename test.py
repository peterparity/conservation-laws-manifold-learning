from sklearn.preprocessing import StandardScaler  # , RobustScaler, MaxAbsScaler


def preprocess(data, scaler=StandardScaler):
    scaled_data = scaler().fit_transform(data.reshape(-1, 2)).reshape(data.shape)
    return scaled_data
