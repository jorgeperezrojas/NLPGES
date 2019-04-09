from sklearn.model_selection import train_test_split


def split_proporcionaly(train_percent, test_percent, data):
    """
    :param training: percentage
    :param test: percentage
    :param data: is a dict key : PRESTA_EST, VALUE : array of SOSPECHA_DIAG
    :return: training data dict test data dict
    """
    train_data = dict()
    test_data = dict()

    for key in data.keys():
        train, test = train_test_split(data[key], test_size=test_percent, training_size=train_percent)
        train_data[key] = train
        test_data[key] = test
    return train_data, test_data
