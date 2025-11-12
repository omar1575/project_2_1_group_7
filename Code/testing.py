


def get_hardware():
    # Get hardware
    pass
def get_hyperparametrs():
    # Get parametrs(or generate random)
    pass
def get_data_for_training():
    pass



class ModelAdapter:
    #Instead of model
    def __init__(self):
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        pass


class ProgramAdapter:
    def run_training():
        pass



if __name__ == "__main__":
    
    model = ModelAdapter()
    program = ProgramAdapter()
    X_train, y_train = get_data_for_training()

    model.train(X_train, y_train)

    hardware = get_hardware()
    hyperparametrs = get_hyperparametrs()

    # Change to appropiate data types
    X_test = hardware + hyperparametrs

    predictions = model.predict(X_test)
    print("Predicted by model info:" + predictions)

    results = program.run_training(hyperparametrs)
    print("Actual results:" + results)

