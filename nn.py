class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, grads):
        all_grads = []
        for layer in reversed(self.layers):
            grads = layer.backword(grads)
            all_grads.append(layer.grads)

        return all_grads[::-1]

    def get_params_grads(self):
        for layer in self.layers:
            yield layer.paras, layer.grads

    def set_params(self, params):
        for i, layer in self.layers:
            for key in layer.paras.keys():
                layer.paras[key] = params[i][key]

