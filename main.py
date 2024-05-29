import math
import random
import time

import pygame



# https://youtu.be/hfMk-kjRv4c?si=lIxiKkFQ_n5resQN

def index_of_max_value(array):
    max_value = float("-inf")
    best_index = None
    for i, x in enumerate(array):
        if x > max_value:
            max_value = x
            best_index = i
    return best_index





def node_cost(activation, expected_output):
    return (activation - expected_output) ** 2
def node_derivative_cost(activation, expected_output):
    return 2 * (activation - expected_output)


def step_activation(weighted_input):
    if weighted_input > 0:
        return 1
    else:
        return 0
def step_derivative_activation(weighted_input):
    return 0

def sigmoid_activation(weighted_input):
    return 1 / (1 + math.exp(-weighted_input))
def sigmoid_derivative_activation(weighted_input):
    a = sigmoid_activation(weighted_input)
    return a * (1 - a)

def relu_activation(weighted_input):
    if weighted_input > 0:
        return weighted_input
    else:
        return 0
def relu_derivative_activation(weighted_input):
    if weighted_input > 0:
        return 1
    else:
        return 0


class Layer:
    def __init__(self, input_node_count, output_node_count, activation_function_type="Sigmoid"):#, weights, biases):
        self.input_node_count = input_node_count
        self.output_node_count = output_node_count

        # https://youtu.be/hfMk-kjRv4c?si=29HR3nxxcH9byHm8&t=734
        # there are more options!!
        if activation_function_type == "Step":
            self.activation_function = step_activation
            self.derivative_activation_function = step_derivative_activation
        elif activation_function_type == "Sigmoid":
            self.activation_function = sigmoid_activation
            self.derivative_activation_function = sigmoid_derivative_activation
        elif activation_function_type == "Hyperbolic Tangent":
            raise NotImplementedError
        elif activation_function_type == "SiLU":
            raise NotImplementedError
        elif activation_function_type == "ReLU":
            self.activation_function = relu_activation
            self.derivative_activation_function = relu_derivative_activation
        else:
            raise ValueError(activation_function_type)
        self.activation_function_type = activation_function_type

        # dict
        # (node_out * self.input_node_count + node_in): weight
        self.weights = [0 for _ in range(self.input_node_count * self.output_node_count)]

        # list
        # node_index: bias
        self.biases = [0 for _ in range(self.output_node_count)]

        self.input_activations = [0 for _ in range(self.input_node_count)]
        self.weighted_inputs = [0 for _ in range(self.output_node_count)]
        self.output_activations = [0 for _ in range(self.output_node_count)]

        self.weight_cost_gradient = [0 for _ in self.weights]
        self.bias_cost_gradient = [0 for _ in self.biases]

    def calculate_outputs(self, inputs):
        self.input_activations = inputs
        for node_out in range(self.output_node_count):
            self.weighted_inputs[node_out] = self.biases[node_out]
            for node_in in range(self.input_node_count):
                self.weighted_inputs[node_out] += inputs[node_in] * self.weights[node_out * self.input_node_count + node_in]

            self.output_activations[node_out] = self.activation_function(self.weighted_inputs[node_out])
        return self.output_activations
    
    def initialize_randomness(self):
        self.weights = [random.random() * 2 - 1 for _ in self.weights]
        # self.biases = [random.random() * 2 - 1 for _ in self.weights]
        if self.activation_function_type == "Step":
            raise NotImplementedError
        elif self.activation_function_type == "Sigmoid":
            d = math.sqrt(self.input_node_count)
            self.weights = [x / d for x in self.weights]
            # self.biases = [x / d for x in self.biases]
        elif self.activation_function_type == "Hyperbolic Tangent":
            raise NotImplementedError
        elif self.activation_function_type == "SiLU":
            raise NotImplementedError
        elif self.activation_function_type == "ReLU":
            raise NotImplementedError
        else:
            raise ValueError(self.activation_function_type)
    
    def apply_gradient(self, learn_rate):
        for node_out in range(self.output_node_count):
            change = self.bias_cost_gradient[node_out]
            for node_in in range(self.input_node_count):
                change += self.weight_cost_gradient[node_out * self.input_node_count + node_in]
            self.biases[node_out] -= change * learn_rate
    
    def calculate_output_layer_node_values(self, expected_output):
        node_values = []
        for i in range(self.output_node_count):
            cost_derivative = self.derivative_activation_function(self.weighted_inputs[i])
            activation_derivative = node_derivative_cost(self.output_activations[i], expected_output[i])
            node_values.append(activation_derivative * cost_derivative)
        return node_values
    
    def calculate_hidden_layer_node_values(self, old_layer, old_node_values):
        new_node_values = []

        for new_node in range(self.output_node_count):
            new_node_value = 0
            for old_node in range(old_layer.output_node_count):
                # how is this a derivative?
                weighted_input_derivative = old_layer.weights[new_node * old_layer.output_node_count + old_node]
                new_node_value += weighted_input_derivative * old_node_values[old_node]
            new_node_value *= self.derivative_activation_function(self.weighted_inputs[new_node])
            new_node_values.append(new_node_value)
        
        return new_node_values
    
    def update_gradients(self, node_values):
        for node_out in range(self.output_node_count):
            for node_in in range(self.input_node_count):
                derivative_cost_wrt_weight = self.input_activations[node_in] * node_values[node_out]
                self.weight_cost_gradient[node_out * self.input_node_count + node_in] += derivative_cost_wrt_weight

            derivative_cost_wrt_bias = 1 * node_values[node_out]
            self.bias_cost_gradient[node_out] += derivative_cost_wrt_bias
    
    def clear_gradients(self):
        self.weight_cost_gradient = [0 for _ in self.weights]
        self.bias_cost_gradient = [0 for _ in self.biases]


class NeuralNetwork:
    def __init__(self, layer_sizes, init_random=True):
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
        self.layers[-1] = self.layers[-1]

        if init_random:
            for layer in self.layers:
                layer.initialize_randomness()

    def calculate_outputs(self, inputs, classify=True):
        inputs = list(inputs)
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)

        if classify:
            self.classify()

        return self.layers[-1].output_activations

    def classify(self):
        i = index_of_max_value(self.layers[-1].output_activations)

        self.layers[-1].output_activations = [0 for _ in range(self.layer_sizes[-1])]
        self.layers[-1].output_activations[i] = 1

        return self.layers[-1].output_activations

    def cost(self, data_input, expected_output):
        self.calculate_outputs(data_input)

        cost = 0
        for i, output in enumerate(self.layers[-1].output_activations):
            cost += node_cost(output, expected_output[i])

        return cost

    def average_cost(self, data_expected_pairs):
        total_cost = 0

        for data, expected in data_expected_pairs:
            total_cost += self.cost(data, expected)

        return total_cost / len(data_expected_pairs)
    
    def apply_all_gradients(self, learn_rate):
        for layer in self.layers:
            layer.apply_gradient(learn_rate)

    def learn_numeric_method(self, training_data, learn_rate):
        h = 0.001
        original_cost = self.average_cost(training_data)

        for layer in self.layers:
            for i in range(len(layer.weights)):
                layer.weights[i] += h
                layer.weight_cost_gradient[i] = (self.average_cost(training_data) - original_cost) / h
                layer.weights[i] -= h
                
            for i in range(len(layer.biases)):
                layer.biases[i] += h
                layer.bias_cost_gradient[i] = (self.average_cost(training_data) - original_cost) / h
                layer.biases[i] -= h
    
        self.apply_all_gradients(learn_rate)
    
    def learn(self, training_batch, learn_rate):
        self.clear_all_gradients()
    
        for data_point in training_batch:
            self.update_all_gradients(data_point)
        
        self.apply_all_gradients(learn_rate / len(training_batch))

    def update_all_gradients(self, data_expected_pair):
        data, expected = data_expected_pair
        self.calculate_outputs(data)

        output_layer = self.layers[-1]
        node_values = output_layer.calculate_output_layer_node_values(expected)
        output_layer.update_gradients(node_values)

        for i in range(len(self.layers)-2, -1, -1):
            hidden_layer = self.layers[i]
            node_values = hidden_layer.calculate_hidden_layer_node_values(self.layers[i+1], node_values)
            hidden_layer.update_gradients(node_values)
    
    def clear_all_gradients(self):
        for layer in self.layers:
            layer.clear_gradients()
    
    # def backpropagation(self, training_data, learn_rate):
    #     for i in range(len(self.layers), 0, -1):
    #         print(i)


training_data = []
for i in range(12345):
    x = random.random()
    y = random.random()
    # x = (i // 100 + 0.5) / 100
    # y = (i % 100 + 0.5) / 100
    # if x >= 1: break

    red = 0
    green = 0
    blue = 0

    # if (x - 300) ** 2 + (y - 300) ** 2 < 200 ** 2:
    #     red = 1
    # if (x - 600) ** 2 + (y - 500) ** 2 < 300 ** 2:
    #     blue = 1
    # if (x - 700) ** 2 + (y - 100) ** 2 < 150 ** 2:
    #     green = 1

    # if (x - 0.6) ** 2 + (y - 0.3) ** 2 < 0.4 ** 2:
    if (x - 0) ** 2 + (y - 0) ** 2 < 0.6 ** 2:
        green = 1
    else:
        red = 1

    # training_data.append(((x, y), (red, green, blue)))
    training_data.append(((x, y), (red, green)))

network = NeuralNetwork((2, 3, 2))



# for layer in network.layers:
#     for i, weight in enumerate(layer.weights):
#         print(layer.weights[i])
#     print()
#     for i, bias in enumerate(layer.biases):
#         print(layer.biases[i])

# print()

# # first point
# xy = (
#     0.5 / 10,
#     0.5 / 10
# )
# print(xy)
# print(network.calculate_outputs(xy))
# print(network.layers[-1].weighted_inputs)
# print(network.layers[-1].output_activations)


# input()


training_batch_size = 10000
training_batch_index = 0
training_batch = []
def next_training_batch():
    global training_batch_size, training_batch_index, training_batch

    # print(training_batch_index)

    assert training_batch_size <= len(training_data)

    training_batch = training_data[training_batch_index : training_batch_index + training_batch_size]
    
    training_batch_index += training_batch_size
    training_batch_index %= len(training_data)

    if len(training_batch) < training_batch_size:
        next_training_batch()
next_training_batch()


pygame.init()
screen = pygame.display.set_mode((1000, 700))

def draw():
    screen.fill((128, 128, 128))

    screen_size = screen.get_rect()

    display_size = screen_size.h - 20
    transform = lambda x: x * display_size + 10

    resolution = 100
    tile_size = int(display_size / resolution) + 1

    for i in range(100):
        for j in range(100):
            x = i / 100
            y = j / 100
            color = network.calculate_outputs((x + (0.5/100), y + (0.5/100)))
            color = (
                color[0] * 255 // 2,
                color[1] * 255 // 2,
                # color[2] * 255 // 2
                0
            )
            pygame.draw.rect(screen, color, (transform(x), transform(y), tile_size, tile_size))

    for d in training_batch:
        pygame.draw.circle(
            screen,
            (
                d[1][0] * 255,
                d[1][1] * 255,
                # d[1][2] * 255
                0
            ),
            (
                transform(d[0][0]),
                transform(d[0][1])
            ),
            1
        )
    
    pos = [display_size + 20, 10]

    def draw_text(pos, txt):
        font = pygame.font.Font("freesansbold.ttf", 20)
        text = font.render(txt, True, "white")
        screen.blit(text, pos)
        pos[1] += 20
    
    draw_text(pos, f"Cost: {network.average_cost(training_data)}")

    pygame.display.flip()


print("start!")

# clock = pygame.time.Clock()
while True:
    draw()
    pygame.event.get()
    print()

    next_draw = time.time() + 0.2
    while time.time() < next_draw:

        network.learn(training_batch, 0.001)
        print("cost", network.average_cost(training_batch))
        next_training_batch()

    # print()
    # for layer in network.layers:
    #     # for w in layer.weight_cost_gradient:
    #     for w in layer.weights:
    #         print(w)
    #     print()
    #     # for b in layer.bias_cost_gradient:
    #     for b in layer.biases:
    #         print(b)
    #     print()
    # print()

    # clock.tick(20)




# layer = Layer(3, 2)

# layer.weights = [
#     9, 8, 7,
#     3, 2, 1
# ]
# layer.biases = [
#     1000,
#     2000
# ]

# out = layer.calculate_outputs([3, 4, 5])
# print(out)
