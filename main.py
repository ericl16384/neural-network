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

class Layer:
    def __init__(self, input_node_count, output_node_count, activation_function_type="Sigmoid"):#, weights, biases):
        self.input_node_count = input_node_count
        self.output_node_count = output_node_count

        # https://youtu.be/hfMk-kjRv4c?si=29HR3nxxcH9byHm8&t=734
        # there are more options!!
        if activation_function_type == "Step":
            self.activation_function = lambda x: int(x > 0)
        elif activation_function_type == "Sigmoid":
            self.activation_function = lambda x: 1 / (1 + math.exp(-x))
        elif activation_function_type == "Hyperbolic Tangent":
            raise NotImplementedError
        elif activation_function_type == "SiLU":
            raise NotImplementedError
        elif activation_function_type == "ReLU":
            self.activation_function = lambda x: max(0, x)
        else:
            raise ValueError(activation_function_type)
        self.activation_function_type = activation_function_type

        # dict
        # (node_out * self.input_node_count + node_in): weight
        self.weights = [0 for _ in range(self.input_node_count * self.output_node_count)]

        # list
        # node_index: bias
        self.biases = [0 for _ in range(self.output_node_count)]

        self.initialize_randomness()

        self.last_outputs = [0 for _ in range(self.output_node_count)]

    def calculate_outputs(self, inputs):
        for node_out in range(self.output_node_count):
            self.last_outputs[node_out] = self.biases[node_out]
            for node_in in range(self.input_node_count):
                self.last_outputs[node_out] += inputs[node_in] * self.weights[node_out * self.input_node_count + node_in]

            self.last_outputs[node_out] = self.activation_function(self.last_outputs[node_out])
        return self.last_outputs
    
    def initialize_randomness(self):
        self.weights = [random.random() * 2 - 1 for _ in self.weights]
        self.biases = [random.random() * 2 - 1 for _ in self.weights]

        if self.activation_function_type == "Step":
            raise NotImplementedError
        elif self.activation_function_type == "Sigmoid":
            d = math.sqrt(self.input_node_count)
            self.weights = [x / d for x in self.weights]
            self.biases = [x / d for x in self.biases]
        elif self.activation_function_type == "Hyperbolic Tangent":
            raise NotImplementedError
        elif self.activation_function_type == "SiLU":
            raise NotImplementedError
        elif self.activation_function_type == "ReLU":
            raise NotImplementedError
        else:
            raise ValueError(self.activation_function_type)

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def calculate_outputs(self, inputs, classify=True):
        inputs = list(inputs)
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)

        if classify:
            self.classify()

        return self.layers[-1].last_outputs

    def classify(self):
        i = index_of_max_value(self.layers[-1].last_outputs)

        self.layers[-1].last_outputs = [0 for _ in range(self.layer_sizes[-1])]
        self.layers[-1].last_outputs[i] = 1

        return self.layers[-1].last_outputs

    def cost(self, data_input, expected_output):
        self.calculate_outputs(data_input)
        
        output = self.layers[-1].last_outputs

        cost = 0
        for i in range(self.layer_sizes[-1]):
            cost += (output[i] - expected_output[i]) ** 2

        return cost

    def average_cost(self, data_expected_pairs):
        total_cost = 0

        for data, expected in data_expected_pairs:
            total_cost += self.cost(data, expected)

        return total_cost / len(data_expected_pairs)


input_data = []
for i in range(1000):
    x = random.random()
    y = random.random()

    red = 0
    green = 0
    blue = 0

    # if (x - 300) ** 2 + (y - 300) ** 2 < 200 ** 2:
    #     red = 1
    # if (x - 600) ** 2 + (y - 500) ** 2 < 300 ** 2:
    #     blue = 1
    # if (x - 700) ** 2 + (y - 100) ** 2 < 150 ** 2:
    #     green = 1

    if (x - 0.6) ** 2 + (y - 0.3) ** 2 < 0.4 ** 2:
        green = 1
    else:
        red = 1

    input_data.append(((x, y), (red, green, blue)))




network = NeuralNetwork((2, 3))

# for layer in network.layers:
#     for i, weight in enumerate(layer.weights):
#         layer.weights[i] = random.random() * 2 - 0.5

#         # print(layer.weights[i])
    
#     # print()

#     for i, bias in enumerate(layer.biases):
#         layer.biases[i] = random.random() * 2 - 0.5

#         # print(layer.biases[i])



pygame.init()
screen = pygame.display.set_mode((700, 700))

def draw():
    screen.fill((128, 128, 128))

    screen_size = screen.get_rect()

    display_size = min(screen_size.w, screen_size.h) - 100
    transform = lambda x: x * display_size + 50

    resolution = 100
    tile_size = display_size / resolution

    for i in range(100):
        for j in range(100):
            x = i / 100
            y = j / 100
            color = network.calculate_outputs((x, y))
            color = (
                color[0] * 255 // 2,
                color[1] * 255 // 2,
                color[2] * 255 // 2,
            )
            pygame.draw.rect(screen, color, (transform(x) - tile_size/2, transform(y) - tile_size/2, tile_size, tile_size))

    for d in input_data:
        pygame.draw.circle(
            screen,
            (
                d[1][0] * 255,
                d[1][1] * 255,
                d[1][2] * 255
            ),
            (
                transform(d[0][0]),
                transform(d[0][1])
            ),
            5
        )

    pygame.display.flip()


print(network.average_cost(input_data))
draw()

for i in range(1000):
    pygame.event.get()
    time.sleep(1)




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
