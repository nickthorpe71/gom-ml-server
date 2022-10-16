import torch
import random
from model import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

floor_width = random.randint(4, 8)
floor_length = random.randint(4, 8)
num_rooms = floor_width * floor_length
output_size = num_rooms * 6


def get_binary_string(num, length):
    return bin(num)[2:].zfill(length)


def generate_input(floor_level, biome_code):
    # floor level must be between 0 and 127
    # biome code must be 4 characters long

    # convert floor level to binary (7 digit binary) (max floor level is 127)
    floor_level_binary = get_binary_string(floor_level, 7)

    # create 12 digit random binary string
    seed_binary = get_binary_string(random.randint(0, 4095), 12)

    # convert biome code to binary (16 digit binary)
    biome_code_binary = ''
    for char in biome_code:
        biome_code_binary += get_binary_string(ord(char), 8)

    # concatenate binary strings
    input_string = floor_level_binary + seed_binary + biome_code_binary

    # convert to list of floats
    input_list = [float(char) for char in input_string]

    return input_list


model = NeuralNet(51, 435, output_size).to(device)

input = torch.Tensor(generate_input(1, "cave")).to(device)

print(input)
print(len(input))

outputs = model(input)

print(outputs)
