# %load submission_12414038.py

"""Submission for exercise sheet 2

SUBMIT this file as submission_<STUDENTID>.py where
you replace <STUDENTID> with your student ID, e.g.,
submission_12414038.py
"""

from typing import Callable

import torch
import torch.nn as nn


# Exercise 2.1 (AND gate)
def assignment_ex1(x: torch.tensor) -> Callable[[torch.tensor], torch.tensor]:
    # YOUR CODE GOES HERE

    # Define the inputs
    # input was x1 =[0,0]⊤, x2 =[0,1]⊤, x3 =[1,0] ⊤, x4 =[1,1]⊤. transposing a vector or matrix is not represented differently when printed as an array. The notation of the array stays the same, but internally, the data is interpreted or structured differently
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    
    # Linear transformation
    f = nn.Linear(
        2, 1
    )  # the mathematical operation for nn.Linear() is x↦⟨w,x⟩+b. this takes 2 inputs and returns 1 output

    # Set the weights and bias manually
    # Here, to make f(x)>0 it takes ⟨w,x⟩+b=(1⋅x 1)+(1⋅x 1)−1.5 to be >0. it can be upto 1.9 since max possible value is 2 for ⟨w,x⟩
    f.weight = nn.Parameter(torch.tensor([[1.0, 1.0]]))  # AND gate weights.
    f.bias = nn.Parameter(
        torch.tensor([-1.5])
    )  # AND gate bias. A bias of −1.5 ensures that the output only becomes positive when both inputs are 1.

    # Apply the linear transformation and thresholding
    outputs = (f(x) > 0).int()  # outputs will be 0 or 1
    return outputs
    pass


# Exercise 2.2 (OR gate)
def assignment_ex2(x: torch.tensor) -> Callable[[torch.tensor], torch.tensor]:
    # YOUR CODE GOES HERE
    
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

    # Linear transformation
    f = nn.Linear(2, 1)

    # Set the weights and bias manually for OR gate
    # Here, ⟨w,x⟩+b=(1⋅x 1)+(1⋅x 0)−1.5 to be >0. one of the element can be 0 to become this condition true.
    # the bias can be upto -0.1 since min possible value is 0 for ⟨w,x⟩ when both values are 0
    f.weight = nn.Parameter(torch.tensor([[1.0, 1.0]]))  # OR gate weights
    f.bias = nn.Parameter(torch.tensor([-0.5]))  # OR gate bias

    # Apply the linear transformation and thresholding
    outputs = (f(x) > 0).int()  # outputs will be 0 or 1
    print("Outputs for OR gate:")
    print(outputs)
    return outputs

    pass
