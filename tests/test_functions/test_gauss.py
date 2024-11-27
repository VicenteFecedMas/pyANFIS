"""This will test that all the functionalities in the Gauss function to check if they work as expected"""
import random
from typing import Union
import unittest
import torch

from pyanfis.functions import Gauss

class TestGauss(unittest.TestCase):
    """Gauss function test cases"""
    def test_initialization_empty(self):
        """Initialising empty variables"""
        function = Gauss()
        self.assertTrue(torch.equal(function.mean, torch.tensor([])))
        self.assertTrue(torch.equal(function.std, torch.tensor([])))
    def test_initialization_center(self):
        """Initialising mean only"""
        function = Gauss(mean=1)
        self.assertEqual(function.mean, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertTrue(torch.equal(function.std, torch.tensor([])))
    def test_initialization_shape(self):
        """Initialising std only"""
        function = Gauss(std=1)
        self.assertTrue(torch.equal(function.mean, torch.tensor([])))
        self.assertEqual(function.std, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_initialization_all_parameters(self):
        """Initialising all parameters"""
        function = Gauss(1, 1)
        self.assertEqual(function.mean, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertEqual(function.std, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_invalid_initialisation_string(self):
        """Invalid Initialisations with strings"""
        self.assertRaises(TypeError, Gauss, {"mean":"string"})
        self.assertRaises(TypeError, Gauss, {"std":"string"})
    def test_invalid_initialisation_list(self):
        """Invalid Initialisations with lists"""
        self.assertRaises(TypeError, Gauss, {"mean":[1]})
        self.assertRaises(TypeError, Gauss, {"std":[1]})
    def test_invalid_initialisation_tuple(self):
        """Invalid Initialisations with tuples"""
        self.assertRaises(TypeError, Gauss, {"mean":(1)})
        self.assertRaises(TypeError, Gauss, {"std":(1)})
    def test_valid_asignation_int(self):
        """Valid Asignation with int"""
        function = Gauss()
        function.mean = 1 # type: ignore
        self.assertEqual(function.mean, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.std = 1 # type: ignore
        self.assertEqual(function.std, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_valid_asignation_float(self):
        """Valid Asignation with float"""
        function = Gauss()
        function.mean = 1.0 # type: ignore
        self.assertEqual(function.mean, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.std = 1.0 # type: ignore
        self.assertEqual(function.std, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_valid_asignation_tensor(self):
        """Valid Asignation with tensor"""
        function = Gauss()
        function.mean = torch.tensor(1.0)
        self.assertEqual(function.mean, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.std = torch.tensor(1.0)
        self.assertEqual(function.std, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_invalid_asignation_string(self):
        """Invalid Asignation with string"""
        function = Gauss()
        with self.assertRaises(TypeError):
            function.mean = "string" # type: ignore
        with self.assertRaises(TypeError):
            function.std = "string" # type: ignore
    def test_invalid_asignation_list(self):
        """Invalid Asignation with list"""
        function = Gauss()
        with self.assertRaises(TypeError):
            function.mean = ["string"] # type: ignore
        with self.assertRaises(TypeError):
            function.std = ["string"] # type: ignore
    def test_invalid_asignation_tuple(self):
        """Invalid Asignation with tuple"""
        function = Gauss()
        with self.assertRaises(TypeError):
            function.mean = ("string") # type: ignore
        with self.assertRaises(TypeError):
            function.std = ("string") # type: ignore
    def test_forward_return_shape(self):
        """Shape of forward pass of function"""
        function = Gauss(1, 1)
        shapes: list[tuple[int, ...]] = [
            (1,),
            (1, 2),
            (1, 2, 3),
            (1, 2, 3, 4),
            (4, 3, 2, 1),
            (4, 3, 2),
            (4, 3),
            (4,),
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                x = torch.rand(*shape)
                self.assertEqual(x.shape, function(x).shape)
    def test_forward_return_values(self):
        """Values of forward pass of function"""
        mean = random.randint(1, 10)
        std = random.randint(1, 10)
        function = Gauss(
            mean=mean,
            std=std
        )
        def equation(
                x: torch.Tensor, 
                mean: Union[int, float],
                std: Union[int, float], 
            ) -> torch.Tensor:
            """Comparative Gauss equation"""
            return torch.exp(-((x - mean)** 2)/ (2 * (std ** 2)))
        shapes: list[tuple[int, ...]] = [
            (1,),
            (1, 2),
            (1, 2, 3),
            (1, 2, 3, 4),
            (4, 3, 2, 1),
            (4, 3, 2),
            (4, 3),
            (4,),
        ]
        for tensor_shape in shapes:
            x = torch.rand(*tensor_shape)
            y_1 = equation(
                x=x,
                mean=mean,
                std=std,
            )
            y_2 = function(x)
            with self.subTest(shape=tensor_shape):
                self.assertTrue(torch.equal(y_1, y_2))
    def test_gradient_exists(self):
        """Backpropagation of the function"""
        function = Gauss(1, 1)
        shapes: list[tuple[int, ...]] = [
            (1,),
            (1, 2),
            (1, 2, 3),
            (1, 2, 3, 4),
            (4, 3, 2, 1),
            (4, 3, 2),
            (4, 3),
            (4,),
        ]
        for tensor_shape in shapes:
            with self.subTest(shape=tensor_shape):
                x = torch.rand(*tensor_shape)
                y = function(x)
                y.backward(y)
                self.assertIsNotNone(function.mean.grad)
                self.assertIsNotNone(function.std.grad)