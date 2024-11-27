"""This will test that all the functionalities in the LinearS function to check if they work as expected"""
import random
from typing import Union
import unittest
import torch

from pyanfis.functions import LinearS

class TestLinearS(unittest.TestCase):
    """Gauss function test cases"""
    def test_initialization_empty(self):
        """Initialising empty variables"""
        function = LinearS()
        self.assertTrue(torch.equal(function.foot, torch.tensor([])))
        self.assertTrue(torch.equal(function.shoulder, torch.tensor([])))
    def test_initialization_center(self):
        """Initialising foot only"""
        function = LinearS(foot=1)
        self.assertEqual(function.foot, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertTrue(torch.equal(function.shoulder, torch.tensor([])))
    def test_initialization_shape(self):
        """Initialising shoulder only"""
        function = LinearS(shoulder=1)
        self.assertTrue(torch.equal(function.foot, torch.tensor([])))
        self.assertEqual(function.shoulder, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_initialization_all_parameters(self):
        """Initialising all parameters"""
        function = LinearS(1, 1)
        self.assertEqual(function.foot, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertEqual(function.shoulder, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_invalid_initialisation_string(self):
        """Invalid Initialisations with strings"""
        self.assertRaises(TypeError, LinearS, {"foot":"string"})
        self.assertRaises(TypeError, LinearS, {"shoulder":"string"})
    def test_invalid_initialisation_list(self):
        """Invalid Initialisations with lists"""
        self.assertRaises(TypeError, LinearS, {"foot":[1]})
        self.assertRaises(TypeError, LinearS, {"shoulder":[1]})
    def test_invalid_initialisation_tuple(self):
        """Invalid Initialisations with tuples"""
        self.assertRaises(TypeError, LinearS, {"foot":(1)})
        self.assertRaises(TypeError, LinearS, {"shoudler":(1)})
    def test_valid_asignation_int(self):
        """Valid Asignation with int"""
        function = LinearS()
        function.foot = 1 # type: ignore
        self.assertEqual(function.foot, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.shoulder = 1 # type: ignore
        self.assertEqual(function.shoulder, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_valid_asignation_float(self):
        """Valid Asignation with float"""
        function = LinearS()
        function.foot = 1.0 # type: ignore
        self.assertEqual(function.foot, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.shoulder = 1.0 # type: ignore
        self.assertEqual(function.shoulder, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_valid_asignation_tensor(self):
        """Valid Asignation with tensor"""
        function = LinearS()
        function.foot = torch.tensor(1.0)
        self.assertEqual(function.foot, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.shoulder = torch.tensor(1.0)
        self.assertEqual(function.shoulder, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_invalid_asignation_string(self):
        """Invalid Asignation with string"""
        function = LinearS()
        with self.assertRaises(TypeError):
            function.foot = "string" # type: ignore
        with self.assertRaises(TypeError):
            function.shoulder = "string" # type: ignore
    def test_invalid_asignation_list(self):
        """Invalid Asignation with list"""
        function = LinearS()
        with self.assertRaises(TypeError):
            function.foot = ["string"] # type: ignore
        with self.assertRaises(TypeError):
            function.shoulder = ["string"] # type: ignore
    def test_invalid_asignation_tuple(self):
        """Invalid Asignation with tuple"""
        function = LinearS()
        with self.assertRaises(TypeError):
            function.foot = ("string") # type: ignore
        with self.assertRaises(TypeError):
            function.shoulder = ("string") # type: ignore
    def test_forward_return_shape(self):
        """Shape of forward pass of function"""
        function = LinearS(1, 1)
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
        foot = random.randint(1, 10)
        shoulder = random.randint(1, 10)
        function = LinearS(
            foot=foot,
            shoulder=shoulder
        )
        def equation(
                x: torch.Tensor, 
                foot: Union[int, float],
                shoulder: Union[int, float], 
            ) -> torch.Tensor:
            """Comparative LinearS equation"""
            return torch.minimum(torch.maximum((x - foot) / (shoulder - foot), torch.tensor(0)), torch.tensor(1))
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
                foot=foot,
                shoulder=shoulder,
            )
            y_2 = function(x)
            with self.subTest(shape=tensor_shape):
                self.assertTrue(torch.equal(y_1, y_2))
    def test_gradient_exists(self):
        """Backpropagation of the function"""
        function = LinearS(1, 1)
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
                self.assertIsNotNone(function.foot.grad)
                self.assertIsNotNone(function.shoulder.grad)