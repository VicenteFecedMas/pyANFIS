"""This will test that all the functionalities in the Bell function to check if they work as expected"""
import random
from typing import Union
import unittest
import torch

from pyanfis.functions import Bell

class TestBell(unittest.TestCase):
    """Bell function test cases"""
    def test_initialization_empty(self):
        """Initialisation with empty variables"""
        function = Bell()
        self.assertTrue(torch.equal(function.center, torch.tensor([])))
        self.assertTrue(torch.equal(function.shape, torch.tensor([])))
        self.assertTrue(torch.equal(function.width, torch.tensor([])))
    def test_initialization_center(self):
        """Initialising center only"""
        function = Bell(center=1)
        self.assertEqual(function.center, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertTrue(torch.equal(function.shape, torch.tensor([])))
        self.assertTrue(torch.equal(function.width, torch.tensor([])))
    def test_initialization_shape(self):
        """Initialising shape only"""
        function = Bell(shape=1)
        self.assertTrue(torch.equal(function.center, torch.tensor([])))
        self.assertEqual(function.shape, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertTrue(torch.equal(function.width, torch.tensor([])))
    def test_initialization_width(self):
        """Initialising width only"""
        function = Bell(width=1)
        self.assertTrue(torch.equal(function.center, torch.tensor([])))
        self.assertTrue(torch.equal(function.shape, torch.tensor([])))
        self.assertEqual(function.width, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_initialization_all_parameters(self):
        """Initialising all parameters"""
        function = Bell(1, 1, 1)
        self.assertEqual(function.center, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertEqual(function.shape, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        self.assertEqual(function.width, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_invalid_initialisation_string(self):
        """Invalid Initialisations with strings"""
        self.assertRaises(TypeError, Bell, {"center":"string"})
        self.assertRaises(TypeError, Bell, {"shape":"string"})
        self.assertRaises(TypeError, Bell, {"width":"string"})
    def test_invalid_initialisation_list(self):
        """Invalid Initialisations with lists"""
        self.assertRaises(TypeError, Bell, {"center":[1]})
        self.assertRaises(TypeError, Bell, {"shape":[1]})
        self.assertRaises(TypeError, Bell, {"width":[1]})
    def test_invalid_initialisation_tuple(self):
        """Invalid Initialisations with tuples"""
        self.assertRaises(TypeError, Bell, {"center":(1)})
        self.assertRaises(TypeError, Bell, {"shape":(1)})
        self.assertRaises(TypeError, Bell, {"width":(1)})
    def test_valid_asignation_int(self):
        """Valid Asignation with int"""
        function = Bell()
        function.center = 1 # type: ignore
        self.assertEqual(function.center, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.width = 1 # type: ignore
        self.assertEqual(function.width, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.shape = 1 # type: ignore
        self.assertEqual(function.shape, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_valid_asignation_float(self):
        """Valid Asignation with float"""
        function = Bell()
        function.center = 1.0 # type: ignore
        self.assertEqual(function.center, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.width = 1.0 # type: ignore
        self.assertEqual(function.width, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.shape = 1.0 # type: ignore
        self.assertEqual(function.shape, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_valid_asignation_tensor(self):
        """Valid Asignation with tensor"""
        function = Bell()
        function.center = torch.tensor(1.0)
        self.assertEqual(function.center, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.width = torch.tensor(1.0)
        self.assertEqual(function.width, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
        function.shape = torch.tensor(1.0)
        self.assertEqual(function.shape, torch.nn.Parameter(torch.tensor(1).float(), requires_grad=True))
    def test_invalid_asignation_string(self):
        """Invalid Asignation with string"""
        function = Bell()
        with self.assertRaises(TypeError):
            function.center = "string" # type: ignore
        with self.assertRaises(TypeError):
            function.width = "string" # type: ignore
        with self.assertRaises(TypeError):
            function.shape = "string" # type: ignore
    def test_invalid_asignation_list(self):
        """Invalid Asignation with list"""
        function = Bell()
        with self.assertRaises(TypeError):
            function.center = ["string"] # type: ignore
        with self.assertRaises(TypeError):
            function.width = ["string"] # type: ignore
        with self.assertRaises(TypeError):
            function.shape = ["string"] # type: ignore
    def test_invalid_asignation_tuple(self):
        """Invalid Asignation with tuple"""
        function = Bell()
        with self.assertRaises(TypeError):
            function.center = ("string") # type: ignore
        with self.assertRaises(TypeError):
            function.width = ("string") # type: ignore
        with self.assertRaises(TypeError):
            function.shape = ("string") # type: ignore
    def test_forward_return_shape(self):
        """Shape of forward pass of function"""
        function = Bell(1, 1, 1)
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
        width = random.randint(1, 10)
        shape = random.randint(1, 10)
        center = random.randint(1, 10)
        function = Bell(
            width=width,
            shape=shape,
            center=center
        )
        def equation(
                x: torch.Tensor, 
                center: Union[int, float],
                width: Union[int, float], 
                shape: Union[int, float]
            ) -> torch.Tensor:
            """Comparative Bell equation"""
            return 1 / ((torch.abs((x - torch.tensor(center)) / torch.tensor(width)) ** (2*torch.tensor(shape))) + 1)
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
                center=center,
                width=width,
                shape=shape
            )
            y_2 = function(x)
            with self.subTest(shape=tensor_shape):
                self.assertTrue(torch.equal(y_1, y_2))
    def test_gradient_exists(self):
        """Backpropagation of the function"""
        function = Bell(1, 1, 1)
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
                self.assertIsNotNone(function.center.grad)
                self.assertIsNotNone(function.width.grad)
                self.assertIsNotNone(function.shape.grad)