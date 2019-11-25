from unittest import TestCase

import torch
import numpy as np

from models import Pi
from retrictions import get_F


class TestRestrictions(TestCase):

    def setUp(self):
        self.model = Pi(poly_order=3)
        self.model.w = torch.nn.Parameter(torch.tensor([0.5, 0.2, 0.6]))

    def test_f(self):
        """The polynomial: f(x) = x³ - 6x² + 11x - 6 has the following roots:
        r₁ = 1, r₂ = 2 and r₃=3. When the neural network has as weights:
        w₁ = 0.5, w₂ = 0.2 and w₃=0.6 the restriction vectors must be:
        F₁ = [-1, 0.8, -0.12], F₂ = [-1, 1.1, -0.3], F₃ = [-1, 0.7, -0.1]"""
        F1 = get_F(W=self.model.w, i=0)
        self.assertTrue(np.allclose(F1, np.array([[-1, 0.8, -0.12]])))
        F2 = get_F(W=self.model.w, i=1)
        self.assertTrue(np.allclose(F2, np.array([[-1, 1.1, -0.3]])))
        F3 = get_F(W=self.model.w, i=2)
        self.assertTrue(np.allclose(F3, np.array([[-1, 0.7, -0.1]])))
