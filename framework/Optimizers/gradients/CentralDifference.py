import abc
import copy
import pandas as pd
import numpy as np
import os
import sys
from utils import InputData, InputTypes, randomUtils, mathUtils

from utils import InputData, InputTypes, randomUtils, mathUtils

from .GradientApproximater import GradientApproximater

"Author:--gairabhi"

class CentralDifference(GradientApproximater):
  """
    Enables gradient estimation via central differencing
  """
  def chooseEvaluationPoints(self, opt, stepSize):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """
    dh = self._proximity * stepSize
    evalPoints = []
    evalInfo = []
    # submit a positive and negative side of the opt point for each dimension
    for _, optVar in enumerate(self._optVars):
      optValue = opt[optVar]
      neg = copy.deepcopy(opt)
      pos = copy.deepcopy(opt)
      delta = dh
      neg[optVar] = optValue - delta
      pos[optVar] = optValue + delta

      evalPoints.append(neg)
      evalInfo.append({'type': 'grad',
                      'optVar': optVar,
                      'delta': delta,
                      'side': 'negative'})

      evalPoints.append(pos)
      evalInfo.append({'type': 'grad',
                      'optVar': optVar,
                      'delta': delta,
                      'side': 'positive'})
    return evalPoints, evalInfo

  def evaluate(self, opt, grads, infos, objVar):
    """
      Approximates gradient based on evaluated points.
      @ In, opt, dict, current opt point (normalized)
      @ In, grads, list(dict), evaluated neighbor points
      @ In, infos, list(dict), info about evaluated neighbor points
      @ In, objVar, string, objective variable
      @ Out, magnitude, float, magnitude of gradient
      @ Out, direction, dict, versor (unit vector) for gradient direction
    """
    gradient = {}
    delta = {}
    for g, pt in enumerate(grads):
      info = infos[g]
      activeVar = info['optVar']
      delta[activeVar] = info['delta']
    for i, var in enumerate(self._optVars):
      # TODO fix this for parallel-safe indexing (using 'optVar' and 'side' from infos)
      gradient[var] = (-3 * grads[2*i+1][objVar] + 4 * opt[objVar] - grads[2*i][objVar]) / (2 * delta[var])
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))
    return magnitude, direction, foundInf

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """
    return self.N*2





