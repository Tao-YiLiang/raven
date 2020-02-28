import abc
import copy
import pandas as pd
import numpy as np
import os
import sys
import functools
from .GradientApproximater import GradientApproximater

from utils import InputData, InputTypes, randomUtils, mathUtils

from utils import InputData, InputTypes, randomUtils, mathUtils

"Author: gairabhi"

class SPSA(GradientApproximater):

  def chooseEvaluationPoints(self, opt, stepSize):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """
    dh = self._proximity * stepSize
    perturb = randomUtils.randPointsOnHypersphere(self.N)
    delta = {}
    new = {}
    for i,var in enumerate(opt.keys()):
      if var != 'ans':
        delta[var] = perturb[i]*dh
        
        new[var] = opt[var]+delta[var]

    evalPoints = []
    evalInfo = []  

    evalPoints.append(new)
  
    
    evalInfo.append({'type': 'grad',
                       
                       'delta': delta})


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

    delta = infos[0]['delta']

    gradient = {}

    lossDiff = np.atleast_1d(mathUtils.diffWithInfinites(grads[0][objVar],opt[objVar]))
      
    for var in grads[0].keys():
      if var != objVar:
        gradient [var] = lossDiff/(delta[var])
   
    
  
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))
 
    return magnitude, direction, foundInf

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """
    return self.N/self.N
  






    
  
  



