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



"Author:--"

"Strictly as per the guidelines given in the"
#https://www.jhuapl.edu/SPSA/PDF-SPSA/Matlab-Auto_gain_sel-b-w.pdf#

#https://www.jhuapl.edu/SPSA/PDF-SPSA/Matlab-Auto_gain_sel-b-w.pdf#


class SPSA(GradientApproximater):
  ### Some global parameters ###
  
  counter = 0
  ak = None
  
  a = 0.001
  A = 0.5
  alpha = 0.602
  ck = None
  c = None
  gamma = 0.101
  def chooseEvaluationPoints(self, opt, stepSize):
    self.counter+=1
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """
    dh = self._proximity * stepSize 
    self.c = dh
    self.ak = self.a/np.power(self.counter+self.A, self.alpha)
    self.ck=self.c/np.power(self.counter,self.gamma)
    perturb = randomUtils.randPointsOnHypersphere(1)
    
    

    evalPoints = []
    evalInfo = []
    
   


    for o, optVar in enumerate(self._optVars):
      

   

 
      optValue = opt[optVar]
      new = copy.deepcopy(opt)
      new2 = copy.deepcopy(opt)


 

      
      
      delta = self.ck*perturb
      
    
     
      new[optVar] = optValue + self.ck*perturb
      new2[optVar] = optValue - self.ck*perturb
   

      evalPoints.append(new)
      evalPoints.append(new2)
      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
                       'delta': delta})

      evalInfo.append({'type': 'grad',
                       'optVar': optVar,
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


    gradient = {}

   
    

    for i in range(0,len(grads),2):

     
      

      info = infos[i]
      delta = info['delta']
      activeVar = info['optVar']

      

      
      lossDiff = (grads[i][objVar]-grads[i+1][objVar])

      grad = lossDiff/(2*delta)

     


      
  
      gradient[activeVar] = grad
  
  
    magnitude, direction, foundInf = mathUtils.calculateMagnitudeAndVersor(list(gradient.values()))
    direction = dict((var, float(direction[v])) for v, var in enumerate(gradient.keys()))
    return magnitude, direction, foundInf

  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """

    return self.N*2
  






    
  
  



