# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  TODO

  Reworked 2020-01
  @author: talbpaul
"""
import abc

from utils import utils, InputData, InputTypes

class GradientApproximater(utils.metaclass_insert(abc.ABCMeta, object)):
  """
    GradientApproximators use provided information to both select points
    required to estimate gradients as well as calculate the estimates.
  """
  ##########################
  # Initialization Methods #
  ##########################
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for specifying input of cls.
    """
    specs = InputData.parameterInputFactory(cls.__name__, ordered=False, strictMode=True)
    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    ## Instance Variable Initialization
    # public
    # _protected
    self._optVars = None
    self._proximity = None
    self.N = None
    # __private
    # additional methods

  def handleInput(self, specs):
    """
      Read input specs
      @ In, paramInput, InputData.ParameterInput, parameter specs interpreted
      @ Out, None
    """
    pass

  def initialize(self, optVars, proximity):
    """
      After construction, finishes initialization of this approximator.
      @ In, optVars, list(str), list of optimization variable names
      @ In, proximity, float, percentage of step size away that neighbor samples should be taken
      @ Out, None
    """
    self._optVars = optVars
    self._proximity = proximity
    self.N = len(self._optVars)

  ###############
  # Run Methods #
  ###############
  @abc.abstractmethod
  def chooseEvaluationPoints(self, opt, stepSize):
    """
      Determines new point(s) needed to evaluate gradient
      @ In, opt, dict, current opt point (normalized)
      @ In, stepSize, float, distance from opt point to sample neighbors
      @ Out, evalPoints, list(dict), list of points that need sampling
      @ Out, evalInfo, list(dict), identifying information about points
    """

  @abc.abstractmethod
  def numGradPoints(self):
    """
      Returns the number of grad points required for the method
    """

  @abc.abstractmethod
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

  def needDenormalized(self):
    """
      Determines if this algorithm needs denormalized input spaces
      @ In, None
      @ Out, needDenormalized, bool, True if normalizing should NOT be performed
    """
    return False

  def updateSolutionExport(self, grads, gradInfos):
    """
      Prints information to the solution export.
      @ In, grads, list, list of gradient magnitudes and versors
      @ In, gradInfos, list, list of identifying information for each grad entry
      @ Out, info, dict, realization of data to go in the solutionExport object
    """
    # overload in inheriting classes at will
    return {}
  ###################
  # Utility Methods #
  ###################


