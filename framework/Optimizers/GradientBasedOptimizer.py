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
  This module contains the Gradient Based Optimization strategy

  Created on June 16, 2016
  @author: chenj
"""
#for future compatibility with Python 3--------------------------------------------------------------
from __future__ import division, print_function, unicode_literals, absolute_import
import warnings
warnings.simplefilter('default',DeprecationWarning)
#if not 'xrange' in dir(__builtins__): xrange = range
#End compatibility block for Python 3----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
import sys
import os
import copy
import abc
import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .Optimizer import Optimizer
from Assembler import Assembler
from utils import utils,cached_ndarray
#Internal Modules End--------------------------------------------------------------------------------

class GradientBasedOptimizer(Optimizer):
  """
    This is the base class for gradient based optimizer. The following methods need to be overridden by all derived class
    self.localLocalInputAndChecks(self, xmlNode)
    self.localLocalInitialize(self, solutionExport = None)
    self.localLocalGenerateInput(self,model,oldInput)
    self.localEvaluateGradient(self, optVarsValues, gradient = None)
  """
  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    Optimizer.__init__(self)
    self.constraintHandlingPara     = {}              # Dict containing parameters for parameters related to constraints handling
    self.gradientNormTolerance      = 1.e-3           # tolerance on the L2 norm of the gradient
    self.gradDict                   = {}              # Dict containing information for gradient related operations
    self.gradDict['numIterForAve']  = 1               # Number of iterations for gradient estimation averaging
    self.gradDict['pertNeeded']     = 1               # Number of perturbation needed to evaluate gradient
    self.gradDict['pertPoints']     = {}              # Dict containing normalized inputs sent to model for gradient evaluation
    self.readyVarsUpdate            = {}              # Bool variable indicating the finish of gradient evaluation and the ready to update decision variables
    self.counter['perturbation']    = {}              # Counter for the perturbation performed.
    self.counter['gradientHistory'] = {}              # In this dict we store the gradient value (versor) for current and previous iterations {'trajectoryID':[{},{}]}
    self.counter['gradNormHistory'] = {}              # In this dict we store the gradient norm for current and previous iterations {'trajectoryID':[float,float]}
    self.counter['varsUpdate'     ] = {}
    self.counter['solutionUpdate' ] = {}
    self.convergeTraj = {}

  def localInputAndChecks(self, xmlNode):
    """
      Method to read the portion of the xml input that belongs to all gradient based optimizer only
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    convergence = xmlNode.find("convergence")
    if convergence is not None:
      gradientThreshold = convergence.find("gradientThreshold")
      try:
        self.gradientNormTolerance = float(gradientThreshold.text) if gradientThreshold is not None else self.gradientNormTolerance
      except ValueError:
        self.raiseAnError(ValueError, 'Not able to convert <gradientThreshold> into a float.')

  def localInitialize(self,solutionExport=None):
    """
      Method to initialize settings that belongs to all gradient based optimizer
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    self.gradDict['numIterForAve'] = int(self.paramDict.get('numGradAvgIterations', 1))
    for traj in self.optTraj:
      self.gradDict['pertPoints'][traj]      = {}
      self.counter['perturbation'][traj]     = 0
      self.counter['varsUpdate'][traj]       = 0
      self.counter['solutionUpdate'][traj]   = 0
      self.counter['gradientHistory'][traj]  = [{},{}]
      self.counter['gradNormHistory'][traj]  = [0.0,0.0]
      self.optVarsHist[traj]                 = {}
      self.readyVarsUpdate[traj]             = False
      self.convergeTraj[traj]                = False
    for traj in self.optTraj:
      self.gradDict['pertPoints'][traj] = {}
    # end job runnable equal to number of trajectory
    self._endJobRunnable = len(self.optTraj)
    #specializing the self.localLocalInitialize()
    if solutionExport != None:
      self.localLocalInitialize(solutionExport=solutionExport)
    else:
      self.localLocalInitialize()

  @abc.abstractmethod
  def localLocalInitialize(self, solutionExport = None):
    """
      Method to initialize local settings.
      @ In, solutionExport, DataObject, optional, a PointSet to hold the solution
      @ Out, None
    """
    pass

  def localStillReady(self,ready, convergence = False): #,lastOutput=None
    """
      Determines if optimizer is ready to provide another input.  If not, and if jobHandler is finished, this will end sampling.
      @ In, ready, bool, variable indicating whether the caller is prepared for another input.
      @ In, convergence, bool, optional, variable indicating whether the convergence criteria has been met.
      @ Out, ready, bool, boolean variable indicating whether the caller is prepared for another input.
    """
    #let this be handled at the local subclass level for now
    return ready

  def _checkModelFinish(self, traj, updateKey, evalID):
    """
      Determines if the Model has finished running an input and returned the output
      @ In, traj, int, traj on which the input is being checked
      @ In, updateKey, int, the id of variable update on which the input is being checked
      @ In, evalID, int or string, indicating the id of the perturbation (int) or its a variable update (string 'v')
      @ Out, _checkModelFinish, tuple(bool, int), (1,realization dictionary),
            (indicating whether the Model has finished the evaluation over input identified by traj+updateKey+evalID, the index of the location of the input in dataobject)
    """
    if self.mdlEvalHist.isItEmpty():
      return (False,-1)

    prefix = self.mdlEvalHist.getMetadata('prefix')
    for index, pr in enumerate(prefix):
      pr = pr.split(utils.returnIdSeparator())[-1].split('_')
      # use 'prefix' to locate the input sent out. The format is: trajID + iterID + (v for variable update; otherwise id for gradient evaluation) + global ID
      if pr[0] == str(traj) and pr[1] == str(updateKey) and pr[2] == str(evalID):
        return (True, index)
    return (False, -1)

  def localGenerateInput(self,model,oldInput):
    """
      Method to generate input for model to run
      @ In, model, model instance, it is the instance of a RAVEN model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc. etc)
      @ Out, None
    """
    #self.readyVarsUpdate = {traj:False for traj in self.optTrajLive}
    pass

  def evaluateGradient(self, optVarsValues, traj):
    """
      Method to evaluate gradient based on perturbed points and model evaluations.
      @ In, optVarsValues, dict, dictionary containing NORMALIZED perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ In, traj, int, the trajectory id
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    gradArray = {}
    for var in self.optVars:
      gradArray[var] = np.ndarray((0,0)) #why are we initializing to this?
    # Evaluate gradient at each point
    for pertIndex in optVarsValues.keys():
      tempDictPerturbed = self.denormalizeData(optVarsValues[pertIndex])
      lossValue = copy.copy(self.lossFunctionEval(tempDictPerturbed)) #needs to be denormalized?
      lossDiff = lossValue[0] - lossValue[1]
      print('DEBUGG loss val:',lossValue)
      for var in self.optVars:
        if optVarsValues[pertIndex][var][0] != optVarsValues[pertIndex][var][1]:
          # even if the feature space is normalized, we compute the gradient in its space (transformed or not)
          gradArray[var] = np.append(gradArray[var], lossDiff/(optVarsValues[pertIndex][var][0]-optVarsValues[pertIndex][var][1])*1.0)
    gradient = {}
    for var in self.optVars:
      gradient[var] = gradArray[var].mean()
    print('DEBUGG gradient 1:',gradient)
    gradient = self.localEvaluateGradient(optVarsValues, gradient)
    print('DEBUGG gradient 2:',gradient)
    gradientNorm =  np.linalg.norm(gradient.values())
    if gradientNorm > 0.0:
      for var in gradient.keys():
        gradient[var] = gradient[var]/gradientNorm
    print('DEBUGG gradient N:',gradientNorm)
    self.counter['gradientHistory'][traj][1] = self.counter['gradientHistory'][traj][0]
    self.counter['gradientHistory'][traj][0] = gradient
    self.counter['gradNormHistory'][traj][1] = self.counter['gradNormHistory'][traj][0]
    self.counter['gradNormHistory'][traj][0] = gradientNorm
    return gradient

  def _createEvaluationIdentifier(self,trajID,iterID,evalType):
    """
      Create evaluation identifier
      @ In, trajID, integer, trajectory identifier
      @ In, iterID, integer, iteration number (identifier)
      @ In, evalType, integer or string, evaluation type (v for variable update; otherwise id for gradient evaluation)
      @ Out, identifier, string, the evaluation identifier
    """
    identifier = str(trajID) + '_' + str(iterID) + '_' + str(evalType)
    return identifier

  def localEvaluateGradient(self, optVarsValues, gradient = None):
    """
      Local method to evaluate gradient.
      @ In, optVarsValues, dict, dictionary containing perturbed points.
                                 optVarsValues should have the form {pertIndex: {varName: [varValue1 varValue2]}}
                                 Therefore, each optVarsValues[pertIndex] should return a dict of variable values
                                 that is sufficient for gradient evaluation for at least one variable
                                 (depending on specific optimization algorithm)
      @ In, gradient, dict, optional, dictionary containing gradient estimation by the caller.
                                      gradient should have the form {varName: gradEstimation}
      @ Out, gradient, dict, dictionary containing gradient estimation. gradient should have the form {varName: gradEstimation}
    """
    return gradient

  def checkConvergence(self):
    """
      Method to check whether the convergence criteria has been met.
      @ In, None
      @ Out, convergence, list, list of bool variable indicating whether the convergence criteria has been met for each trajectory.
    """
    convergence = True
    for traj in self.optTraj:
      if self.convergeTraj[traj] == False:
        convergence = False
        break
    return convergence

  def _updateConvergenceVector(self, traj, varsUpdate, currentLossValue):
    """
      Local method to update convergence vector.
      @ In, traj, int, identifier of the trajector to update
      @ In, varsUpdate, int, current variables update iteration number
      @ In, currentLossValue, float, current loss function value
      @ Out, None
    """
    if not self.convergeTraj[traj]:
      #if varsUpdate >= 1: multilevel can't rely on the varsUpdate number
      if len(self.counter['gradientHistory'][traj][1]) > 0:
        print('DEBUGG ... updating convergence vector ...')
        sizeArray = 1
        if self.gradDict['numIterForAve'] > 1:
          sizeArray+=self.gradDict['numIterForAve']
        objectiveOutputs = np.zeros(sizeArray)
        objectiveOutputs[0] = self.getLossFunctionGivenId(self._createEvaluationIdentifier(traj,varsUpdate-1,'v'))
        if sizeArray > 1:
          for i in range(sizeArray-1):
            identifier = (i+1)*2
            objectiveOutputs[i+1] = self.getLossFunctionGivenId(self._createEvaluationIdentifier(traj,varsUpdate-1,identifier))
        if any(np.isnan(objectiveOutputs)):
          self.raiseAnError(Exception,"the objective function evaluation for trajectory " +str(traj)+ "and iteration "+str(varsUpdate-1)+" has not been found!")
        oldVal = objectiveOutputs.mean()
        gradNorm = self.counter['gradNormHistory'][traj][0]
        varK = self.optVarsHist[traj][self.counter['varsUpdate'][traj]]
        varK = self.denormalizeData(varK)
        absDifference = abs(currentLossValue-oldVal)
        relativeDifference = abs(absDifference/oldVal)
        # checks
        sameCoordinateCheck = set(self.optVarsHist[traj][varsUpdate].items()) == set(self.optVarsHist[traj][varsUpdate-1].items())
        gradientNormCheck = gradNorm <= self.gradientNormTolerance
        absoluteTolCheck = absDifference <= self.absConvergenceTol
        relativeTolCheck = relativeDifference <= self.relConvergenceTol
        self.raiseAMessage("Trajectory: "+"%8i"% (traj)+      " | Iteration    : "+"%8i"% (varsUpdate)+ " | Loss function: "+"%8.2E"% (currentLossValue)+" |")
        self.raiseAMessage("Grad Norm : "+"%8.2E"% (gradNorm)+" | Relative Diff: "+"%8.2E"% (relativeDifference)+" | Abs Diff     : "+"%8.2E"% (absDifference)+" |")
        self.raiseAMessage("Variables :" +str(varK))

        if sameCoordinateCheck or gradientNormCheck or absoluteTolCheck or relativeTolCheck:
          if sameCoordinateCheck:
            reason="same-coordinate"
          if gradientNormCheck:
            reason="gradient-norm  "
          if absoluteTolCheck:
            reason="absolute-tolerance"
          if relativeTolCheck:
            reason="relative-tolerance"
          self.raiseAMessage("Trajectory: "+"%8i"% (traj) +"   converged. Reason: "+reason)
          self.raiseAMessage("Grad Norm : "+"%8.2E"% (gradNorm)+" | Relative Diff: "+"%8.2E"% (relativeDifference)+" | Abs Diff     : "+"%8.2E"% (absDifference)+" |")
          self.convergeTraj[traj] = True
          self.removeConvergedTrajectory(traj)
          #for trajInd, tr in enumerate(self.optTrajLive):
          #  if tr == traj:
          #    self.optTrajLive.pop(trajInd)
          #    break

  def _removeRedundantTraj(self, trajToRemove, currentInput):
    """
      Local method to remove multiple trajectory
      @ In, trajToRemove, int, identifier of the trajector to remove
      @ In, currentInput, dict, the last variable on trajectory traj
      @ Out, None
    """
    removeFlag = False
    for traj in self.optTraj:
      if traj != trajToRemove:
        for updateKey in self.optVarsHist[traj].keys():
          inp = copy.deepcopy(self.optVarsHist[traj][updateKey]) #FIXME deepcopy needed?
          removeLocalFlag = True
          for var in self.optVars:
            if abs(inp[var] - currentInput[var]) > self.thresholdTrajRemoval:
              removeLocalFlag = False
              break
          if removeLocalFlag:
            removeFlag = True
            break
        if removeFlag:
          break

    if removeFlag:
      for trajInd, tr in enumerate(self.optTrajLive):
        if tr == trajToRemove:
          self.optTrajLive.pop(trajInd)
          break

  def localCheckConstraint(self, optVars, satisfaction = True):
    """
      Local method to check whether a set of decision variables satisfy the constraint or not
      @ In, optVars, dict, dictionary containing the value of decision variables to be checked, in form of {varName: varValue}
      @ In, satisfaction, bool, optional, variable indicating how the caller determines the constraint satisfaction at the point optVars
      @ Out, satisfaction, bool, variable indicating the satisfaction of constraints at the point optVars
    """
    return satisfaction

  def _getJobsByID(self,traj):
    """
      Overwrite only if you need something special at the end of each run....
      This function is used by optimizers that need to collect information from the just ended run
      @ In, traj, int, ID of the trajectory for whom we collect jobs
      @ Out, solutionExportUpdatedFlag, bool, True if the solutionExport needs updating
      @ Out, solutionIndeces, list(int), location of updates within the full targetEvaluation data object
    """
    solutionExportUpdatedFlag, index = self._checkModelFinish(traj, self.counter['solutionUpdate'][traj], 'v')
    solutionUpdateList = [solutionExportUpdatedFlag]
    solutionIndeces = [index]
    #sizeArray = 1
    if self.gradDict['numIterForAve'] > 1:
      #sizeArray+=self.gradDict['numIterForAve']
      for i in range(sizeArray-1):
        identifier = (i+1)*2
        solutionExportUpdatedFlag, index = self._checkModelFinish(traj, self.counter['solutionUpdate'][traj], str(identifier))
        solutionUpdateList.append(solutionExportUpdatedFlag)
        solutionIndeces.append(index)
      solutionExportUpdatedFlag = all(solutionUpdateList)
    return solutionExportUpdatedFlag,solutionIndeces

    #### OLD ####
    #  # check convergence FIXME why do we do this here??? This is a "amIready" kind of task!
    #  self._updateConvergenceVector(traj, self.counter['solutionUpdate'][traj], currentObjectiveValue)
