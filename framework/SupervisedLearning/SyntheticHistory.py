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
  Created on Jan 5, 2020

  @author: talbpaul
  Originally from ARMA.py, split for modularity
  Uses TimeSeriesAnalysis (TSA) algorithms to train then generate synthetic histories
"""
import collections
import numpy as np

from utils import InputData, xmlUtils, mathUtils
import TSA

from .SupervisedLearning import supervisedLearning

class SyntheticHistory(supervisedLearning):
  """
    Leverage TSA algorithms to train then generate synthetic signals.
  """
  # class attribute
  ## define the clusterable features for this ROM.
  # _clusterableFeatures = TODO # get from TSA
  @classmethod
  def getInputSpecifications(cls):
    """
      Establish input specs for this class.
      @ In, None
      @ Out, spec, InputData.ParameterInput, class for specifying input template
    """
    specs = InputData.parameterInputFactory('SyntheticHistory', strictMode=True,
        descr=r"""A ROM for characterizing and generating synthetic histories. This ROM makes use of
               a variety of TimeSeriesAnalysis (TSA) algorithms to characterize and generate new
               signals based on training signal sets. """)
    for typ in TSA.knownTypes():
      c = TSA.returnClass(typ, None) # TODO no message handler for second argument
      specs.addSub(c.getInputSpecification())
    return specs

  ### INHERITED METHODS ###
  def __init__(self, messageHandler, **kwargs):
    """
      A constructor that will appropriately intialize a supervised learning object
      @ In, messageHandler: a MessageHandler object in charge of raising errors,
                           and printing messages
      @ In, kwargs: an arbitrary dictionary of keywords and values
    """
    # general infrastructure
    supervisedLearning.__init__(self, messageHandler, **kwargs)
    self.printTag = 'SyntheticHistoryROM'
    self._dynamicHandling = True # This ROM is able to manage the time-series on its own.
    # training storage
    self.algoSettings = {}  # initialization settings for each algorithm
    self.trainedParams = {} # holds results of training each algorithm
    self.tsaAlgorithms = [] # list and order for tsa algorithms to use
    # data manipulation
    self.pivotParameterID = None      # string name for time-like pivot parameter
    self.pivotParameterValues = None  # In here we store the values of the pivot parameter (e.g. Time)

    inputSpecs = kwargs['paramInput']
    self.readInputSpecs(inputSpecs)

  def readInputSpecs(self, inp):
    """
      Reads in the inputs from the user
      @ In, inp, InputData.InputParams, input specifications
      @ Out, None
    """
    self.pivotParameterID = inp.findFirst('pivotParameter').value # TODO does a base class do this?
    tsaCounter = 0
    for sub in inp.subparts:
      if sub.name in TSA.knownTypes():
        tsaCounter += 1
        cls = TSA.returnClass(sub.name, self.messageHandler)
        algo = cls(name=f'{sub.name}{tsaCounter}') # TODO user names?
        self.algoSettings[algo] = algo.handleInput(sub)
        self.tsaAlgorithms.append(algo)
    if self.pivotParameterID not in self.target:
      self.raiseAnError(IOError, 'The pivotParameter must be included in the target space.')

  def __trainLocal__(self, featureVals, targetVals):
    """
      Perform training on input database stored in featureVals.
      @ In, featureVals, array, shape=[n_timeStep, n_dimensions], an array of input data # Not use for ARMA training
      @ In, targetVals, array, shape = [n_timeStep, n_dimensions], an array of time series data
      @ Out, None
    """
    self.raiseADebug('Training...')
    # TODO obtain pivot parameter
    pivotName = self.pivotParameterID
    pivotIndex = self.target.index(pivotName)
    # TODO assumption: only one training signal
    pivots = targetVals[0, :, pivotIndex]
    self.pivotParameterValues = pivots[:] # TODO any way to avoid storing these?
    residual = targetVals[:, :, :] # deep-ish copy, so we don't mod originals
    numAlgo = len(self.tsaAlgorithms)
    for a, algo in enumerate(self.tsaAlgorithms):
      settings = self.algoSettings[algo]
      targets = settings['target']
      indices = tuple(self.target.index(t) for t in targets)
      signal = residual[0, :, indices].T # using tuple "indices" transposes, so transpose back
      params = algo.characterize(signal, pivots, targets, settings)
      # store characteristics
      self.trainedParams[algo] = params
      # obtain residual; the part of the signal not characterized by this algo
      # workaround: skip the last one, since it's often the ARMA and the residual isn't known for
      #             the ARMA
      if a < numAlgo - 1:
        algoResidual = algo.getResidual(signal, params, pivots, settings)
        residual[0, :, indices] = algoResidual.T # transpose, again because of indices
      # TODO meta store signal, residual?

  def __evaluateLocal__(self, featureVals):
    """
      @ In, featureVals, float, a scalar feature value is passed as scaling factor
      @ Out, rlz, dict, realization dictionary of values for each target
    """
    pivots = self.pivotParameterValues
    result = np.zeros((self.pivotParameterValues.size, len(self.target) - 1)) # -1 is pivot
    for algo in self.tsaAlgorithms[::-1]:
      settings = self.algoSettings[algo]
      targets = settings['target']
      indices = tuple(self.target.index(t) for t in targets)
      params = self.trainedParams[algo]
      signal = algo.generate(params, pivots, settings)
      result[:, indices] += signal
    # RAVEN realization construction
    rlz = dict((target, result[:, t]) for t, target in enumerate(self.target) if target != self.pivotParameterID)
    rlz[self.pivotParameterID] = self.pivotParameterValues
    return rlz

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    pass # TODO

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Allows the SVE to put whatever it wants into an XML to print to file.
      Overload in subclasses.
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused (kept for compatability)
      @ In, skip, list, optional, unused (kept for compatability)
      @ Out, None
    """
    root = writeTo.getRoot()
    for algo in self.tsaAlgorithms:
      # only write trained information
      if algo not in self.trainedParams:
        continue
      algoNode = xmlUtils.newNode(algo.name)
      algo.writeXML(algoNode, self.trainedParams[algo])
      root.append(algoNode)

  ### Segmenting

  def getGlobalRomSegmentSettings(self, trainingDict, divisions):
    """
      Allows the ROM to perform some analysis before segmenting.
      Note this is called on the GLOBAL templateROM from the ROMcollection, NOT on the LOCAL subsegment ROMs!
      @ In, trainingDict, dict, data for training, full and unsegmented
      @ In, divisions, tuple, (division slice indices, unclustered spaces)
      @ Out, settings, object, arbitrary information about ROM clustering settings
      @ Out, trainingDict, dict, adjusted training data (possibly unchanged)
    """
    # nothing to do for now, but we'll hold this in case.
    return super().getGlobalRomSegmentSettings(trainingDict, divisions)

  ### Clustering
  def isClusterable(self):
    """
      Allows ROM to declare whether it has methods for clustring. Default is no.
      @ In, None
      @ Out, isClusterable, bool, if True then has clustering mechanics.
    """
    # clustering methods have been added
    return True

  def checkRequestedClusterFeatures(self, request):
    """
      Takes the user-requested features (sometimes "all") and interprets them for this ROM.
      @ In, request, dict(list), as from ROMColletion.Cluster._extrapolateRequestedClusterFeatures
      @ Out, interpreted, dict(list), interpreted features
    """
    # if no specific request, cluster on everything
    if request is None:
      return self._getClusterableFeatures()
    # if request given, iterate through and check them
    errMsg = []
    badAlgos = []    # keep a list of flagged bad algo names
    badFeatures = collections.defaultdict(list) # keep a record of flagged bad features (but ok algos)
    for algoName, feature in request.items():
      if algoName not in badFeatures and algoName not in TSA.knownTypes():
        errMsg.append(f'Unrecognized TSA algorithm while searching for cluster features: "{algoName}"! Expected one of: {TSA.knownTypes()}')
        badAlgos.append(algoName)
        continue
      algo = TSA.returnClass(algoName, self)
      if feature not in algo._features:
        badFeatures[algoName].append(feature)
    if badFeatures:
      for algoName, features in badFeatures.items():
        algo = TSA.returnClass(algoName, self)
        errMsg.append(f'Unrecognized clusterable features for TSA algorithm "{algoName}": {features}. Acceptable features are: {algo._features}')
    if errMsg:
      self.raiseAnError(IOError, 'The following errors occured while building clusterable features:\n' +
                        '\n  '.join(errMsg))
    return request

  def _getClusterableFeatures(self):
    """
      Provides a list of clusterable features.
      For this ROM, these are as "TSA_algorith|feature" such as "fourier|amplitude"
      @ In, None
      @ Out, features, dict(list(str)), clusterable features by algorithm
    """
    features = {}
    # check: is it possible tsaAlgorithms isn't populated by now?
    for algo in self.tsaAlgorithms:
      features[algo.name] = algo._features
    return features

  def getLocalRomClusterFeatures(self, featureTemplate, settings, request, picker=None):
    """
      Provides metrics aka features on which clustering compatibility can be measured.
      This is called on LOCAL subsegment ROMs, not on the GLOBAL template ROM
      @ In, featureTemplate, str, format for feature inclusion
      @ In, settings, dict, as per getGlobalRomSegmentSettings
      @ In, request, dict(list), requested features to cluster on (by feature set)
      @ In, picker, slice, indexer for segmenting data
      @ Out, features, dict, {target_metric: np.array(floats)} features to cluster on
    """
    features = {}
    for algo in self.tsaAlgorithms:
      if algo.name not in request:
        continue
      algoReq = request[algo.name] if request is not None else None
      algoFeatures = algo.getClusteringValues(featureTemplate, algoReq, self.trainedParams[algo])
      features.update(algoFeatures)
    return features

  def setLocalRomClusterFeatures(self, settings):
    """
      Forcibly set the parameters of this ROM based on those in "settings".
      Settings will have naming conventions as from getLocalRomClusterFeatures.
      @ In, settings, dict, parameters to set
    """
    byAlgo = collections.defaultdict(list)
    for feature, values in settings.items():
      target, algoName, ident = feature.split('|', maxsplit=2)
      byAlgo[algoName].append((target, ident, values))
    for algo in self.tsaAlgorithms:
      settings = byAlgo.get(algo.name, None)
      if settings:
        params = algo.setClusteringValues(settings, self.trainedParams[algo])
        self.trainedParams[algo] = params

  def findAlgoByName(self, name):
    """
      Find the corresponding algorithm by name
      @ In, name, str, name of algorithm
      @ Out, algo, TSA.TimeSeriesAnalyzer, algorithm
    """
    for algo in self.tsaAlgorithms:
      if algo.name == name:
        return algo
    return None


  ### ESSENTIALLY UNUSED ###
  def _localNormalizeData(self,values,names,feat):
    """
      Overwrites default normalization procedure, since we do not desire normalization in this implementation.
      @ In, values, unused
      @ In, names, unused
      @ In, feat, feature to normalize
      @ Out, None
    """
    self.muAndSigmaFeatures[feat] = (0.0,1.0)

  def __confidenceLocal__(self,featureVals):
    """
      This method is currently not needed for ARMA
    """
    pass

  def __resetLocal__(self,featureVals):
    """
      After this method the ROM should be described only by the initial parameter settings
      Currently not implemented for ARMA
    """
    pass

  def __returnInitialParametersLocal__(self):
    """
      there are no possible default parameters to report
    """
    localInitParam = {}
    return localInitParam

  def __returnCurrentSettingLocal__(self):
    """
      override this method to pass the set of parameters of the ROM that can change during simulation
      Currently not implemented for ARMA
    """
    pass
