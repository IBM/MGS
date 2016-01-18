// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef TissueFunctor_H
#define TissueFunctor_H

#include "MaxComputeOrder.h"
#include "Lens.h"
#include "CG_TissueFunctorBase.h"
#include "LensContext.h"
#include "DataItemArrayDataItem.h"
#include "LensConnector.h"
#include "GranuleConnector.h"
#include "NoConnectConnector.h"
#include "rndm.h"
#include "VecPrim.h"

#ifdef HAVE_MPI
#include "Capsule.h"
#include "TissueContext.h"
#include "SegmentDescriptor.h"
#include "Params.h"
#endif

#include <memory>
#include <vector>
#include <utility>
#include <map>

class NDPairList;
class NodeDescriptor;
class GridLayerDescriptor;
class Simulation;
class Connector;
class StructDataItem;
class ComputeBranch;
class CG_CompartmentDimension;
class NodeAccessor;

class TissueFunctor : public CG_TissueFunctorBase
{
  friend class TissueLayoutFunctor;
  friend class TissueNodeInitFunctor;
  friend class TissueConnectorFunctor;
  friend class TissueProbeFunctor;

  public:
  void userInitialize(LensContext* CG_c, String& commandLineArgs1,
                      String& commandLineArgs2, String& compartmentParamFile,
                      String& channelParamFile, String& synapseParamFile,
                      Functor*& layoutFunctor, Functor*& nodeInitFunctor,
                      Functor*& connectorFunctor, Functor*& probeFunctor);
  std::auto_ptr<Functor> userExecute(LensContext* CG_c, String& tissueElement,
                                     NDPairList*& params);

  TissueFunctor();
  TissueFunctor(TissueFunctor const&);
  virtual ~TissueFunctor();
  virtual void duplicate(std::auto_ptr<TissueFunctor>& dup) const;
  virtual void duplicate(std::auto_ptr<Functor>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_TissueFunctorBase>& dup) const;

  private:
  int getCptIndex(Capsule*);
  dyn_var_t getFractionCapsuleVolumeFromPre(ComputeBranch* branch);
  dyn_var_t getFractionCapsuleVolumeFromPost(ComputeBranch* branch);
  int getNumCompartments(ComputeBranch* branch,
                         bool& isDistalEndSeeImplicitBranchingPoint);
  int getNumCompartments(ComputeBranch* branch);
  // perform neuron generating from a given set of parameters
  void neuroGen(Params* params, LensContext* CG_c);
  // perform neuron development
  void neuroDev(Params* params, LensContext* CG_c);
  // perform touch detection from the list of neurons (.swc files) and criteria
  // for
  // touch detection
  void touchDetect(Params* params, LensContext* CG_c);
  //
  int compartmentalize(LensContext* lc, NDPairList* params,
                       std::string& nodeCategory, std::string& nodeType,
                       int nodeIndex, int densityIndex);
  //
  std::vector<DataItem*> const* extractCompartmentalization(NDPairList* params);
  // get StructDataItem
  // from a given point (represented by coordinate  (a pointer to double which
  // is expected to be 3-element array)+ radius + distance to soma)
  StructDataItem* getDimension(LensContext* lc, double* cds, dyn_var_t radius,
                               dyn_var_t dist2soma, dyn_var_t surface_area,
															 dyn_var_t volume);
  StructDataItem* getDimension(LensContext* lc, double* cds1, double* cds2,
                               dyn_var_t radius, dyn_var_t dist2soma,
                               dyn_var_t surface_area,
															 dyn_var_t volume);
  // get the list of name of nodes (passed in GSL via nodekind)
  // from a list of layers represented as NDPairList*
  void getNodekind(const NDPairList* layerNdpl,
                   std::vector<std::string>& nodekind);
  // perform the connection between 2 set of nodetypes from 2 layers
  void connect(Simulation* sim, Connector* connector, NodeDescriptor* from,
               NodeDescriptor* to, NDPairList& ndpl);

  ShallowArray<int> doLayout(LensContext* lc);
  void doNodeInit(LensContext* lc);
  void doConnector(LensContext* lc);
  void doProbe(LensContext* lc, std::auto_ptr<NodeSet>& rval);
  ComputeBranch* findBranch(int nodeIndex, int densityIndex,
                            std::string const& cptVariableType);
  std::vector<int>& findBranchIndices(ComputeBranch*,
                                      std::string const& cptVariableType);
  Capsule* findJunction(int nodeIndex, int densityIndex,
                        std::string const& cptVariableType);
  std::vector<int>& findJunctionIndices(Capsule*,
                                        std::string const& cptVariableType);
  std::vector<int>& findForwardSolvePointIndices(ComputeBranch*,
                                                 std::string& nodeType);
  std::vector<int>& findBackwardSolvePointIndices(ComputeBranch*,
                                                  std::string& nodeType);

  void getModelParams(Params::ModelType modelType, NDPairList& paramsLocal,
                      std::string& nodeType, key_size_t key);
  // void getCompartmentParams(NDPairList& paramsLocal, std::string& nodeType,
  // key_size_t key);
  bool isChannelTarget(key_size_t key, std::string nodeType);
  void getElectricalSynapseProbabilities(std::vector<double>& probabilities,
                                         TouchVector::TouchIterator& titer,
                                         int direction, std::string nodeType);
  // void getSpineBranchProbabilities(std::vector<double>& probabilities,
  // TouchVector::TouchIterator & titer, int direction, std::string nodeType);
  void getChemicalSynapseProbabilities(std::vector<double>& probabilities,
                                       TouchVector::TouchIterator& titer,
                                       int direction, std::string nodeType);
  bool isPointRequired(TouchVector::TouchIterator& titer, int direction,
                       std::string nodeType);

  // number of capsules that are organized into a single compartment
  // NOTE: This is not the exact number for each compartment, just
  //       the suggested one, as #capsules per branch may not the
  //       exact multiple of this value
  int _compartmentSize;

#ifdef HAVE_MPI
  void setGenerated(std::map<Touch*, std::list<std::pair<int, int> > >& smap,
                    TouchVector::TouchIterator& titer, int type, int order);
  bool isGenerated(std::map<Touch*, std::list<std::pair<int, int> > >& smap,
                   TouchVector::TouchIterator& titer, int type, int order);
  void setNonGenerated(TouchVector::TouchIterator& titer, int direction,
                       std::string type, int order);
  bool isNonGenerated(TouchVector::TouchIterator& titer, int direction,
                      std::string nodeType, int order);
  std::list<Params::ChemicalSynapseTarget>::iterator
      getChemicalSynapseTargetFromOrder(TouchVector::TouchIterator& titer,
                                        int direction, std::string type,
                                        int order);

  RNG& findSynapseGenerator(int preRank, int postRank);
  void resetSynapseGenerator(RNG& rng, int rank1, int rank2);
  int getCountAndIncrement(std::map<int, int>& cmap, int);

  int _size;
  int _rank;
  int _nbrGridNodes;  // number of computing nodes for the grid (=X*Y*Z)
  // vector keeping track of different kinds of layers
  //  there are current 10 kinds of layers
  std::vector<GridLayerDescriptor*> _compartmentVariableLayers;
  std::vector<GridLayerDescriptor*> _junctionLayers;
  std::vector<GridLayerDescriptor*> _endPointLayers;
  std::vector<GridLayerDescriptor*> _junctionPointLayers;
  std::vector<GridLayerDescriptor*> _channelLayers;
  std::vector<GridLayerDescriptor*> _electricalSynapseLayers;
  // std::vector<GridLayerDescriptor*> _spineBranchLayers;
  std::vector<GridLayerDescriptor*> _chemicalSynapseLayers;
  std::vector<GridLayerDescriptor*> _preSynapticPointLayers;
  std::vector<GridLayerDescriptor*> _forwardSolvePointLayers;
  std::vector<GridLayerDescriptor*> _backwardSolvePointLayers;

  // the pointer referencing to the most important object
  //          which holds information to all different parts of the simulation
  static TissueContext* _tissueContext;
  static int _instanceCounter;

  // the data will be added to these std::map during doLayout()
  //   map a given name given nodekind in layer statement
  //      to the branch (ComputeBranch*) that the nodetype associated
  //      with that layer
  // <"nodeType", <"nodeIndex", < density-index, ComputeBranch*>
  // e.g.:
  // <"Voltage", <MPI-rank|gridnode|compute-index, < array-element-index,
  // ComputeBranch* the associated branch>>
  std::map<std::string, std::map<int, std::map<int, ComputeBranch*> > >
      _indexBranchMap;
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >
      _branchIndexMap;

  // <"nodeType", <"junctionIndex", < density-index[of-that-junctionIndex],
  // Capsule*>
  // nodeType is passed via nodekind="Junctions[nodeType]" Layer statement
  // junctionIndex = index of the Layer statement that use 'Junctions'
  // Capsule* = the last capsule in the proximal-side ComputeBranch
  std::map<std::string, std::map<int, std::map<int, Capsule*> > >
      _indexJunctionMap;
  // NOTE: A capsule can belong to a junction, e.g. Junction['Voltage'],
  // Junction['Calcium'],
  //         of a particular density-index
  // <"nodeType", <Capsule*, vector{"junctionIndex",
  // "density-index[of-that-junctionIndex]"}>>
  std::map<std::string, std::map<Capsule*, std::vector<int> > >
      _junctionIndexMap;
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >
      _branchForwardSolvePointIndexMap;
  std::map<std::string, std::map<ComputeBranch*, std::vector<int> > >
      _branchBackwardSolvePointIndexMap;
  //<nodeType, <Capsule*, density-index>>
  // NOTE: density-index = index in the density array of Capsules for the
  // current gridnode
  std::map<std::string, std::map<Capsule*, int> > _capsuleCptPointIndexMap;
  std::map<std::string, std::map<Capsule*, int> > _capsuleJctPointIndexMap;

  //[index-layer-for-the-same-nodetype][density-index-of-node-that-channel-getinputs][branch-index]<
  // node-index,  layer-index>
  std::vector<std::vector<std::vector<std::pair<int, int> > > >
      _channelBranchIndices1, _channelJunctionIndices1;
  //[index-layer-for-the-same-nodetype][density-index-of-node-that-channel-produceoutputs][branch-index]<
  // node-index,  layer-index>
  std::vector<std::vector<std::vector<std::pair<int, int> > > >
      _channelBranchIndices2, _channelJunctionIndices2;

  // keep tracks of the current number of layers being declared for nodetype of
  // the associated type (e.g. channel, bidirectionalsynapse, ...)
  int _channelTypeCounter;
  int _electricalSynapseTypeCounter;
  int _chemicalSynapseTypeCounter;
  int _compartmentVariableTypeCounter;
  int _junctionTypeCounter;
  int _preSynapticPointTypeCounter;
  int _endPointTypeCounter;
  int _junctionPointTypeCounter;
  int _forwardSolvePointTypeCounter;
  int _backwardSolvePointTypeCounter;
  LensConnector _lensConnector;
  GranuleConnector _granuleConnector;
  NoConnectConnector _noConnector;

  std::auto_ptr<Functor> _layoutFunctor;
  std::auto_ptr<Functor> _nodeInitFunctor;
  std::auto_ptr<Functor> _connectorFunctor;
  std::auto_ptr<Functor> _probeFunctor;
  std::auto_ptr<NDPairList> _params;
  Params _tissueParams;

  // EXAMPLE: A touch is defined in SynParams.par
  // BIDIRECTIONAL_CONNECTION_TARGETS 4
  // BRANCHTYPE MTYPE
  // BRANCHTYPE MTYPE
  // 3 2     3 0   DenSpine [Voltage Calcium] 1.0
  // 3 2     4 0   DenSpine [Voltage Calcium] 1.0
  // 3 3     3 0   DenSpine [Voltage Calcium] 1.0
  // 3 3     4 0   DenSpine [Voltage Calcium] 1.0
  // /       \
	  //left    right
  //
  // for each Touch (formed by a Capsule on left to a Capsule on right)
  //  it has a list of  <index-(point|chemical|electrical)-layer,order>
  //  in two direction [0] (left->right)
  //                   [1] (right->left)
  std::map<Touch*, std::list<std::pair<int, int> > >
      _generatedChemicalSynapses[2];
  std::map<Touch*,
           std::list<std::list<Params::ChemicalSynapseTarget>::iterator> >
      _nonGeneratedMixedChemicalSynapses[2];
  std::map<Touch*, std::list<std::pair<int, int> > >
      _generatedElectricalSynapses[2];
  // std::map<Touch*, std::list<std::pair<int, int> > >
  // _generatedSpineBranch[2];
  std::map<int, std::map<int, RNG> > _synapseGeneratorMap;

  // the list of names for all diffusible nodetypes pass through nodekind
  // argument
  //    in the Layer statement
  std::vector<std::string> _compartmentVariableTypes;
  // map the name of nodeType
  //     to its associated index (based on the order of Layer adding in GSL)
  //   NOTE: The name of nodeType is passed inside
  //   the square bracket of the associated components of <nodekind="...">
  //   NDPair
  //     BidirectionalConnection[...]
  //     ChemicalSynapses[...]
  //     CompartmentVariables[...]
  //     Junctions[...]
  //     Channels[...]
  //     PreSynapticPoints[...]
  //     EndPoints[...]
  //     JunctionsPoints[...]
  //     ForwardSolvePoints[...]
  //     BackwardSolvePoints[...]
  std::map<std::string, int> _electricalSynapseTypesMap,
      _chemicalSynapseTypesMap;
  std::map<std::string, int> _compartmentVariableTypesMap, _junctionTypesMap;
  std::map<std::string, int> _channelTypesMap;
  std::map<std::string, int> _preSynapticPointTypesMap, _endPointTypesMap,
      _junctionPointTypesMap;
  std::map<int, std::map<std::string, int> > _forwardSolvePointTypesMap,
      _backwardSolvePointTypesMap;

  bool _readFromFile;

  SegmentDescriptor _segmentDescriptor;
#endif
};

#endif
