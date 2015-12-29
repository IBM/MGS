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
   void userInitialize(LensContext* CG_c, String& commandLineArgs1, String& commandLineArgs2, 
			  String& compartmentParamFile, String& channelParamFile, String& synapseParamFile,
			  Functor*& layoutFunctor, Functor*& nodeInitFunctor,
			  Functor*& connectorFunctor, Functor*& probeFunctor);
      std::auto_ptr<Functor> userExecute(LensContext* CG_c, String& tissueElement, NDPairList*& params);

      TissueFunctor();
      TissueFunctor(TissueFunctor const &);
      virtual ~TissueFunctor();
      virtual void duplicate(std::auto_ptr<TissueFunctor>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_TissueFunctorBase>& dup) const;
      
   private:
	  //perform neuron generating from a given set of parameters
      void neuroGen(Params* params, LensContext* CG_c);
	  //perform neuron development
      void neuroDev(Params* params, LensContext* CG_c);
	  //perform touch detection from the list of neurons (.swc files) and criteria for
	  //touch detection
      void touchDetect(Params* params, LensContext* CG_c);
	  //
      int compartmentalize(LensContext* lc, NDPairList* params, std::string& nodeCategory, std::string& nodeType, int nodeIndex, int densityIndex);
	  //
      std::vector<DataItem*> const * extractCompartmentalization(NDPairList* params);
	  //get StructDataItem 
	  //from a given point (represented by coordinate  (a pointer to double which is expected to be 3-element array)+ radius + distance to soma)
      StructDataItem* getDimension(LensContext* lc, double* cds, double radius, double dist2soma);
      StructDataItem* getDimension(LensContext* lc, double* cds1, double* cds2, double radius, double dist2soma);
	  //get the list of name of nodes (passed in GSL via nodekind)
	  //from a list of layers represented as NDPairList*
      void getNodekind(const NDPairList* layerNdpl, std::vector<std::string>& nodekind);
	  //perform the connection between 2 set of nodetypes from 2 layers
      void connect(Simulation* sim, Connector* connector, NodeDescriptor* from, NodeDescriptor* to, NDPairList& ndpl);

      ShallowArray< int > doLayout(LensContext* lc);
      void doNodeInit(LensContext* lc);
      void doConnector(LensContext* lc);
      void doProbe(LensContext* lc, std::auto_ptr<NodeSet>& rval);
      ComputeBranch* findBranch(int nodeIndex, int densityIndex, std::string const & cptVariableType);
      std::vector<int>& findBranchIndices(ComputeBranch*, std::string const & cptVariableType);
      Capsule* findJunction(int nodeIndex, int densityIndex, std::string const & cptVariableType);
      std::vector<int>& findJunctionIndices(Capsule*, std::string const & cptVariableType);
      std::vector<int>& findForwardSolvePointIndices(ComputeBranch*, std::string& nodeType);
      std::vector<int>& findBackwardSolvePointIndices(ComputeBranch*, std::string& nodeType);

      void getModelParams(Params::ModelType modelType, NDPairList& paramsLocal, std::string& nodeType, key_size_t key);
      //void getCompartmentParams(NDPairList& paramsLocal, std::string& nodeType, key_size_t key);
      bool isChannelTarget(key_size_t key, std::string nodeType);
      void getElectricalSynapseProbabilities(std::vector<double>& probabilities, TouchVector::TouchIterator & titer, int direction, std::string nodeType);
      //void getSpineBranchProbabilities(std::vector<double>& probabilities, TouchVector::TouchIterator & titer, int direction, std::string nodeType);
      void getChemicalSynapseProbabilities(std::vector<double>& probabilities, TouchVector::TouchIterator & titer, int direction, std::string nodeType);
      bool isPointRequired(TouchVector::TouchIterator & titer, int direction, std::string nodeType);

      int _compartmentSize; // number of capsules that are organized into a single compartment

#ifdef HAVE_MPI
      void setGenerated(std::map<Touch*, std::list<std::pair<int, int> > >& smap,
			TouchVector::TouchIterator & titer, 
			int type, int order); 
      bool isGenerated(std::map<Touch*, std::list<std::pair<int, int> > >& smap,
		       TouchVector::TouchIterator & titer, 
		       int type, int order);
      void setNonGenerated(TouchVector::TouchIterator & titer,
			   int direction, std::string type, int order);
      bool isNonGenerated(TouchVector::TouchIterator & titer, 
			  int direction, std::string nodeType, int order);
      std::list<Params::ChemicalSynapseTarget>::iterator getChemicalSynapseTargetFromOrder(TouchVector::TouchIterator & titer, 
											   int direction, std::string type, int order);

      RNG& findSynapseGenerator(int preRank, int postRank);
      void resetSynapseGenerator(RNG& rng, int rank1, int rank2);
      int getCountAndIncrement(std::map<int, int>& cmap, int);

      int _size;
      int _rank;
      int _nbrGridNodes; //number of computing nodes for the grid (=X*Y*Z)
	   // vector keeping track of different kinds of layers
	   //  there are current 10 kinds of layers
      std::vector<GridLayerDescriptor*> _compartmentVariableLayers;
      std::vector<GridLayerDescriptor*> _junctionLayers;
      std::vector<GridLayerDescriptor*> _endPointLayers;
      std::vector<GridLayerDescriptor*> _junctionPointLayers;
      std::vector<GridLayerDescriptor*> _channelLayers;
      std::vector<GridLayerDescriptor*> _electricalSynapseLayers;
      //std::vector<GridLayerDescriptor*> _spineBranchLayers;
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
	  // <"Voltage", <compute-index, < array-element-index, ComputeBranch* the associated branch>>
      std::map<std::string, std::map<int, std::map<int, ComputeBranch*> > > _indexBranchMap;
      std::map<std::string, std::map<ComputeBranch*, std::vector<int> > > _branchIndexMap;
      std::map<std::string, std::map<int, std::map<int, Capsule*> > > _indexJunctionMap;
      std::map<std::string, std::map<Capsule*, std::vector<int> > > _junctionIndexMap;
      std::map<std::string, std::map<ComputeBranch*, std::vector<int> > > _branchForwardSolvePointIndexMap;
      std::map<std::string, std::map<ComputeBranch*, std::vector<int> > > _branchBackwardSolvePointIndexMap;
      std::map<std::string, std::map<Capsule*, int> > _capsuleCptPointIndexMap;
      std::map<std::string, std::map<Capsule*, int> > _capsuleJctPointIndexMap; 

      std::vector<std::vector<std::vector<std::pair<int, int> > > > _channelBranchIndices1, _channelBranchIndices2, _channelJunctionIndices1, _channelJunctionIndices2;

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

      std::map<Touch*, std::list<std::pair<int, int> > > _generatedChemicalSynapses[2];
      std::map<Touch*, std::list<std::list<Params::ChemicalSynapseTarget>::iterator> > _nonGeneratedMixedChemicalSynapses[2];
      std::map<Touch*, std::list<std::pair<int, int> > > _generatedElectricalSynapses[2];
      //std::map<Touch*, std::list<std::pair<int, int> > > _generatedSpineBranch[2];
      std::map<int, std::map<int, RNG> > _synapseGeneratorMap;

	   // the list of names for all diffusible nodetypes pass through nodekind argument
	   //    in the Layer statement
      std::vector<std::string> _compartmentVariableTypes;
	   // hold the name of layers passed to nodekind and its associated index (based on the order of adding in GSL)
	   //   NOTE: The name is passed inside the square bracket of the associated components
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
      std::map<std::string, int> _electricalSynapseTypesMap, _chemicalSynapseTypesMap;
      std::map<std::string, int> _compartmentVariableTypesMap, _junctionTypesMap;
      std::map<std::string, int> _channelTypesMap;
      std::map<std::string, int> _preSynapticPointTypesMap, _endPointTypesMap, _junctionPointTypesMap;
      std::map<int, std::map<std::string, int> > _forwardSolvePointTypesMap, _backwardSolvePointTypesMap;
      
      bool _readFromFile;

      SegmentDescriptor _segmentDescriptor;
#endif
};

#endif
