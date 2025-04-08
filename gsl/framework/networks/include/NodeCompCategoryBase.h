// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef NodeCompCategoryBase_H
#define NodeCompCategoryBase_H

#define NODESARRAY

#include "Copyright.h"

#include <memory>
#include <string>
#include <deque>

#include "DistributableCompCategoryBase.h"
#include "NodeType.h"

class ParameterSet;
class NodeAccessor;
class GridLayerDescriptor;
class GridLayerData;
class Simulation;
class WorkUnit;
class NodePartitionItem;

class NodeCompCategoryBase : public DistributableCompCategoryBase, public NodeType
{

   public:
      NodeCompCategoryBase(Simulation& sim, const std::string& modelName);
      virtual ~NodeCompCategoryBase();

      // NodeType functions to be implemented
      virtual void getNodeAccessor(
	 std::unique_ptr<NodeAccessor>&& nodeAccessor, 
	 GridLayerDescriptor* gridLayerDescriptor) = 0;
      virtual void getInitializationParameterSet(
	 std::unique_ptr<ParameterSet>&& initPSet) = 0;
      virtual void getInAttrParameterSet(
	 std::unique_ptr<ParameterSet>&& inAttrPSet) = 0;
      virtual void getOutAttrParameterSet(
	 std::unique_ptr<ParameterSet>&& outAttrPSet) = 0;
      virtual std::string getModelName() {
	return _modelName;
      }
      virtual const char* c_str() const {
	return _modelName.c_str();
      }
      virtual std::deque<int> const & getGridLayerDataOffsets() {
	return _gridLayerDataOffsets;
      }

      // virtual void store(std::ostream&) =0;
      // virtual void reload(std::istream&) =0;

      virtual void initPartitions(int numCores, int numGPUs);
      virtual int getNbrComputationalUnits() =0;
      virtual void allocateNode(NodeDescriptor* nd) = 0;

#ifdef HAVE_MPI
      virtual void allocateProxy(int partitionId, NodeDescriptor* nd) = 0;
      virtual void addToSendMap(int partitionId, Node* node) = 0;
#endif
//      virtual ConnectionIncrement** getComputeCost() = 0;

   protected:

      // Initialization time Single Threaded
      std::deque<GridLayerData*> _gridLayerDataList;
      std::deque<int> _gridLayerDataOffsets;

      // Run time Multi Threaded
      GridLayerData** _gridLayerDataArray;
      int _gridLayerDataArraySize;

      NodePartitionItem* _CPUpartitions;
      NodePartitionItem* _GPUpartitions;
      int _nbrCPUpartitions;
      int _nbrGPUpartitions;

      std::string _modelName;
      
   private:
//       // Disable since the GridLayerData isn't copiable.
//       NodeCompCategoryBase(const NodeCompCategoryBase& rv) 
// 	 : CompCategoryBase(rv), NodeType(rv) {}
//       // Disable since the GridLayerData isn't copiable.
//       NodeCompCategoryBase& operator=(const NodeCompCategoryBase& rv) {
// 	 return *this;
//       }    
};
#endif
