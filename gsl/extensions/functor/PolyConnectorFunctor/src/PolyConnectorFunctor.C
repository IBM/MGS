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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "PolyConnectorFunctor.h"
#include "CG_PolyConnectorFunctorBase.h"
#include "LensContext.h"
#include <memory>
#include <cassert>
#include "NodeType.h"
#include "GridLayerDescriptor.h"
#include "ConstantDataItem.h"
#include "VariableDataItem.h"
#include "EdgeSetDataItem.h"
#include "NodeSetDataItem.h"
#include "NDPairListDataItem.h"
#include "FunctorDataItem.h"
#include "Constant.h"
#include "Variable.h"
#include "VariableInstanceAccessor.h"
#include "EdgeSet.h"
#include "Edge.h"
#include "NodeSet.h"
#include "Node.h"
#include "NDPairList.h"
#include "ParameterSet.h"
#include "Simulation.h"
#include "Granule.h"
#include "Connector.h"
#include "LensConnector.h"
#include "GranuleConnector.h"
#include "NoConnectConnector.h"
#include "Simulation.h"

#include <iostream>

void PolyConnectorFunctor::userInitialize(LensContext* CG_c) 
{
}

void PolyConnectorFunctor::userExecute(LensContext* CG_c, std::vector<DataItem*>::const_iterator begin, std::vector<DataItem*>::const_iterator end) 
{
#ifdef DEBUG
   int mySpaceId;
   MPI_Comm_rank(MPI_COMM_WORLD, &mySpaceId);
#endif
   std::string mes = "";
   mes = mes + "PolyConnectorFunctor operates on four arguments:\n" +
      "1) source: [Functor | Constant | Variable | NodeSet | EdgeSet]\n" +
      "2) destination: [Functor | Variable | NodeSet | EdgeSet]\n" +
      "3) Source OutAttrPSet initializer [NDPairList]\n" + 
      "4) Destination InAttrPSet initializer [NDPairList]";
   if ((end - begin) != 4) {
      throw SyntaxErrorException(mes);
   }
   enum COMPONENTS {_Constant, _Variable, _NodeSet, _EdgeSet};
   COMPONENTS sourceType, destinationType;

   std::vector<DataItem*>::const_iterator it = begin;
   ConstantDataItem* sourceConstantDI=0;
   VariableDataItem* sourceVariableDI=0;
   VariableDataItem* destinationVariableDI=0;
   EdgeSetDataItem* sourceEdgeSetDI=0;
   EdgeSetDataItem* destinationEdgeSetDI=0;
   NodeSetDataItem* sourceNodeSetDI=0;
   NodeSetDataItem* destinationNodeSetDI=0;
   FunctorDataItem* sourceFunctorDI=0;
   FunctorDataItem* destinationFunctorDI=0;

   DataItem* sourceDI=0;
   sourceFunctorDI = dynamic_cast<FunctorDataItem*>(*it);
   std::auto_ptr<DataItem> ap_sourceDI;
   if (sourceFunctorDI) {
     std::vector<DataItem*> nullArgs;
     sourceFunctorDI->getFunctor()->execute(CG_c, nullArgs, ap_sourceDI);
     sourceDI=ap_sourceDI.get();
   }
   else {
     sourceDI=*it;
   }

   sourceConstantDI = dynamic_cast<ConstantDataItem*>(sourceDI);
   if (sourceConstantDI) {
      sourceType = _Constant;
   } else {
      sourceVariableDI = dynamic_cast<VariableDataItem*>(sourceDI);
      if (sourceVariableDI) {
	 sourceType = _Variable;
      } else {
	 sourceNodeSetDI = dynamic_cast<NodeSetDataItem*>(sourceDI);
	 if (sourceNodeSetDI) {
	    sourceType = _NodeSet;
	 } else {
	    sourceEdgeSetDI = dynamic_cast<EdgeSetDataItem*>(sourceDI);
	    if (sourceEdgeSetDI) {
	       sourceType = _EdgeSet;
	    } else {
	      throw SyntaxErrorException(
		 "Argument 1 is non-compliant.\n" + mes);
	    }
	 }
      }
   }

   ++it;

   DataItem* destinationDI=0;
   destinationFunctorDI = dynamic_cast<FunctorDataItem*>(*it);
   std::auto_ptr<DataItem> ap_destinationDI;
   if (destinationFunctorDI) {
     std::vector<DataItem*> nullArgs;
     destinationFunctorDI->getFunctor()->execute(CG_c, nullArgs, ap_destinationDI);
     destinationDI=ap_destinationDI.get();
   }
   else {
     destinationDI=*it;
   }

   destinationVariableDI = dynamic_cast<VariableDataItem*>(destinationDI);
   if (destinationVariableDI) {
      destinationType = _Variable;
   } else {
      destinationNodeSetDI = dynamic_cast<NodeSetDataItem*>(destinationDI);
      if (destinationNodeSetDI) {
	 destinationType = _NodeSet;
      } else {
	 destinationEdgeSetDI = dynamic_cast<EdgeSetDataItem*>(destinationDI);
	 if (destinationEdgeSetDI) {
	    destinationType = _EdgeSet;
	 } else {
	    throw SyntaxErrorException(
	       "Argument 2 is non-compliant.\n" + mes);
	 }
      }
   }
   
   ++it;
   
   NDPairListDataItem* ndpListDI = dynamic_cast<NDPairListDataItem*>(*it);
   if (ndpListDI == 0) {
      throw SyntaxErrorException("Argument 3 is non-compliant.\n" + mes);
   }
   NDPairList* sourceOutAttr = ndpListDI->getNDPairList();

   ++it;

   ndpListDI = dynamic_cast<NDPairListDataItem*>(*it);
   if (ndpListDI == 0) {
      throw SyntaxErrorException("Argument 4 is non-compliant.\n" + mes);
   }
   NDPairList* destinationInAttr = ndpListDI->getNDPairList();


   Connector* lc;

   if (CG_c->sim->isGranuleMapperPass()) {
     lc=&_noConnector;
   } else if (CG_c->sim->isCostAggregationPass()) {
     lc=&_granuleConnector;
   } else if (CG_c->sim->isSimulatePass()) {
     lc=&_lensConnector;
   } else {
     std::cerr<<"Error, ConnectSets3Functor : no connection context set!"<<std::endl;
     exit(0);
   }

   std::string error;
   std::vector<NodeDescriptor*> nodes;
   std::vector<NodeDescriptor*>::iterator nodesIter;
   bool allSamePartition;
   int pid;

   switch (sourceType) {
   case _Constant:
      switch (destinationType) {
      case _Variable:
	 lc->constantToVariable(sourceConstantDI->getConstant(),
				destinationVariableDI->getVariable(),
				sourceOutAttr,
				destinationInAttr);
	 break;
      case _NodeSet:
	 lc->constantToNodeSet(sourceConstantDI->getConstant(),
			       destinationNodeSetDI->getNodeSet(),
			       sourceOutAttr,
			       destinationInAttr);
	 break;
      case _EdgeSet:
	 lc->constantToEdgeSet(sourceConstantDI->getConstant(),
			       destinationEdgeSetDI->getEdgeSet(),
			       sourceOutAttr,
			       destinationInAttr);
	 break;
      default:
	 assert(0);
	 break;
      }
      break;
   case _Variable:
      switch (destinationType) {
      case _Variable:
	 lc->variableToVariable(sourceVariableDI->getVariable(),
				destinationVariableDI->getVariable(),
				sourceOutAttr,
				destinationInAttr, CG_c->sim);
	 break;
      case _NodeSet:
	destinationNodeSetDI->getNodeSet()->getNodes(nodes);
	if (nodes.size()>0) {
	  pid=CG_c->sim->getGranule(**nodes.begin())->getPartitionId();
	  allSamePartition=true;
	  for (nodesIter=nodes.begin();
	       nodesIter!=nodes.end() && allSamePartition;
	       ++nodesIter) 
	    allSamePartition = (pid==CG_c->sim->getGranule(**nodesIter)->getPartitionId());
	  if (!CG_c->sim->isSimulatePass() || !allSamePartition ||
	      pid==CG_c->sim->getGranule(*sourceVariableDI->getVariable())->getPartitionId() ) {
	    lc->variableToNodeSet(sourceVariableDI->getVariable(),
				  destinationNodeSetDI->getNodeSet(),
				  sourceOutAttr,
				  destinationInAttr, CG_c->sim);
	  }
	}
	break;
      case _EdgeSet:
	 lc->variableToEdgeSet(sourceVariableDI->getVariable(),
			       destinationEdgeSetDI->getEdgeSet(),
			       sourceOutAttr,
			       destinationInAttr, CG_c->sim);
	 break;
      default:
	 assert(0);
	 break;
      }
      break;
   case _NodeSet:
      error = "NodeSets can not be connected to ";
      switch (destinationType) {
      case _Variable:
	sourceNodeSetDI->getNodeSet()->getNodes(nodes);
	if (nodes.size()>0) {
	  pid=CG_c->sim->getGranule(**nodes.begin())->getPartitionId();
	  allSamePartition=true;
	  for (nodesIter=nodes.begin();
	       nodesIter!=nodes.end() && allSamePartition;
	       ++nodesIter) 
	    allSamePartition = (pid==CG_c->sim->getGranule(**nodesIter)->getPartitionId());
	  if (!CG_c->sim->isSimulatePass() || !allSamePartition ||
	      pid==CG_c->sim->getGranule(*destinationVariableDI->getVariable())->getPartitionId() ) 
	    lc->nodeSetToVariable(sourceNodeSetDI->getNodeSet(),
				  destinationVariableDI->getVariable(),
				  sourceOutAttr,
				  destinationInAttr, CG_c->sim);	      
	}
	break;
      case _NodeSet:
	throw SyntaxErrorException(
				   error + "NodeSets using this functor.\n" + mes);
	break;
      case _EdgeSet:
	throw SyntaxErrorException(
				   error + "EdgeSets using this functor.\n" + mes);
	break;
      default:
	assert(0);
	break;
      }
     break;
   case _EdgeSet:
      error = "EdgeSets can not be connected to ";
      switch (destinationType) {
      case _Variable:
	 lc->edgeSetToVariable(sourceEdgeSetDI->getEdgeSet(),
			       destinationVariableDI->getVariable(),
			       sourceOutAttr,
			       destinationInAttr, CG_c->sim);
	 break;
      case _NodeSet:
	 throw SyntaxErrorException(
	    error + "NodeSets using this functor.\n" + mes);
	 break;
      case _EdgeSet:
	 throw SyntaxErrorException(
	    error + "EdgeSets using this functor.\n" + mes);
	 break;
      default:
	 assert(0);
	 break;
      }
      break;
   default:
      assert(0);
      break;
   }
}

PolyConnectorFunctor::PolyConnectorFunctor() 
   : CG_PolyConnectorFunctorBase()
{
}

PolyConnectorFunctor::~PolyConnectorFunctor() 
{
}

void PolyConnectorFunctor::duplicate(std::auto_ptr<PolyConnectorFunctor>& dup) const
{
   dup.reset(new PolyConnectorFunctor(*this));
}

void PolyConnectorFunctor::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new PolyConnectorFunctor(*this));
}

void PolyConnectorFunctor::duplicate(std::auto_ptr<CG_PolyConnectorFunctorBase>& dup) const
{
   dup.reset(new PolyConnectorFunctor(*this));
}

