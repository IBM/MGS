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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "ConstantBase.h"
//#include "CompCategory.h"
#include "Simulation.h"
//#include "VariableRelationalDataUnit.h"
#include "Publisher.h"
#include "ConstantRelationalDataUnit.h"

// For now
#include <iostream>

ConstantBase::ConstantBase(Simulation& sim)
   : Constant(), _publisher(0), _relationalDataUnit(0), _sim(sim)
{
   if (relationalDataEnabled()) {
      _relationalDataUnit = new ConstantRelationalDataUnit;
   }
}

ConstantBase::ConstantBase(const ConstantBase& rv)
   : Constant(rv), _sim(rv._sim)
{
   copyOwnedHeap(rv);
}

void ConstantBase::checkAndAddPostEdge(Edge* e) {
   if (_relationalDataUnit) {
      _relationalDataUnit->getPostEdges().push_back(e);
   }
}

void ConstantBase::checkAndAddPostNode(NodeDescriptor* n) {
   if (_relationalDataUnit) {
      _relationalDataUnit->getPostNodes().push_back(n);
   }
}

void ConstantBase::checkAndAddPostVariable(VariableDescriptor* v) {
   if (_relationalDataUnit) {
      _relationalDataUnit->getPostVariables().push_back(v);
   }
}

bool ConstantBase::relationalDataEnabled() {
  return getSimulation().isEdgeRelationalDataEnabled();
}

const std::deque<Edge*>& ConstantBase::getPostEdgeList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPostEdgeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPostEdges();
}

const std::deque<NodeDescriptor*>& ConstantBase::getPostNodeList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPostNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPostNodes();
}

const std::deque<VariableDescriptor*>& ConstantBase::getPostVariableList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPostNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPostVariables();
}

ConstantBase::~ConstantBase() 
{
   destructOwnedHeap();
}

void ConstantBase::copyOwnedHeap(const ConstantBase& rv)
{
   if (rv._publisher) {
      std::unique_ptr<Publisher> dup;
      rv._publisher->duplicate(std::move(dup));
      _publisher = dup.release();
   } else {
      _publisher = 0;
   }
   if (rv._relationalDataUnit) {
      std::unique_ptr<ConstantRelationalDataUnit> dup;
      rv._relationalDataUnit->duplicate(dup);
      _relationalDataUnit = dup.release();
   } else {
      _relationalDataUnit = 0;
   }
}

void ConstantBase::destructOwnedHeap()
{
   delete _publisher;
   delete _relationalDataUnit;
}
