// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "VariableBase.h"
//#include "CompCategory.h"
#include "Simulation.h"
#include "VariableRelationalDataUnit.h"
#include "VariableCompCategoryBase.h"
#include "Publisher.h"
// For now
//#include <iostream>

VariableBase::VariableBase()
   : Variable(), _publisher(0), _relationalDataUnit(0), _variableInstanceAccessor(0)
{
}

VariableBase::VariableBase(const VariableBase& rv)
   : Variable(rv)
{
   copyOwnedHeap(rv);
}

void VariableBase::checkAndAddPreConstant(Constant* c) {
   if (relationalDataEnabled()) {
      _relationalDataUnit->getPreConstants().push_back(c);
   }
}

void VariableBase::checkAndAddPreEdge(Edge* e) {
   if (relationalDataEnabled()) {
      _relationalDataUnit->getPreEdges().push_back(e);
   }
}

void VariableBase::checkAndAddPreNode(NodeDescriptor* n) {
   if (relationalDataEnabled()) {
      _relationalDataUnit->getPreNodes().push_back(n);
   }
}

void VariableBase::checkAndAddPreVariable(VariableDescriptor* v) {
   if (relationalDataEnabled()) {
      _relationalDataUnit->getPreVariables().push_back(v);
   }
}

void VariableBase::checkAndAddPostEdge(Edge* e) {
   if (relationalDataEnabled()) {
      _relationalDataUnit->getPostEdges().push_back(e);
   }
}

void VariableBase::checkAndAddPostNode(NodeDescriptor* n) {
   if (relationalDataEnabled()) {
      _relationalDataUnit->getPostNodes().push_back(n);
   }
}

void VariableBase::checkAndAddPostVariable(VariableDescriptor* v) {
   if (relationalDataEnabled()) {
      _relationalDataUnit->getPostVariables().push_back(v);
   }
}

bool VariableBase::relationalDataEnabled() {
  bool rval=false;
  if (getSimulation().isEdgeRelationalDataEnabled()) {
    if (_relationalDataUnit==0) _relationalDataUnit = new VariableRelationalDataUnit;
    rval=true;
  }
  return rval;
}

Simulation& VariableBase::getSimulation() {
  return getVariableType()->getSimulation();
}

const std::deque<Constant*>& VariableBase::getPreConstantList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPreConstantList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPreConstants();
}

const std::deque<Edge*>& VariableBase::getPreEdgeList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPreEdgeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPreEdges();
}

const std::deque<NodeDescriptor*>& VariableBase::getPreNodeList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPreNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPreNodes();
}

const std::deque<VariableDescriptor*>& VariableBase::getPreVariableList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPreNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPreVariables();
}

const std::deque<Edge*>& VariableBase::getPostEdgeList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPostEdgeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPostEdges();
}

const std::deque<NodeDescriptor*>& VariableBase::getPostNodeList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPostNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPostNodes();
}

const std::deque<VariableDescriptor*>& VariableBase::getPostVariableList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPostNodeList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPostVariables();
}

VariableBase::~VariableBase() 
{
   destructOwnedHeap();
}

void VariableBase::copyOwnedHeap(const VariableBase& rv)
{
   if (rv._publisher) {
      std::unique_ptr<Publisher> dup;
      rv._publisher->duplicate(std::move(dup));
      _publisher = dup.release();
   } else {
      _publisher = 0;
   }
   if (rv._relationalDataUnit) {
      std::unique_ptr<VariableRelationalDataUnit> dup;
      rv._relationalDataUnit->duplicate(dup);
      _relationalDataUnit = dup.release();
   } else {
      _relationalDataUnit = 0;
   }
}

void VariableBase::destructOwnedHeap()
{
   delete _publisher;
   delete _relationalDataUnit;
}
