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
#include "EdgeBase.h"
//#include "CompCategory.h"
#include "Simulation.h"
//#include "Grid.h"
#include "EdgeCompCategoryBase.h"
#include "EdgeRelationalDataUnit.h"
#include "Publisher.h"


// For now
#include <iostream>

EdgeBase::EdgeBase()
   : Edge(), _publisher(0), _relationalDataUnit(0), _edgeCompCategoryBase(0),
     _postNode(0)
{
}

void EdgeBase::setEdgeCompCategoryBase(EdgeCompCategoryBase* cc)
{
   _edgeCompCategoryBase = cc;
   delete _relationalDataUnit;
   if (getSimulation().isEdgeRelationalDataEnabled()) {
      _relationalDataUnit = new EdgeRelationalDataUnit;
   }
}


EdgeBase::EdgeBase(const EdgeBase& rv)
   : Edge(rv), _publisher(0), _relationalDataUnit(0), 
     _edgeCompCategoryBase(rv._edgeCompCategoryBase)
{
   if (rv._publisher) {
      std::auto_ptr<Publisher> dup;
      rv._publisher->duplicate(dup);
      _publisher = dup.release();
   }
   if (rv._relationalDataUnit) {
      std::auto_ptr<EdgeRelationalDataUnit> dup;
      rv._relationalDataUnit->duplicate(dup);
      _relationalDataUnit = dup.release();
   }   
}

void EdgeBase::checkAndAddPreConstant(Constant* c) {
   if (_relationalDataUnit) {
      _relationalDataUnit->getPreConstants().push_back(c);
   }
}

void EdgeBase::checkAndAddPreVariable(VariableDescriptor* v) {
   if (_relationalDataUnit) {
      _relationalDataUnit->getPreVariables().push_back(v);
   }
}

void EdgeBase::checkAndAddPostVariable(VariableDescriptor* v) {
   if (_relationalDataUnit) {
      _relationalDataUnit->getPostVariables().push_back(v);
   }
}

void EdgeBase::checkAndSetPreNode(NodeDescriptor* n) {
   if (_relationalDataUnit) {
      _relationalDataUnit->setPreNode(n);
   }
}

void EdgeBase::checkAndSetPostNode(NodeDescriptor* n) {
   _postNode = n;
}

const std::deque<Constant*>& EdgeBase::getPreConstantList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPreConstantList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPreConstants();

}

const std::deque<VariableDescriptor*>& EdgeBase::getPreVariableList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPreVariableList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPreVariables();

}

const std::deque<VariableDescriptor*>& EdgeBase::getPostVariableList()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPostVariableList called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPostVariables();
}

NodeDescriptor* EdgeBase::getPreNode()
{
   if (_relationalDataUnit == 0) {
      // @TODO: Use exceptions
      std::cerr<<"getPreNode called when RelationalData is disabled!"
	       << std::endl;
      exit(-1);
   }
   return _relationalDataUnit->getPreNode();

}

NodeDescriptor* EdgeBase::getPostNode()
{
   return _postNode;
}

EdgeBase::~EdgeBase()
{
   delete _publisher;
   delete _relationalDataUnit;
}

Simulation& EdgeBase::getSimulation() 
{
   return _edgeCompCategoryBase->getSimulation();
}

std::string EdgeBase::getModelName() 
{
   return _edgeCompCategoryBase->getModelName();
}
