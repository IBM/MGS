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

#include "C_nodeset.h"
#include "C_gridset.h"
#include "C_nodeset_extension.h"
#include "C_declarator.h"
#include "C_declarator_nodeset_extension.h"
#include "NodeSet.h"
#include "LensContext.h"
#include "C_relative_nodeset.h"
#include "RelativeNodeSetDataItem.h"
#include "Grid.h"
#include "NodeSetDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

void C_nodeset::internalExecute(LensContext *c)
{
   // let children do work
   if(_gridSet) _gridSet->execute(c);
   if(_nodesetExtension) _nodesetExtension->execute(c);
   if(_gridSetDecl) _gridSetDecl->execute(c);
   if(_relNodeSetDecl) _relNodeSetDecl->execute(c);
   if(_declaratorNodesetExtension) _declaratorNodesetExtension->execute(c);

   // do own work
   if (_declaratorNodesetExtension) _nodeset = new NodeSet(
      *_declaratorNodesetExtension->getNodeSet());
   else if (_gridSet) {
      _nodeset = new NodeSet(*_gridSet->getGridSet());
      Grid* g = _gridSet->getGridSet()->getGrid();
      if (_nodesetExtension) {
	 const std::list<GridLayerDescriptor*>& llayers = 
	    _nodesetExtension->getLayers(g);
	 const std::vector<int>& indices = _nodesetExtension->getIndices();
	 if (indices.size()>0)
	    _nodeset->setIndices(_nodesetExtension->getIndices());
	 _nodeset->setLayers(llayers);
      } else { // include all the layers and indices of the gridset
	//_nodeset->setLayers(g->getLayers());
	assert(_nodeset->isAllLayers());
      }
   }
   else {
      // get the reference point gridcoord
      const NodeSetDataItem* gsdi = dynamic_cast<const NodeSetDataItem*>(
	 c->symTable.getEntry(_gridSetDecl->getName()) );
      if (gsdi == 0) {
	 std::string mes = 
	    "dynamic cast of DataItem to NodeSetDataItem failed";
	 throwError(mes);
      }
      NodeSet* gs = gsdi->getNodeSet();

      // get the relative nodeset
      DataItem const *di = c->symTable.getEntry(_relNodeSetDecl->getName());
      const RelativeNodeSetDataItem* rsdi = 
	 dynamic_cast<const RelativeNodeSetDataItem*>(di );
      if (rsdi == 0) {
	 std::string mes = 
	    "dynamic cast of DataItem to RelativeNodeSetDataItem failed";
	 throwError(mes);
      }
      C_relative_nodeset* rns = rsdi->getRelativeNodeSet();

      // create combined nodeset
      _nodeset = new NodeSet(*gs);
      rns->completeNodeSet(_nodeset);
   }
}


C_nodeset::C_nodeset(const C_nodeset& rv)
   : C_production(rv), _gridSet(0), _nodesetExtension(0),  _relNodeSetDecl(0), 
     _gridSetDecl(0), _declaratorNodesetExtension(0), _nodeset(0)
{
   if(rv._gridSet) {
      _gridSet = rv._gridSet->duplicate();
   }
   if(rv._nodesetExtension) {
      _nodesetExtension = rv._nodesetExtension->duplicate();
   }
   if(rv._gridSetDecl) {
      _gridSetDecl = rv._gridSetDecl->duplicate();
   }
   if(rv._relNodeSetDecl) {
      _relNodeSetDecl = rv._relNodeSetDecl->duplicate();
   }
   if(rv._declaratorNodesetExtension) {
      _declaratorNodesetExtension = 
	 rv._declaratorNodesetExtension->duplicate();
   }
   if (rv._nodeset) {
      _nodeset = new NodeSet(*rv._nodeset);
   }
}

C_nodeset::C_nodeset(C_gridset *g, SyntaxError * error)
   : C_production(error), _gridSet(g), _nodesetExtension(0), 
     _relNodeSetDecl(0), _gridSetDecl(0), _declaratorNodesetExtension(0), 
     _nodeset(0)
{
}

C_nodeset::C_nodeset(C_gridset *g, C_nodeset_extension *n, SyntaxError * error)
   : C_production(error), _gridSet(g), _nodesetExtension(n), 
     _relNodeSetDecl(0), _gridSetDecl(0), _declaratorNodesetExtension(0), 
     _nodeset(0)
{
}

C_nodeset::C_nodeset(C_declarator *d1, C_declarator *d2, SyntaxError * error)
   : C_production(error), _gridSet(0), _nodesetExtension(0), 
     _relNodeSetDecl(d2), _gridSetDecl(d1), _declaratorNodesetExtension(0), 
     _nodeset(0)
{
}

C_nodeset::C_nodeset(C_declarator_nodeset_extension *d, SyntaxError * error)
   : C_production(error), _gridSet(0), _nodesetExtension(0), 
     _relNodeSetDecl(0), _gridSetDecl(0), _declaratorNodesetExtension(d), 
     _nodeset(0)
{
}

C_nodeset* C_nodeset::duplicate() const
{
   return new C_nodeset(*this);
}


C_nodeset::~C_nodeset()
{
   delete _gridSet;
   delete _gridSetDecl;
   delete _relNodeSetDecl;
   delete _nodesetExtension;
   delete _declaratorNodesetExtension;
   delete _nodeset;
}

void C_nodeset::checkChildren() 
{
   if (_gridSet) {
      _gridSet->checkChildren();
      if (_gridSet->isError()) {
         setError();
      }
   }
   if (_nodesetExtension) {
      _nodesetExtension->checkChildren();
      if (_nodesetExtension->isError()) {
         setError();
      }
   }
   if (_relNodeSetDecl) {
      _relNodeSetDecl->checkChildren();
      if (_relNodeSetDecl->isError()) {
         setError();
      }
   }
   if (_gridSetDecl) {
      _gridSetDecl->checkChildren();
      if (_gridSetDecl->isError()) {
         setError();
      }
   }
   if (_declaratorNodesetExtension) {
      _declaratorNodesetExtension->checkChildren();
      if (_declaratorNodesetExtension->isError()) {
         setError();
      }
   }
} 

void C_nodeset::recursivePrint() 
{
   if (_gridSet) {
      _gridSet->recursivePrint();
   }
   if (_nodesetExtension) {
      _nodesetExtension->recursivePrint();
   }
   if (_relNodeSetDecl) {
      _relNodeSetDecl->recursivePrint();
   }
   if (_gridSetDecl) {
      _gridSetDecl->recursivePrint();
   }
   if (_declaratorNodesetExtension) {
      _declaratorNodesetExtension->recursivePrint();
   }
   printErrorMessage();
} 
