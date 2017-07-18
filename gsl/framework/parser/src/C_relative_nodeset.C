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

#include "C_relative_nodeset.h"
#include "C_gridnodeset.h"
#include "C_nodeset_extension.h"
#include "NodeSet.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

class Grid;

//#include <iostream>
#include <list>
#include <cassert>

void C_relative_nodeset::internalExecute(LensContext *c)
{
   _gridNodeSet->execute(c);
   delete _storedContext;

   // Note: executing this copy constructor can lead to recursion in 
   // ConnectionScriptFunctor, for now we'll set to NULL
   //_storedContext = new LensContext(c);
   _storedContext = 0;

}


void C_relative_nodeset::completeNodeSet(NodeSet* ns)
{
   Grid* g = ns->getGrid();
   if(_nodeSetExtension) {
     _nodeSetExtension->execute(_storedContext);
     const std::list<GridLayerDescriptor*>& layers = 
       _nodeSetExtension->getLayers(g);
     std::list<GridLayerDescriptor*>::const_iterator iter, end = layers.end();
     std::list<GridLayerDescriptor*> exportLayers;
     for (iter = layers.begin(); iter != end; ++iter)
       exportLayers.push_back(*iter);
     ns->setLayers(exportLayers);
   }

   std::vector<int> beginCoords = ns->getBeginCoords();
   std::vector<int> endCoords = ns->getEndCoords();

   const std::vector<int>& gnsBegin = _gridNodeSet->getBeginCoords();
   const std::vector<int>& gnsEnd = _gridNodeSet->getEndCoords();

   if (gnsBegin.size() != beginCoords.size()) {
      std::string mes = 
	 "relativeNodeSet applied to NodeSet of different dimensions";
      throwError(mes);
   }

   for(unsigned int idx = 0; idx < gnsBegin.size(); ++idx) {
      beginCoords[idx] += gnsBegin[idx];
      endCoords[idx] +=  gnsEnd[idx];
      if (beginCoords[idx] > endCoords[idx]) {
	 std::string mes = "application of RelativeNodeSet resulted in inverted NodeSet coordinates";
	 throwError(mes);
      }
   }
   ns->setCoords(beginCoords, endCoords);

   std::vector<int> const &indices = _nodeSetExtension->getIndices();
   if (indices.size()>0)
      ns->setIndices(indices);
}


C_relative_nodeset::C_relative_nodeset(const C_relative_nodeset& rv)
   : C_production(rv), _gridNodeSet(0), _nodeSetExtension(0), _storedContext(0)
{
   if (rv._nodeSetExtension) {
      _nodeSetExtension = rv._nodeSetExtension->duplicate();
   }
   if (rv._gridNodeSet) {
      _gridNodeSet = rv._gridNodeSet->duplicate();
   }
   if (rv._storedContext) {
      std::auto_ptr<LensContext> dup;
      rv._storedContext->duplicate(dup);
      _storedContext = dup.release();
   }
}


C_relative_nodeset::C_relative_nodeset(C_gridnodeset *g, SyntaxError * error)
   : C_production(error), _gridNodeSet(g), _nodeSetExtension(0), 
     _storedContext(0)
{
}


C_relative_nodeset::C_relative_nodeset(
   C_gridnodeset *g, C_nodeset_extension *n, SyntaxError * error)
   : C_production(error), _gridNodeSet(g), _nodeSetExtension(n), 
     _storedContext(0)
{
}


C_relative_nodeset* C_relative_nodeset::duplicate() const
{
   return new C_relative_nodeset(*this);
}


C_relative_nodeset::~C_relative_nodeset()
{
   delete _nodeSetExtension;
   delete _gridNodeSet;
   delete _storedContext;
}

void C_relative_nodeset::checkChildren() 
{
   if (_gridNodeSet) {
      _gridNodeSet->checkChildren();
      if (_gridNodeSet->isError()) {
         setError();
      }
   }
   if (_nodeSetExtension) {
      _nodeSetExtension->checkChildren();
      if (_nodeSetExtension->isError()) {
         setError();
      }
   }
} 

void C_relative_nodeset::recursivePrint() 
{
   if (_gridNodeSet) {
      _gridNodeSet->recursivePrint();
   }
   if (_nodeSetExtension) {
      _nodeSetExtension->recursivePrint();
   }
   printErrorMessage();
} 
