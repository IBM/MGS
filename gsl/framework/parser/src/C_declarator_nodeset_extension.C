// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declarator_nodeset_extension.h"
#include "C_declarator.h"
#include "C_nodeset_extension.h"
#include "GslContext.h"
#include "NodeSet.h"
#include "NodeSetDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"


void C_declarator_nodeset_extension::internalExecute(GslContext *c)
{
   _declarator->execute(c);
   _nodesetExtension->execute(c);

   const NodeSetDataItem* di = dynamic_cast<const NodeSetDataItem*>(
      c->symTable.getEntry(_declarator->getName()));
   if (di == 0) {
      std::string mes = "dynamic cast of DataItem to NodeSetDataItem failed";
      throwError(mes);
   }
   NodeSet* gs = di->getNodeSet();
   _nodeset = new NodeSet(*gs);

   const std::list<GridLayerDescriptor*>& llayers = 
      _nodesetExtension->getLayers(gs->getGrid());
   const std::vector<int>& indices = _nodesetExtension->getIndices();
   if (indices.size()>0)
      _nodeset->setIndices(indices);
   _nodeset->setLayers(llayers);
}


C_declarator_nodeset_extension::C_declarator_nodeset_extension(
   const C_declarator_nodeset_extension& rv)
   : C_production(rv), _declarator(0), _nodesetExtension(0), _nodeset(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._nodesetExtension) {
      _nodesetExtension = rv._nodesetExtension->duplicate();
   }
   if (rv._nodeset) {
      _nodeset = new NodeSet(*rv._nodeset);
   }
}


C_declarator_nodeset_extension::C_declarator_nodeset_extension(
   C_declarator *d, C_nodeset_extension *n, SyntaxError * error)
   : C_production(error), _declarator(d), _nodesetExtension(n), _nodeset(0)
{
}


C_declarator_nodeset_extension* 
C_declarator_nodeset_extension::duplicate() const
{
   return new C_declarator_nodeset_extension(*this);
}


C_declarator_nodeset_extension::~C_declarator_nodeset_extension()
{
   delete _declarator;
   delete _nodesetExtension;
   delete _nodeset;
}

void C_declarator_nodeset_extension::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_nodesetExtension) {
      _nodesetExtension->checkChildren();
      if (_nodesetExtension->isError()) {
         setError();
      }
   }
} 

void C_declarator_nodeset_extension::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_nodesetExtension) {
      _nodesetExtension->recursivePrint();
   }
   printErrorMessage();
} 
