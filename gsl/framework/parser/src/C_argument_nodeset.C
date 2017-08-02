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

#include "C_argument_nodeset.h"
#include "C_nodeset.h"
#include "C_gridset.h"
#include "NodeSet.h"
#include "NodeSetDataItem.h"
#include "SyntaxError.h"

void C_argument_nodeset::internalExecute(LensContext *c)
{
   _nodeset_DI = new NodeSetDataItem();

   // nodeset case
   if (_nodeset) {
      _nodeset->execute(c);
      // Now _nodeset has a pointer to a NodeSet object
      _nodeset_DI->setNodeSet(_nodeset->getNodeSet());
   }

   // gridset case
   if (_gridset) {
      _gridset->execute(c);
      // Now we have a pointer to a GridSet object
      _nodeset_DI->setNodeSet(_gridset->getGridSet());
   }
}


C_argument_nodeset::C_argument_nodeset(const C_argument_nodeset& rv)
   : C_argument(rv), _nodeset(0), _gridset(0), _nodeset_DI(0)
{
   if (rv._nodeset) {
      _nodeset = rv._nodeset->duplicate();
   }
   if (rv._gridset) {
      _gridset = rv._gridset->duplicate();
   }
   if (rv._nodeset_DI) {
      std::auto_ptr<DataItem> cc_di;
      rv._nodeset_DI->duplicate(cc_di);
      _nodeset_DI = dynamic_cast<NodeSetDataItem*>(cc_di.release());
   }
}


C_argument_nodeset::C_argument_nodeset(C_gridset *gd, SyntaxError * error)
   : C_argument(_NODESET, error), _nodeset(0), _gridset(gd),_nodeset_DI(0)
{
}


C_argument_nodeset::C_argument_nodeset(C_nodeset *nd, SyntaxError * error)
   : C_argument(_NODESET, error), _nodeset(nd), _gridset(0), _nodeset_DI(0)
{
}


C_argument_nodeset* C_argument_nodeset::duplicate() const
{
   return new C_argument_nodeset(*this);
}


C_argument_nodeset::~C_argument_nodeset()
{
   delete _nodeset;
   delete _gridset;
   delete _nodeset_DI;
}


DataItem* C_argument_nodeset::getArgumentDataItem() const
{
   return _nodeset_DI;
}

void C_argument_nodeset::checkChildren() 
{
   if (_nodeset) {
      _nodeset->checkChildren();
      if (_nodeset->isError()) {
         setError();
      }
   }
   if (_gridset) {
      _gridset->checkChildren();
      if (_gridset->isError()) {
         setError();
      }
   }
} 

void C_argument_nodeset::recursivePrint() 
{
   if (_nodeset) {
      _nodeset->recursivePrint();
   }
   if (_gridset) {
      _gridset->recursivePrint();
   }
   printErrorMessage();
} 
