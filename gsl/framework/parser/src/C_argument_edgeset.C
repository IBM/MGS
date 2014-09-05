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

#include "C_argument_edgeset.h"
#include "C_edgeset.h"
#include "EdgeSet.h"
#include "EdgeSetDataItem.h"
#include "SyntaxError.h"

void C_argument_edgeset::internalExecute(LensContext *c)
{
   _edgesetDI = new EdgeSetDataItem();
   _edgeset->execute(c);
   _edgesetDI->setEdgeSet(_edgeset->getEdgeSet());
}

C_argument_edgeset::C_argument_edgeset(const C_argument_edgeset& rv)
   : C_argument(rv), _edgeset(0), _edgesetDI(0)
{
   if (rv._edgeset) {
      _edgeset = rv._edgeset->duplicate();
   }
   if (rv._edgesetDI) {
      std::auto_ptr<DataItem> cc_di;
      rv._edgesetDI->duplicate(cc_di);
      _edgesetDI = dynamic_cast<EdgeSetDataItem*>(cc_di.release());
   }
}

C_argument_edgeset::C_argument_edgeset(C_edgeset *es, SyntaxError * error)
   : C_argument(_EDGESET, error), _edgeset(es), _edgesetDI(0)
{
}

DataItem* C_argument_edgeset::getArgumentDataItem() const 
{
   return _edgesetDI;
}

C_argument_edgeset* C_argument_edgeset::duplicate() const
{
   return new C_argument_edgeset(*this);
}

C_argument_edgeset::~C_argument_edgeset()
{
   delete _edgeset;
   delete _edgesetDI;
}

void C_argument_edgeset::checkChildren() 
{
   if (_edgeset) {
      _edgeset->checkChildren();
      if (_edgeset->isError()) {
         setError();
      }
   }
} 

void C_argument_edgeset::recursivePrint() 
{
   if (_edgeset) {
      _edgeset->recursivePrint();
   }
   printErrorMessage();
} 
