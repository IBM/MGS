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

#include "C_argument_relative_nodeset.h"
#include "C_relative_nodeset.h"
#include "RelativeNodeSetDataItem.h"
#include "SyntaxError.h"
#include <memory>

void C_argument_relative_nodeset::internalExecute(LensContext *c)
{
   _rel_nodeset->execute(c);

   _rel_nodeset_DI = new RelativeNodeSetDataItem();
   _rel_nodeset_DI->setRelativeNodeSet(_rel_nodeset);
}


C_argument_relative_nodeset::C_argument_relative_nodeset(
   const C_argument_relative_nodeset& rv)
   : C_argument(rv), _rel_nodeset(0), _rel_nodeset_DI(0)
{
   if (rv._rel_nodeset) {
      _rel_nodeset = rv._rel_nodeset->duplicate();
   }
   if (rv._rel_nodeset_DI) {
      std::auto_ptr<DataItem> cc_di;
      rv._rel_nodeset_DI->duplicate(cc_di);
      _rel_nodeset_DI = dynamic_cast<RelativeNodeSetDataItem*>(
	 cc_di.release());
   }
}


C_argument_relative_nodeset::C_argument_relative_nodeset(
   C_relative_nodeset *r, SyntaxError * error)
   : C_argument(_REL_NODESET, error), _rel_nodeset(r), _rel_nodeset_DI(0)
{
}


C_argument_relative_nodeset* C_argument_relative_nodeset::duplicate() const
{
   return new C_argument_relative_nodeset(*this);
}


C_argument_relative_nodeset::~C_argument_relative_nodeset()
{
   delete _rel_nodeset;
   delete _rel_nodeset_DI;
}


DataItem* C_argument_relative_nodeset::getArgumentDataItem() const
{
   return _rel_nodeset_DI;
}

void C_argument_relative_nodeset::checkChildren() 
{
   if (_rel_nodeset) {
      _rel_nodeset->checkChildren();
      if (_rel_nodeset->isError()) {
         setError();
      }
   }
} 

void C_argument_relative_nodeset::recursivePrint() 
{
   if (_rel_nodeset) {
      _rel_nodeset->recursivePrint();
   }
   printErrorMessage();
} 
