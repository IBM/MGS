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

#include "C_argument_gridset.h"
#include "C_gridset.h"
#include "GridSet.h"
#include "DataItem.h"
#include "GridSetDataItem.h"
#include "SyntaxError.h"
#include <memory>

void C_argument_gridset::internalExecute(LensContext *c)
{
   _gridset->execute(c);

   _dataitem = new GridSetDataItem();
   _dataitem->setGridSet(_gridset->getGridSet());
}

C_argument_gridset::C_argument_gridset(const C_argument_gridset& rv)
   : C_argument(rv), _gridset(0), _dataitem(0)
{
   if (rv._gridset) {
      _gridset = rv._gridset->duplicate();
   }
   if (rv._dataitem) {
      std::auto_ptr<DataItem> di_ap;
      rv._dataitem->duplicate(di_ap);
      _dataitem = dynamic_cast<GridSetDataItem*>(di_ap.release());
   }
}

C_argument_gridset::C_argument_gridset(C_gridset *g, SyntaxError * error)
   : C_argument(_GRIDSET, error), _gridset(g), _dataitem(0)
{
}

C_argument_gridset* C_argument_gridset::duplicate() const
{
   return new C_argument_gridset(*this);
}

DataItem* C_argument_gridset::getArgumentDataItem() const 
{
   return _dataitem;
}

C_argument_gridset::~C_argument_gridset()
{
   delete _gridset;
   delete _dataitem;
}

void C_argument_gridset::checkChildren() 
{
   if (_gridset) {
      _gridset->checkChildren();
      if (_gridset->isError()) {
         setError();
      }
   }
} 

void C_argument_gridset::recursivePrint() 
{
   if (_gridset) {
      _gridset->recursivePrint();
   }
   printErrorMessage();
} 
