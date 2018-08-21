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

#include "C_argument_constant.h"
#include "C_constant.h"
#include "IntDataItem.h"
#include "FloatDataItem.h"

void C_argument_constant::internalExecute(LensContext *c)
{
   _constant->execute(c);

   if(_constant->getType() == C_constant::_INT) {
      delete _int_dataitem;
      _int_dataitem = new IntDataItem;
      _int_dataitem->setInt(_constant->getInt());
   }
   else {
      delete _float_dataitem;
      _float_dataitem = new FloatDataItem;
      _float_dataitem->setFloat(_constant->getFloat());
   }

}

C_argument_constant::C_argument_constant(const C_argument_constant& rv)
   : C_argument(rv), _constant(0), _int_dataitem(0), _float_dataitem(0)
{
   if (rv._constant) {
      _constant = rv._constant->duplicate();
   }
   if (rv._int_dataitem) {
      std::auto_ptr<DataItem> cc_di;
      rv._int_dataitem->duplicate(cc_di);
      _int_dataitem = dynamic_cast<IntDataItem*>(cc_di.release());
   }

   if (rv._float_dataitem) {
      std::auto_ptr<DataItem> cc_di;
      rv._float_dataitem->duplicate(cc_di);
      _float_dataitem = dynamic_cast<FloatDataItem*>(cc_di.release());
   }
}

C_argument_constant::C_argument_constant(C_constant *cn, SyntaxError * error)
   : C_argument(_CONSTANT, error), _constant(cn), _int_dataitem(0), _float_dataitem(0)
{
}

C_argument_constant* C_argument_constant::duplicate() const
{
   return new C_argument_constant(*this);
}

C_argument_constant::~C_argument_constant()
{
   delete _constant;
   delete _int_dataitem;
   delete _float_dataitem;
}

DataItem* C_argument_constant::getArgumentDataItem() const
{
   if(_int_dataitem)
      return  _int_dataitem;
   else
      return _float_dataitem;
}

void C_argument_constant::checkChildren() 
{
   if (_constant) {
      _constant->checkChildren();
      if (_constant->isError()) {
         setError();
      }
   }
} 

void C_argument_constant::recursivePrint() 
{
   if (_constant) {
      _constant->recursivePrint();
   }
   printErrorMessage();
} 
