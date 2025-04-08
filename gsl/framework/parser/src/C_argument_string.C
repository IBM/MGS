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

#include "C_argument_string.h"
#include "CustomStringDataItem.h"
#include "SyntaxError.h"

void C_argument_string::internalExecute(LensContext *c)
{
   _str_dataitem = new CustomStringDataItem();
   _str_dataitem->setString(*_string);
}


C_argument_string::C_argument_string(const C_argument_string& rv)
   : C_argument(rv), _str_dataitem(0),_string(0)
{
   if (rv._str_dataitem) {
      std::unique_ptr<DataItem> cc_di;
      rv._str_dataitem->duplicate(cc_di);
      _str_dataitem = dynamic_cast<CustomStringDataItem*>(cc_di.release());
   }
   if (rv._string) {
      _string = new std::string(*(rv._string));
   }
}


C_argument_string::C_argument_string(std::string *ss, SyntaxError * error)
   : C_argument(_STRING, error), _str_dataitem(0), _string(ss)
{
}


C_argument_string* C_argument_string::duplicate() const
{
   return new C_argument_string(*this);
}


C_argument_string::~C_argument_string()
{
   delete _str_dataitem;
   delete _string;
}


DataItem* C_argument_string::getArgumentDataItem() const
{
   return _str_dataitem;
}

void C_argument_string::checkChildren() 
{
} 

void C_argument_string::recursivePrint() 
{
   printErrorMessage();
} 
