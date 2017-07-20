// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_outAttrPSet.h"
#include "C_struct.h"
#include "MdlContext.h"
#include "C_dataTypeList.h"
#include "StructType.h"
#include "DataType.h"
#include "InternalException.h"
#include "DuplicateException.h"
#include "C_generalList.h"
#include <memory>
#include <vector>
#include <iostream>
#include <cassert>

void C_outAttrPSet::addToList(C_generalList* gl)
{
   assert(_struct != 0);
   std::auto_ptr<StructType> iaps(_struct);
   _struct = 0;
   gl->addOutAttrPSet(iaps);
}

C_outAttrPSet::C_outAttrPSet(C_dataTypeList* dtl) 
   : C_struct(dtl)
{

}

void C_outAttrPSet::duplicate(std::auto_ptr<C_outAttrPSet>& rv) const
{
   rv.reset(new C_outAttrPSet(*this));
}

void C_outAttrPSet::duplicate(std::auto_ptr<C_struct>& rv) const
{
   rv.reset(new C_outAttrPSet(*this));
}

void C_outAttrPSet::duplicate(std::auto_ptr<C_general>& rv)const
{
   rv.reset(new C_outAttrPSet(*this));
}


C_outAttrPSet::~C_outAttrPSet() 
{
}


