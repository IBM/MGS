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

#include "C_inAttrPSet.h"
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

void C_inAttrPSet::addToList(C_generalList* gl)
{
   assert(_struct != 0);
   std::unique_ptr<StructType> iaps(_struct);
   _struct = 0;
   gl->addInAttrPSet(std::move(iaps));
}

C_inAttrPSet::C_inAttrPSet(C_dataTypeList* dtl) 
   : C_struct(dtl)
{

}

void C_inAttrPSet::duplicate(std::unique_ptr<C_inAttrPSet>&& rv) const
{
   rv.reset(new C_inAttrPSet(*this));
}

void C_inAttrPSet::duplicate(std::unique_ptr<C_struct>&& rv) const
{
   rv.reset(new C_inAttrPSet(*this));
}

void C_inAttrPSet::duplicate(std::unique_ptr<C_general>&& rv)const
{
   rv.reset(new C_inAttrPSet(*this));
}


C_inAttrPSet::~C_inAttrPSet() 
{
}


