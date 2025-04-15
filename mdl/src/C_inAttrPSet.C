// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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


