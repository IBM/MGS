// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ScriptFunctorTypeDataItem.h"
#include "FunctorType.h"
#include "ScriptFunctorType.h"

ScriptFunctorTypeDataItem::ScriptFunctorTypeDataItem()
:_ownedFunctorType(0)
{
}

ScriptFunctorTypeDataItem::ScriptFunctorTypeDataItem(ScriptFunctorTypeDataItem const *f)
: FunctorTypeDataItem(f)
{
   ScriptFunctorType *sft = dynamic_cast<ScriptFunctorType*>(f->_ownedFunctorType);
   _ownedFunctorType = new ScriptFunctorType(sft);
   FunctorTypeDataItem::setFunctorType(_ownedFunctorType);
}

void ScriptFunctorTypeDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   DataItem *p= new ScriptFunctorTypeDataItem(this);
   r_aptr.reset(p);
}


void ScriptFunctorTypeDataItem::setFunctorType(FunctorType *type)
{
   ScriptFunctorType *sft = dynamic_cast<ScriptFunctorType*>(type);
   _ownedFunctorType = new ScriptFunctorType(sft);
   FunctorTypeDataItem::setFunctorType(_ownedFunctorType);
}


ScriptFunctorTypeDataItem::~ScriptFunctorTypeDataItem()
{
   delete _ownedFunctorType;
}
