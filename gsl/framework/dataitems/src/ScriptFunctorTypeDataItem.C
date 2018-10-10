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
