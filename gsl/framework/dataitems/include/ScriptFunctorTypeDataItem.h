// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SCRIPTFUNCTORFACTORYDATAITEM_H
#define SCRIPTFUNCTORFACTORYDATAITEM_H
#include "Copyright.h"

#include "FunctorTypeDataItem.h"
class FunctorType;

class ScriptFunctorTypeDataItem : public FunctorTypeDataItem
{
   public:
      ScriptFunctorTypeDataItem();
      ScriptFunctorTypeDataItem(ScriptFunctorTypeDataItem const *);
      void setFunctorType(FunctorType *type);
      ~ScriptFunctorTypeDataItem();
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;

   private:
      FunctorType *_ownedFunctorType;

};
#endif
