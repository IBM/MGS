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
