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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;

   private:
      FunctorType *_ownedFunctorType;

};
#endif
