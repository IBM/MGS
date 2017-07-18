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

#ifndef STRUCTTYPEDATAITEM_H
#define STRUCTTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class StructType;

class StructTypeDataItem : public InstanceFactoryDataItem
{
   private:
      StructType *_data;

   public:
      static const char* _type;

      virtual StructTypeDataItem& operator=(const StructTypeDataItem& DI);

      // Constructors
      StructTypeDataItem(StructType *data = 0);
      StructTypeDataItem(const StructTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;

      void setStructType(StructType*);
      StructType* getStructType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
