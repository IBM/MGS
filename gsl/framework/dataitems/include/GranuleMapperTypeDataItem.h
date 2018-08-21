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

#ifndef GRANULEMAPPERTYPEDATAITEM_H
#define GRANULEMAPPERTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class GranuleMapperType;

class GranuleMapperTypeDataItem : public InstanceFactoryDataItem
{
   private:
      GranuleMapperType *_data;

   public:
      static const char* _type;

      virtual GranuleMapperTypeDataItem& operator=(const GranuleMapperTypeDataItem& DI);

      // Constructors
      GranuleMapperTypeDataItem(GranuleMapperType *data = 0);
      GranuleMapperTypeDataItem(const GranuleMapperTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;

      void setGranuleMapperType(GranuleMapperType*);
      GranuleMapperType* getGranuleMapperType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
