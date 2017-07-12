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

#ifndef CONSTANTTYPEDATAITEM_H
#define CONSTANTTYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class ConstantType;

class ConstantTypeDataItem : public InstanceFactoryDataItem
{
   private:
      ConstantType *_data;

   public:
      static const char* _type;

      virtual ConstantTypeDataItem& operator=(const ConstantTypeDataItem& DI);

      // Constructors
      ConstantTypeDataItem(ConstantType *data = 0);
      ConstantTypeDataItem(const ConstantTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;

      void setConstantType(ConstantType*);
      ConstantType* getConstantType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
