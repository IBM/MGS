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

#ifndef VARIABLETYPEDATAITEM_H
#define VARIABLETYPEDATAITEM_H
#include "Copyright.h"

#include "InstanceFactoryDataItem.h"

#include <memory>

class VariableType;

class VariableTypeDataItem : public InstanceFactoryDataItem
{
   private:
      VariableType *_data;

   public:
      static const char* _type;

      virtual VariableTypeDataItem& operator=(const VariableTypeDataItem& DI);

      // Constructors
      VariableTypeDataItem(VariableType *data = 0);
      VariableTypeDataItem(const VariableTypeDataItem& DI);

      const char* getType() const;
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;

      void setVariableType(VariableType*);
      VariableType* getVariableType() const;
      InstanceFactory* getInstanceFactory() const;
      void setInstanceFactory(InstanceFactory*);
};
#endif
