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

#ifndef DataTypeAttribute_H
#define DataTypeAttribute_H
#include "Mdl.h"

#include "AccessType.h"
#include "DataType.h"
#include "Attribute.h"
#include <string>
#include <vector>

class DataTypeAttribute : public Attribute
{
   public:
      DataTypeAttribute(std::auto_ptr<DataType>& data, 
			int accessType = AccessType::PUBLIC);
      DataTypeAttribute(const DataTypeAttribute& rv);
      void duplicate(std::auto_ptr<Attribute>& dup) const;
      DataTypeAttribute& operator=(const DataTypeAttribute& rv);
      virtual ~DataTypeAttribute();
      
      const DataType* getDataType() const;
      void releaseDataType(std::auto_ptr<DataType>& rv);
      void setDataType(std::auto_ptr<DataType>& rv);

      virtual std::string getName() const;
      virtual std::string getType() const;
      virtual bool isBasic() const;
      virtual bool isPointer() const;
      virtual bool isOwned() const;

   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const DataTypeAttribute& rv);
      DataType* _dataType;
};

#endif
