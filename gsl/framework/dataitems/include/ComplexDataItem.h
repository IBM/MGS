// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMPLEXDATAITEM_H
#define COMPLEXDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

#include <map>

class ComplexDataItem : public DataItem
{
   private:
      std::string _complexType;
      std::map<std::string, DataItem*> _members;

   public:
      static const char* _type;

      virtual ComplexDataItem& operator=(const ComplexDataItem& DI);

      ComplexDataItem();
      ComplexDataItem(std::string & complexType);
      ComplexDataItem(const ComplexDataItem& DI);

      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      std::string getString(Error* error=0) const;
      //    void setString(std::string i, Error* error=0);

      const std::map<std::string, DataItem* >* getMembers() const;
      std::map<std::string, DataItem* >* getModifiableMembers();
      const std::string getComplexType();
      void setComplexType(std::string & complexType);
};
#endif
