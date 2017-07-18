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

      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      std::string getString(Error* error=0) const;
      //    void setString(std::string i, Error* error=0);

      const std::map<std::string, DataItem* >* getMembers() const;
      std::map<std::string, DataItem* >* getModifiableMembers();
      const std::string getComplexType();
      void setComplexType(std::string & complexType);
};
#endif
