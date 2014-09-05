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

#ifndef STRINGDATAITEM_H
#define STRINGDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include "String.h"
#include <string>

class StringDataItem : public DataItem
{
   private:
      std::string _data;

   public:
      virtual StringDataItem& operator=(const StringDataItem& DI);

      static const char* _type;

      // Constructors
      StringDataItem(const std::string& data = "");
      StringDataItem(String& data);
      StringDataItem(const StringDataItem& DI);

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      std::string getString(Error* error=0) const;
      String getLensString(Error* error=0) const;
      void setString(std::string i, Error* error=0);
};
#endif
