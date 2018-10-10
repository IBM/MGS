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
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      std::string getString(Error* error=0) const;
      String getLensString(Error* error=0) const;
      void setString(std::string i, Error* error=0);
};
#endif
