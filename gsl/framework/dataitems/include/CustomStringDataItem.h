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

#ifndef CUSTOMSTRINGDATAITEM_H
#define CUSTOMSTRINGDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include "CustomString.h"
#include <string>

class CustomStringDataItem : public DataItem
{
   private:
      std::string _data;

   public:
      virtual CustomStringDataItem& operator=(const CustomStringDataItem& DI);

      static const char* _type;

      // Constructors
      CustomStringDataItem(const std::string& data = "");
      CustomStringDataItem(CustomString& data);
      CustomStringDataItem(const CustomStringDataItem& DI);

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      std::string getString(Error* error=0) const;
      CustomString getLensString(Error* error=0) const;
      void setString(std::string i, Error* error=0);
};
#endif
