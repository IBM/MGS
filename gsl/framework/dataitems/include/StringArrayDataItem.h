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

#ifndef STRINGARRAYDATAITEM_H
#define STRINGARRAYDATAITEM_H
#include "Copyright.h"

#include "ArrayDataItem.h"
#include "ShallowArray.h"

class StringArrayDataItem : public ArrayDataItem
{
   private:
      StringArrayDataItem & operator=(StringArrayDataItem const &);
      StringArrayDataItem & assign(const StringArrayDataItem &);
      std::vector<std::string> *_data;

   public:
      static const char* _type;

      // Constructors
      StringArrayDataItem();
      StringArrayDataItem(const StringArrayDataItem& DI);
      StringArrayDataItem(std::vector<int> const & dimensions);
      StringArrayDataItem(ShallowArray<std::string> const & data);
      ~StringArrayDataItem();

      // Utility methods
      void setDimensions(std::vector<int> const &dimensions);
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Array Methods
      std::string getString(Error* error=0) const;
      std::string getString(std::vector<int> coords, Error* error=0) const;
      void setString(std::vector<int> coords, std::string value, Error* error=0);
      const std::vector<std::string>* getStringVector() const;
      std::vector<std::string>* getModifiableStringVector();
};
#endif
