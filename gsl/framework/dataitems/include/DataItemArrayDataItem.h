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

#ifndef DATAITEMARRAYDATAITEM_H
#define DATAITEMARRAYDATAITEM_H
#include "Copyright.h"

#include "ArrayDataItem.h"
#include "ShallowArray.h"

#include <memory>

class DataItemArrayDataItem : public ArrayDataItem
{
   private:
      DataItemArrayDataItem &operator=(DataItemArrayDataItem const &);
      DataItemArrayDataItem & assign(const DataItemArrayDataItem &);
      std::vector<DataItem*> *_data;

   public:
      static char const* _type;

      // Constructors & Destructor
      DataItemArrayDataItem();
      DataItemArrayDataItem(const DataItemArrayDataItem& DI);
      DataItemArrayDataItem(std::vector<int> const &dimensions);
      DataItemArrayDataItem(ShallowArray<DataItem*> const & data);
      ~DataItemArrayDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      void setDimensions(std::vector<int> const &dimensions);
      const char* getType() const;

      // Array Methods
      std::string getString(Error* error) const;
      std::string getString(std::vector<int> coords, Error* error) const;

      DataItem* getDataItem(std::vector<int> coords, Error* error=0) const;
      void setDataItem(std::vector<int> coords, std::auto_ptr<DataItem> & value, Error* error=0);
      const std::vector<DataItem*>*  getDataItemVector() const;
      std::vector<DataItem*>* getModifiableDataItemVector();
};
#endif
