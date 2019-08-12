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

#ifndef STRUCTDATAITEM_H
#define STRUCTDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <vector>
#include <memory>

class Struct;

class StructDataItem : public DataItem
{

   private:
      Struct *_data;
      void copyOwnedHeap(const StructDataItem& rv);
      void destructOwnedHeap();

   public:
      static char const* _type;

      StructDataItem & operator=(const StructDataItem &);

      // Constructors
      StructDataItem();
      StructDataItem(std::unique_ptr<Struct>& data);
      StructDataItem(const StructDataItem& rv);

      // Destructor
      ~StructDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      Struct* getStruct(Error* error=0) const;
      void setStruct(std::unique_ptr<Struct>& s, Error* error=0);
      std::string getString(Error* error=0) const;

};
#endif
