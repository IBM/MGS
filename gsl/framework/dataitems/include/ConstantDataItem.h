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

#ifndef CONSTANTDATAITEM_H
#define CONSTANTDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <vector>
#include <memory>

class Constant;

class ConstantDataItem : public DataItem
{

   private:
      Constant *_data;
      void copyOwnedHeap(const ConstantDataItem& rv);
      void destructOwnedHeap();

   public:
      static char const* _type;

      ConstantDataItem& operator=(const ConstantDataItem& rv);

      // Constructors
      ConstantDataItem();
      ConstantDataItem(std::auto_ptr<Constant> data);
      ConstantDataItem(const ConstantDataItem& rv);

      // Destructor
      ~ConstantDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      Constant* getConstant(Error* error=0) const;
      void setConstant(std::auto_ptr<Constant>& c, Error* error=0);
      std::string getString(Error* error=0) const;

};
#endif
