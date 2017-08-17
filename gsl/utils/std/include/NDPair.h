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

#ifndef NDPAIR_H
#define NDPAIR_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "DataItem.h"

class NDPair
{
   public:
      NDPair(const std::string& name, std::auto_ptr<DataItem>& di);
      NDPair(const std::string& name, const std::string& value);
      NDPair(const std::string& name, double value);
      NDPair(const std::string& name, int value);
      NDPair(const NDPair& rv);
      const NDPair& operator=(const NDPair& rv);
      const std::string& getName() const;
      std::string getValue() const;
      DataItem* getDataItem() const;
      void setDataItem(DataItem* di);
      void setValue(const std::string& value);
      void setValue(int value);
      void setValue(double value);
      void getDataItemOwnership(std::auto_ptr<DataItem>& di);
      void setDataItemOwnership(std::auto_ptr<DataItem>& di);
      int operator==(const std::string& n) const;
      int operator!=(const std::string& n) const;
      ~NDPair();
   private:
      void copyContents(const NDPair& rv);
      void destructContents();
      std::string _name;
      std::auto_ptr<DataItem> _dataItem;
};
#endif
