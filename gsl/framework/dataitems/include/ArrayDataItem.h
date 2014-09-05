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

// Common characteristics of all array data items
// Created		: 09/27/2001 ~ Charles Peck
#include "Copyright.h"

#ifndef ARRAYDATAITEM_H
#define ARRAYDATAITEM_H

#include "DataItem.h"

#include <vector>
#include <map>

class NodeSet;

class ArrayDataItem: public DataItem
{
   public:
      ArrayDataItem();
      ArrayDataItem(std::vector<int> dimensions);

      std::vector<int> const *getDimensions() const;
      virtual void setDimensions(std::vector<int> const &dimensions) =0;
      unsigned getOffset(std::vector<int> const &coords, Error* error = 0) const;
      unsigned getOffset(std::vector<int> const &dimensions,
			 std::vector<int> const &coords, Error* error = 0) const;
      unsigned getSize() const;
      unsigned getSize(std::vector<int> const &dimensions) const;

      ~ArrayDataItem() {}

   protected:
      std::vector<int> _dimensions;
      void _setDimensions(std::vector<int> const &dimensions);
};
#endif
