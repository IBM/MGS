// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRIDDATAITEM_H
#define GRIDDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Grid;

class GridDataItem : public DataItem
{
   private:
      Grid *_grid;

   public:
      static const char* _type;

      virtual GridDataItem& operator=(const GridDataItem& DI);

      // Constructors
      GridDataItem(Grid *grid = 0);
      GridDataItem(const GridDataItem& DI);

      // Destructor
      ~GridDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Grid* getGrid() const;
      void setGrid(Grid* g);
      std::string getString(Error* error=0) const;

};
#endif
