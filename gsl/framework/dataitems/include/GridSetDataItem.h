// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GRIDSETDATAITEM_H
#define GRIDSETDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class GridSet;
class C_gridset;

class GridSetDataItem : public DataItem
{
   private:
      GridSet *_gridset;

   public:
      static char const * _type;

      virtual GridSetDataItem& operator=(const GridSetDataItem& DI);

      // Constructors
      GridSetDataItem();
      GridSetDataItem(std::unique_ptr<GridSet> gridset);
      GridSetDataItem(const GridSetDataItem& DI);

      // Destructor
      ~GridSetDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet methods
      GridSet* getGridSet() const;
      void setGridSet(GridSet* ns);
      std::string getString(Error* error=0) const;
};
#endif
