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
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      Grid* getGrid() const;
      void setGrid(Grid* g);
      std::string getString(Error* error=0) const;

};
#endif
