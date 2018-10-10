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
