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

#include "GridDataItem.h"
#include "Grid.h"

// Type
const char* GridDataItem::_type = "GRID";

// Constructors
GridDataItem::GridDataItem(Grid *grid) 
   : _grid(grid)
{
}


GridDataItem::GridDataItem(const GridDataItem& DI)
{
   _grid = DI._grid;
}


// Utility methods
void GridDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new GridDataItem(*this));
}


GridDataItem& GridDataItem::operator=(const GridDataItem& DI)
{
   _grid = DI.getGrid();
   return(*this);
}


const char* GridDataItem::getType() const
{
   return _type;
}


// Singlet methods

Grid* GridDataItem::getGrid() const
{
   return _grid;
}


void GridDataItem::setGrid(Grid* g)
{
   _grid = g;
}


GridDataItem::~GridDataItem()
{
}


std::string GridDataItem::getString(Error* error) const
{
   return "";
}
