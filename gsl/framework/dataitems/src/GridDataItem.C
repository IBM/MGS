// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
void GridDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
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
