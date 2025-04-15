// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GridSetDataItem.h"
#include "GridSet.h"

// Type
const char* GridSetDataItem::_type = "GRID_SET";

// Constructors
GridSetDataItem::GridSetDataItem() 
   : _gridset(0)
{
}

GridSetDataItem::GridSetDataItem(std::unique_ptr<GridSet> gridset) 
{
   _gridset = gridset.release();
}

GridSetDataItem::GridSetDataItem(const GridSetDataItem& DI)
{
   _gridset = new GridSet(*DI._gridset);
}


// Utility methods
void GridSetDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new GridSetDataItem(*this)));
}


GridSetDataItem& GridSetDataItem::operator=(const GridSetDataItem& DI)
{
   _gridset = DI.getGridSet();
   return(*this);
}


const char* GridSetDataItem::getType() const
{
   return _type;
}


// Singlet methods

GridSet* GridSetDataItem::getGridSet() const
{
   return _gridset;
}


void GridSetDataItem::setGridSet(GridSet* gs)
{
   delete _gridset;
   _gridset = new GridSet(*gs);
}


GridSetDataItem::~GridSetDataItem()
{
   delete _gridset;
}


std::string GridSetDataItem::getString(Error* error) const
{
   return "";
}
