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

#include "GridSetDataItem.h"
#include "GridSet.h"

// Type
const char* GridSetDataItem::_type = "GRID_SET";

// Constructors
GridSetDataItem::GridSetDataItem() 
   : _gridset(0)
{
}

GridSetDataItem::GridSetDataItem(std::auto_ptr<GridSet> gridset) 
{
   _gridset = gridset.release();
}

GridSetDataItem::GridSetDataItem(const GridSetDataItem& DI)
{
   _gridset = new GridSet(*DI._gridset);
}


// Utility methods
void GridSetDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
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
