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

#include "DataItem.h"
#include "ArrayDataItem.h"
#include "NodeSet.h"

#include <iostream>
#include <sstream>

ArrayDataItem::ArrayDataItem()
{
}


ArrayDataItem::ArrayDataItem(std::vector<int> dimensions)
: _dimensions(dimensions)
{
}


std::vector<int> const *ArrayDataItem::getDimensions()  const
{
   return &_dimensions;
}


void ArrayDataItem::_setDimensions(std::vector<int> const &dimensions)
{
   _dimensions = dimensions;
}


unsigned ArrayDataItem::getOffset(std::vector<int> const &dimensions, std::vector<int> const &coords, Error* error) const
{
   unsigned offset = 0;
   bool inRange=(dimensions.size()!=0 && dimensions.size()==coords.size());
   if (inRange) {
     for (int i=0; i<dimensions.size(); ++i) {
       if (coords[i]>dimensions[i]-1) {
	 inRange=false;
	 break;
       }
     }
   }
   if (!inRange) {
     if (error) *error = COORDS_OUT_OF_RANGE;
     offset=INT_MAX;
   }
   else {
     int tmp = 1;
     // last coordinate is least significant (C++ convention)
     for (int i=dimensions.size()-1 ; i>=0; i--) {
       offset += coords[i] * tmp;
       tmp *= dimensions[i];
     } // this loop counts the offset of coords within array of size dimensions
   }
   return offset;
}


unsigned ArrayDataItem::getOffset(std::vector<int> const &coords, Error* error) const
{
   unsigned offset = 0;
   bool inRange=(_dimensions.size()!=0 && _dimensions.size()==coords.size());
   if (inRange) {
     for (int i=0; i<_dimensions.size(); ++i) {
       if (coords[i]>_dimensions[i]-1) {
	 inRange=false;
	 break;
       }
     }
   }
   if (!inRange) {
     if (error) *error = COORDS_OUT_OF_RANGE;
     offset=INT_MAX;
   }
   else {
     int tmp = 1;
     // last coordinate is least significant (C++ convention)
     for (int i=_dimensions.size()-1 ; i>=0; i--) {
       offset += coords[i] * tmp;
       tmp *= _dimensions[i];
     } // this loop counts the offset of coords within array of size dimensions
   }
   return offset;
}


unsigned ArrayDataItem::getSize() const
{
   unsigned size=1;
   std::vector<int>::const_iterator i, end =_dimensions.end();
   for(i=_dimensions.begin();i!=end;++i) size *= *i;
   return size;
}


unsigned ArrayDataItem::getSize(std::vector<int> const &dimensions) const
{
   unsigned size=1;
   std::vector<int>::const_iterator i, end =dimensions.end();
   for(i=dimensions.begin();i!=end;++i) size *= *i;
   return size;
}
