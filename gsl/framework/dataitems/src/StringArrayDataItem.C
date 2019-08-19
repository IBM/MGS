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

#include "StringArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include "MaxFloatFullPrecision.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* StringArrayDataItem::_type = "INT_ARRAY";

// Constructors
StringArrayDataItem::StringArrayDataItem()
{
   _data = new std::vector<std::string>;
}


StringArrayDataItem::~StringArrayDataItem()
{
   delete _data;
}


StringArrayDataItem::StringArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<std::string>(size);
   _data->assign(size,0);
}

StringArrayDataItem::StringArrayDataItem(ShallowArray<std::string> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<std::string>(size);  
  ShallowArray<std::string>::const_iterator iter=data.begin(), end=data.end();
  std::vector<std::string>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}


StringArrayDataItem::StringArrayDataItem(const StringArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<std::string>(*DI._data);
}


// Utility methods

void StringArrayDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new StringArrayDataItem(*this)));
}


StringArrayDataItem& StringArrayDataItem::assign(const StringArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   *_data = *DI._data;
   return *this;
}


const char* StringArrayDataItem::getType() const
{
   return _type;
}


// Array methods
std::string StringArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   std::vector<std::string>::const_iterator end = _data->end();
   for (std::vector<std::string>::const_iterator iter = _data->begin(); iter != end; iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string StringArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"StringArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (*_data)[offset];
}


void StringArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"StringArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = value;
}


void StringArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::vector<std::string> *dest = new std::vector<std::string>(getSize(dimensions));
   dest->assign(dest->size(),0);

   // find overlap between old dimensions and new
   unsigned oldsize = _dimensions.size();
   unsigned newsize = dimensions.size();
   int minNumDimensions = (oldsize> newsize)?oldsize:newsize;
   std::vector<int> minDimensions(minNumDimensions);
   std::vector<int> begin(minNumDimensions);
   for(int i = 0;i<minNumDimensions;++i) {
      begin[i] = 0;
      minDimensions[i] = (_dimensions[i]>dimensions[i])? _dimensions[i]:dimensions[i];
   }

   // Copy appropriate values to new vector
   VolumeOdometer odmtr(begin, minDimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() ) {
      unsigned sourceOffset = getOffset(coords);
      unsigned destOffset = getOffset(dimensions,coords);
      (*dest)[destOffset] = (*_data)[sourceOffset];
   }

   // replace old vector with new
   delete _data;
   _data = dest;
   _setDimensions(dimensions);
}

const std::vector<std::string>* StringArrayDataItem::getStringVector() const
{
   return _data;
}


std::vector<std::string>* StringArrayDataItem::getModifiableStringVector()
{
   return _data;
}

