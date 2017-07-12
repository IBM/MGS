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

#include "DataItemArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* DataItemArrayDataItem::_type = "DATA_ITEM_ARRAY";

// Constructor
DataItemArrayDataItem::DataItemArrayDataItem()
{
   _data = new std::vector<DataItem*>;
}


DataItemArrayDataItem::DataItemArrayDataItem(const DataItemArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   std::auto_ptr<DataItem> diap;
   std::vector<DataItem*> &source = *DI._data;
   _data = new std::vector<DataItem*>(source.size());
   for (unsigned i=0;i<_data->size();++i) {
      source[i]->duplicate(diap);
      (*_data)[i] = diap.release();
   }
}


DataItemArrayDataItem::DataItemArrayDataItem(std::vector<int> const &dimensions)
:ArrayDataItem(dimensions), _data(0)
{
   int size=1;
   std::vector<int>::iterator i, end =_dimensions.end();
   for(i=_dimensions.begin();i!=end;++i) size *= *i;
   _data = new std::vector<DataItem*>(0);
   _data->clear();
   for(int i=0; i < size; i++) {
      _data->push_back(0);
   }
   //_data->assign(size,0);
   //(*_data)[size]=0;
}

DataItemArrayDataItem::DataItemArrayDataItem(ShallowArray<DataItem*> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<DataItem*>(size);  
  ShallowArray<DataItem*>::const_iterator iter=data.begin(), end=data.end();
  std::vector<DataItem*>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}


DataItemArrayDataItem::~DataItemArrayDataItem()
{
   for (unsigned i=0;i<_data->size();++i) {
      delete (*_data)[i];
   }
   delete _data;
}


// Utility method
void DataItemArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::vector<DataItem*> *dest = new std::vector<DataItem*>(getSize(dimensions));
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

   // Copy appropriate DataItem pointers to new vector
   VolumeOdometer odmtr(begin, minDimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() ) {
      unsigned sourceOffset = getOffset(coords);
      unsigned destOffset = getOffset(dimensions,coords);
      (*dest)[destOffset] = (*_data)[sourceOffset];
      (*_data)[sourceOffset]=0;
   }

   // get rid of any DataItems not retained
   for (unsigned i=0;i<_data->size();++i) {
      delete (*_data)[i];
   }

   // replace old vector with new
   delete _data;
   _data = dest;
   ArrayDataItem::_setDimensions(dimensions);
}


void DataItemArrayDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new DataItemArrayDataItem(*this)));
}


DataItemArrayDataItem& DataItemArrayDataItem::assign(const DataItemArrayDataItem& DI)
{
   // clean up old
   for (unsigned i=0;i<_data->size();++i) {
      delete (*_data)[i];
   }
   delete _data;

   // Build new
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   std::auto_ptr<DataItem> diap;
   std::vector<DataItem*> const &source = *DI.getDataItemVector();
   _data = new std::vector<DataItem*>(source.size());
   for (unsigned i=0;i<_data->size();++i) {
      source[i]->duplicate(diap);
      (*_data)[i] = diap.release();
   }
   return(*this);
}


const char* DataItemArrayDataItem::getType() const
{
   return _type;
}


// Array method
std::string DataItemArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   std::vector<DataItem*>::const_iterator end = _data->end();
   for (std::vector<DataItem*>::const_iterator iter = _data->begin(); iter != end; iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<*(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string DataItemArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   std::string rval;
   if (offset>_data->size()) std::cerr<<"DataItemArrayDataItem: offset out of range in getString!"<<std::endl;
   std::ostringstream str_value;
   str_value<<(*_data)[offset];
   rval = str_value.str();
   return rval;
}


/* * * */

DataItem* DataItemArrayDataItem::getDataItem(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>=_data->size()) {
      std::cerr<<"DataItemArrayDataItem: coordinates out of range!"<<std::endl;
   }
   return (*_data)[offset];
}


void DataItemArrayDataItem::setDataItem(std::vector<int> coords, std::auto_ptr<DataItem> & value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>=_data->size()) {
      std::cerr<<"DataItemArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = value.release();
}


const std::vector<DataItem*>* DataItemArrayDataItem::getDataItemVector() const
{
   return _data;
}


std::vector<DataItem*>* DataItemArrayDataItem::getModifiableDataItemVector()
{
   return _data;
}
