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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "BoolArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* BoolArrayDataItem::_type = "BOOL_ARRAY";

// Constructors
BoolArrayDataItem::BoolArrayDataItem()
{
   _data = new std::vector<bool>;
}


BoolArrayDataItem::~BoolArrayDataItem()
{
   delete _data;
}


BoolArrayDataItem::BoolArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<bool>(size);
   _data->assign(size,0);
}

BoolArrayDataItem::BoolArrayDataItem(ShallowArray<bool> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<bool>(size);  
  ShallowArray<bool>::const_iterator iter=data.begin(), end=data.end();
  std::vector<bool>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}

BoolArrayDataItem::BoolArrayDataItem(const BoolArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<bool>(*DI._data);
}


// Utility methods
void BoolArrayDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new BoolArrayDataItem(*this)));
}


NumericArrayDataItem& BoolArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<bool>(size);

   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setBool(coords,DI.getBool(coords));
   return(*this);
}


void BoolArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::auto_ptr<DataItem*> diap;
   unsigned size = getOffset(dimensions);
   std::vector<bool> *dest = new std::vector<bool>(size);
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
   ArrayDataItem::_setDimensions(dimensions);
}


const char* BoolArrayDataItem::getType() const
{
   return _type;
}


// Array methods
std::string BoolArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   for (std::vector<bool>::const_iterator iter = _data->begin(); iter != _data->end(); iter++) {
      if (iter != _data->begin()) str_value<<",";
      if ((*iter) == true) str_value<<"true";
      if ((*iter) == false) str_value<<"false";
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string BoolArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   std::string rval;
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) std::cerr<<"BoolArrayDataItem: offset out of range in getString"<<std::endl;
   if ((*_data)[offset] == true) rval = "true";
   if ((*_data)[offset] == false) rval = "false";
   return rval;
}


void BoolArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) std::cerr<<"BoolArrayDataItem: offset out of range in setString"<<std::endl;
   if ((value=="true") || (value=="TRUE") || (value=="True") || (atoi(value.c_str())!=0)) (*_data)[offset] = true;
   else if ((value=="false") || (value=="FALSE") || (value=="False") || (atoi(value.c_str())==0)) (*_data)[offset] = false;
   else {
      if (error) *error = CONVERSION_OUT_OF_RANGE;
      (*_data)[offset] = false;
   }
}


/* * * */

bool BoolArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (bool)(*_data)[offset];
}


void BoolArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


const std::vector<bool>* BoolArrayDataItem::getBoolVector() const
{
   return _data;
}


std::vector<bool>* BoolArrayDataItem::getModifiableBoolVector()
{
   return _data;
}


/* * * */

char BoolArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (char)(*_data)[offset];
}


void BoolArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

unsigned char BoolArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned char)(*_data)[offset];
}


void BoolArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

signed char BoolArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (signed char)(*_data)[offset];
}


void BoolArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (char)value;
}


/* * * */

short BoolArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (short)(*_data)[offset];
}


void BoolArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

unsigned short BoolArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned short)(*_data)[offset];
}


void BoolArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

int BoolArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (int)(*_data)[offset];
}


void BoolArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

unsigned int BoolArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned int)(*_data)[offset];
}


void BoolArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

long BoolArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (long)(*_data)[offset];
}


void BoolArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

float BoolArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (float)(*_data)[offset];
}


void BoolArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}


/* * * */

double BoolArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void BoolArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"BoolArrayDataItem:: coordinate out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (bool)value;
}
