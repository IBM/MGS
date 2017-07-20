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

#include "IntArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include "MaxFloatFullPrecision.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* IntArrayDataItem::_type = "INT_ARRAY";

// Constructors
IntArrayDataItem::IntArrayDataItem()
{
   _data = new std::vector<int>;
}


IntArrayDataItem::~IntArrayDataItem()
{
   delete _data;
}


IntArrayDataItem::IntArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<int>(size);
   _data->assign(size,0);
}

IntArrayDataItem::IntArrayDataItem(ShallowArray<int> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<int>(size);  
  ShallowArray<int>::const_iterator iter=data.begin(), end=data.end();
  std::vector<int>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}


IntArrayDataItem::IntArrayDataItem(const IntArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<int>(*DI._data);
}


// Utility methods

void IntArrayDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new IntArrayDataItem(*this));
}


NumericArrayDataItem& IntArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<int>(size);
   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setInt(coords,DI.getInt(coords));
   return *this;
}


const char* IntArrayDataItem::getType() const
{
   return _type;
}


// Array methods
std::string IntArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   std::vector<int>::const_iterator end = _data->end();
   for (std::vector<int>::const_iterator iter = _data->begin(); iter != end; iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string IntArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   std::ostringstream str_value;
   str_value << (*_data)[offset];
   return str_value.str();
}


void IntArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   double val;
   std::istringstream str_value(value);
   str_value>>val;
   int conv_val = (int)val;
   if (error) {
      if ((val > INT_MAX) || (val < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (val != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

bool IntArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (bool) (*_data)[offset];
}


void IntArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (int)value;
}


/* * * */

char IntArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (( (*_data)[offset] > CHAR_MAX) || ( (*_data)[offset] < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(*_data)[offset];
}


void IntArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (int)value;
}


/* * * */

unsigned char IntArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > UCHAR_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(*_data)[offset];
}


void IntArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (int)value;
}


/* * * */

signed char IntArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > SCHAR_MAX) || ((*_data)[offset] < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(*_data)[offset];
}


void IntArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (int)value;
}


/* * * */

short IntArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > SHRT_MAX) || ((*_data)[offset] < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (short)(*_data)[offset];
}


void IntArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (int)value;
}


/* * * */

unsigned short IntArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > USHRT_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(*_data)[offset];
}


void IntArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (int)value;
}


/* * * */

int IntArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (int)(*_data)[offset];
}


void IntArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (int)value;
}


const std::vector<int>* IntArrayDataItem::getIntVector() const
{
   return _data;
}


std::vector<int>* IntArrayDataItem::getModifiableIntVector()
{
   return _data;
}


/* * * */

unsigned int IntArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned int)(*_data)[offset];
}


void IntArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (value > INT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (int)value;
}


/* * * */

long IntArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (long)(*_data)[offset];
}


void IntArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (int)value;
}


/* * * */

float IntArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   float max = maxFloatFullPrecision.value();
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > max) || ((*_data)[offset] < -max))
         *error = LOSS_OF_PRECISION;
   }
   return (float)(*_data)[offset];
}


void IntArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   int conv_val = (int)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

double IntArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void IntArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"IntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   int conv_val = (int)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */
void IntArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::vector<int> *dest = new std::vector<int>(getSize(dimensions));
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
