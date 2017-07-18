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

#include "UnsignedIntArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include "MaxFloatFullPrecision.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* UnsignedIntArrayDataItem::_type = "INT_ARRAY";

// Constructors
UnsignedIntArrayDataItem::UnsignedIntArrayDataItem()
{
   _data = new std::vector<unsigned>;
}


UnsignedIntArrayDataItem::~UnsignedIntArrayDataItem()
{
   delete _data;
}


UnsignedIntArrayDataItem::UnsignedIntArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<unsigned>(size);
   _data->assign(size,0);
}

UnsignedIntArrayDataItem::UnsignedIntArrayDataItem(ShallowArray<unsigned int> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<unsigned int>(size);  
  ShallowArray<unsigned int>::const_iterator iter=data.begin(), end=data.end();
  std::vector<unsigned int>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}


UnsignedIntArrayDataItem::UnsignedIntArrayDataItem(const UnsignedIntArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<unsigned>(*DI._data);
}


// Utility methods

void UnsignedIntArrayDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new UnsignedIntArrayDataItem(*this)));
}


NumericArrayDataItem& UnsignedIntArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<unsigned>(size);
   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setUnsignedInt(coords,DI.getUnsignedInt(coords));
   return *this;
}


const char* UnsignedIntArrayDataItem::getType() const
{
   return _type;
}


// Array methods
std::string UnsignedIntArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   std::vector<unsigned>::const_iterator end = _data->end();
   for (std::vector<unsigned>::const_iterator iter = _data->begin(); iter != end; iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string UnsignedIntArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   std::ostringstream str_value;
   str_value << (*_data)[offset];
   return str_value.str();
}


void UnsignedIntArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   double val;
   std::istringstream str_value(value);
   str_value>>val;
   unsigned conv_val = (unsigned)val;
   if (error) {
      if ((val > INT_MAX) || (val < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (val != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

bool UnsignedIntArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (bool) (*_data)[offset];
}


void UnsignedIntArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

char UnsignedIntArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (( (*_data)[offset] > CHAR_MAX) || ( (*_data)[offset] < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

unsigned char UnsignedIntArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > UCHAR_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

signed char UnsignedIntArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

short UnsignedIntArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] > SHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (short)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

unsigned short UnsignedIntArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > USHRT_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

int UnsignedIntArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (int)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned)value;
}


const std::vector<unsigned>* UnsignedIntArrayDataItem::getUnsignedIntVector() const
{
   return _data;
}


std::vector<unsigned>* UnsignedIntArrayDataItem::getModifiableUnsignedIntVector()
{
   return _data;
}


/* * * */

unsigned int UnsignedIntArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned int)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (value > INT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

long UnsignedIntArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (unsigned)value;
}


/* * * */

float UnsignedIntArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   float max = maxFloatFullPrecision.value();
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > max) || ((*_data)[offset] < -max))
         *error = LOSS_OF_PRECISION;
   }
   return (float)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned conv_val = (unsigned)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

double UnsignedIntArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void UnsignedIntArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedIntArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned conv_val = (unsigned)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */
void UnsignedIntArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::vector<unsigned> *dest = new std::vector<unsigned>(getSize(dimensions));
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
