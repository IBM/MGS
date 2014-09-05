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

#include "UnsignedShortArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include "MaxFloatFullPrecision.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* UnsignedShortArrayDataItem::_type = "INT_ARRAY";

// Constructors
UnsignedShortArrayDataItem::UnsignedShortArrayDataItem()
{
   _data = new std::vector<unsigned short>;
}


UnsignedShortArrayDataItem::~UnsignedShortArrayDataItem()
{
   delete _data;
}


UnsignedShortArrayDataItem::UnsignedShortArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<unsigned short>(size);
   _data->assign(size,0);
}

UnsignedShortArrayDataItem::UnsignedShortArrayDataItem(ShallowArray<unsigned short> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<unsigned short>(size);  
  ShallowArray<unsigned short>::const_iterator iter=data.begin(), end=data.end();
  std::vector<unsigned short>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}

UnsignedShortArrayDataItem::UnsignedShortArrayDataItem(const UnsignedShortArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<unsigned short>(*DI._data);
}


// Utility methods

void UnsignedShortArrayDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new UnsignedShortArrayDataItem(*this)));
}


NumericArrayDataItem& UnsignedShortArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<unsigned short>(size);
   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setUnsignedShort(coords,DI.getUnsignedShort(coords));
   return *this;
}


const char* UnsignedShortArrayDataItem::getType() const
{
   return _type;
}


// Array methods
std::string UnsignedShortArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   std::vector<unsigned short>::const_iterator end = _data->end();
   for (std::vector<unsigned short>::const_iterator iter = _data->begin(); iter != end; iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string UnsignedShortArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   std::ostringstream str_value;
   str_value << (*_data)[offset];
   return str_value.str();
}


void UnsignedShortArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   double val;
   std::istringstream str_value(value);
   str_value>>val;
   unsigned short conv_val = (unsigned short)val;
   if (error) {
      if ((val > INT_MAX) || (val < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (val != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

bool UnsignedShortArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (bool) (*_data)[offset];
}


void UnsignedShortArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

char UnsignedShortArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] > CHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

unsigned char UnsignedShortArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] > UCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

signed char UnsignedShortArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

short UnsignedShortArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] > SHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (short)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

unsigned short UnsignedShortArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   /*
   if (error) {
      if ((*_data)[offset] > USHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   */
   return (unsigned short)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

int UnsignedShortArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (int)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (unsigned short)value;
}


const std::vector<unsigned short>* UnsignedShortArrayDataItem::getUnsignedShortVector() const
{
   return _data;
}


std::vector<unsigned short>* UnsignedShortArrayDataItem::getModifiableUnsignedShortVector()
{
   return _data;
}


/* * * */

unsigned int UnsignedShortArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned int)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (value > INT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

long UnsignedShortArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned short)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (unsigned short)value;
}


/* * * */

float UnsignedShortArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   float max = maxFloatFullPrecision.value();
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > max) || ((*_data)[offset] < -max))
         *error = LOSS_OF_PRECISION;
   }
   return (float)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned short conv_val = (unsigned short)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

double UnsignedShortArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void UnsignedShortArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"UnsignedShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned short conv_val = (unsigned short)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */
void UnsignedShortArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::vector<unsigned short> *dest = new std::vector<unsigned short>(getSize(dimensions));
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
