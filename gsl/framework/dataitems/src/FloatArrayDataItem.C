// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FloatArrayDataItem.h"
#include <sstream>
#include "VolumeOdometer.h"
#include "MaxFloatFullPrecision.h"
#include <float.h>
#include <climits>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "VectorOstream.h"

// Type
const char* FloatArrayDataItem::_type = "FLOAT_ARRAY";

// Constructor
FloatArrayDataItem::FloatArrayDataItem()
:_data(0)
{
   _data = new std::vector<float>;
}


FloatArrayDataItem::~FloatArrayDataItem()
{
   delete _data;
}

FloatArrayDataItem::FloatArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<float>(size);
   _data->assign(size,0);
}

FloatArrayDataItem::FloatArrayDataItem(ShallowArray<float> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<float>(size);  
  ShallowArray<float>::const_iterator iter=data.begin(), end=data.end();
  std::vector<float>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}

FloatArrayDataItem::FloatArrayDataItem(const FloatArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<float>(*DI._data);
}

// Utility method
void FloatArrayDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new FloatArrayDataItem(*this)));
}


NumericArrayDataItem& FloatArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<float>(size);
   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setFloat(coords,DI.getFloat(coords));
   return(*this);
}


const char* FloatArrayDataItem::getType() const
{
   return _type;
}


// Array method
std::string FloatArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   std::vector<float>::const_iterator end = _data->end();
   for (std::vector<float>::const_iterator iter = _data->begin(); iter != end; iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string FloatArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   std::string rval;
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   std::ostringstream str_value;
   str_value<<(*_data)[offset];
   rval = str_value.str();
   return rval;
}


void FloatArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   std::istringstream str_value (value);
   str_value>>(*_data)[offset];
}


/* * * */
bool FloatArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (bool)(*_data)[offset];
}


void FloatArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (float)value;
}


/* * * */
char FloatArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   char conv_val = (char)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > CHAR_MAX) || ((*_data)[offset] < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (float)value;
}


/* * * */
unsigned char FloatArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned char conv_val = (unsigned char)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > UCHAR_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (float)value;
}


/* * * */

signed char FloatArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   signed char conv_val = (signed char)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > SCHAR_MAX) || ((*_data)[offset] < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (float)value;
}


/* * * */

short FloatArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   short conv_val = (short)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > SHRT_MAX) || ((*_data)[offset] < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (float)value;
}


/* * * */

unsigned short FloatArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned short conv_val = (unsigned short)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > USHRT_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (float)value;
}


/* * * */

int FloatArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   int conv_val = (int)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > INT_MAX) || ((*_data)[offset] < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   float max = maxFloatFullPrecision.value();
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > max) || (value < -max)) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = (float)value;
}


/* * * */

unsigned int FloatArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned int conv_val = (unsigned int)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > UINT_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   float max = maxFloatFullPrecision.value();
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (value > max) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = (float)value;
}


/* * * */

long FloatArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   long conv_val = (long)(*_data)[offset];
   if (error) {
      if ((*_data)[offset] != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   float max = maxFloatFullPrecision.value();
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > max) || (value < -max)) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = (float)value;
}


/* * * */

float FloatArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (float)(*_data)[offset];
}


void FloatArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (float)value;
}


const std::vector<float>* FloatArrayDataItem::getFloatVector() const
{
   return _data;
}


std::vector<float>* FloatArrayDataItem::getModifiableFloatVector()
{
   return _data;
}


/* * * */

double FloatArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void FloatArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr <<"FloatArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   float conv_val = (float)value;
   if (error) {
      if ((value > FLT_MAX) || (value < FLT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

void FloatArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   unsigned size = getSize(dimensions);
   std::vector<float> *dest = new std::vector<float>(size);
   dest->assign(dest->size(),0);

   // find overlap between old dimensions and new
   unsigned oldsize = _dimensions.size();
   unsigned newsize = dimensions.size();
   int minNumDimensions = (oldsize> newsize)?oldsize:newsize;
   std::vector<int> minDimensions(minNumDimensions);
   std::vector<int> begin(minNumDimensions);
   for(int i = 0;i<minNumDimensions;++i) {
      begin[i] = 0;
      if (i>=oldsize) minDimensions[i]=dimensions[i];
      else if (i>=newsize) minDimensions[i]=_dimensions[i];
      else minDimensions[i] = (_dimensions[i]>dimensions[i]) ? _dimensions[i]:dimensions[i];
   }

   // Copy appropriate values to new vector
   VolumeOdometer odmtr(begin, minDimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() ) {
      unsigned sourceOffset = getOffset(coords);
      unsigned destOffset = getOffset(dimensions,coords);
      if (destOffset<dest->size() && sourceOffset<_data->size())
	(*dest)[destOffset] = (*_data)[sourceOffset];
   }

   // replace old vector with new
   delete _data;
   _data = dest;
   _setDimensions(dimensions);
}
