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

#include "ShortArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include "MaxFloatFullPrecision.h"
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* ShortArrayDataItem::_type = "INT_ARRAY";

// Constructors
ShortArrayDataItem::ShortArrayDataItem()
{
   _data = new std::vector<short>;
}


ShortArrayDataItem::~ShortArrayDataItem()
{
   delete _data;
}


ShortArrayDataItem::ShortArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<short>(size);
   _data->assign(size,0);
}

ShortArrayDataItem::ShortArrayDataItem(ShallowArray<short> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<short>(size);  
  ShallowArray<short>::const_iterator iter=data.begin(), end=data.end();
  std::vector<short>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}


ShortArrayDataItem::ShortArrayDataItem(const ShortArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<short>(*DI._data);
}


// Utility methods

void ShortArrayDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new ShortArrayDataItem(*this)));
}


NumericArrayDataItem& ShortArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<short>(size);
   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setShort(coords,DI.getShort(coords));
   return *this;
}


const char* ShortArrayDataItem::getType() const
{
   return _type;
}


// Array methods
std::string ShortArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   std::vector<short>::const_iterator end = _data->end();
   for (std::vector<short>::const_iterator iter = _data->begin(); iter != end; iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string ShortArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   std::ostringstream str_value;
   str_value << (*_data)[offset];
   return str_value.str();
}


void ShortArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   double val;
   std::istringstream str_value(value);
   str_value>>val;
   short conv_val = (short)val;
   if (error) {
      if ((val > INT_MAX) || (val < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (val != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

bool ShortArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (bool) (*_data)[offset];
}


void ShortArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (short)value;
}


/* * * */

char ShortArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (( (*_data)[offset] > CHAR_MAX) || ( (*_data)[offset] < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(*_data)[offset];
}


void ShortArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (short)value;
}


/* * * */

unsigned char ShortArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > UCHAR_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(*_data)[offset];
}


void ShortArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (short)value;
}


/* * * */

signed char ShortArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > SCHAR_MAX) || ((*_data)[offset] < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(*_data)[offset];
}


void ShortArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (short)value;
}


/* * * */

short ShortArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (short)(*_data)[offset];
}


void ShortArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (short)value;
}


/* * * */

unsigned short ShortArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(*_data)[offset];
}


void ShortArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (short)value;
}


/* * * */

int ShortArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (int)(*_data)[offset];
}


void ShortArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (short)value;
}


const std::vector<short>* ShortArrayDataItem::getShortVector() const
{
   return _data;
}


std::vector<short>* ShortArrayDataItem::getModifiableShortVector()
{
   return _data;
}


/* * * */

unsigned int ShortArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((*_data)[offset] < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned int)(*_data)[offset];
}


void ShortArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (value > INT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (short)value;
}


/* * * */

long ShortArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (long)(*_data)[offset];
}


void ShortArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (short)value;
}


/* * * */

float ShortArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   float max = maxFloatFullPrecision.value();
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (((*_data)[offset] > max) || ((*_data)[offset] < -max))
         *error = LOSS_OF_PRECISION;
   }
   return (float)(*_data)[offset];
}


void ShortArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   short conv_val = (short)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

double ShortArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void ShortArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"ShortArrayDataItem:: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   short conv_val = (short)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */
void ShortArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::vector<short> *dest = new std::vector<short>(getSize(dimensions));
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
