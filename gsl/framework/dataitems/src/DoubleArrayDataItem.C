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

#include "DoubleArrayDataItem.h"
#include <sstream>
#include "VolumeOdometer.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* DoubleArrayDataItem::_type = "DOUBLE_ARRAY";

// Constructors
DoubleArrayDataItem::DoubleArrayDataItem()
  : _data(0)
{
   _data = new std::vector<double>;
}


DoubleArrayDataItem::~DoubleArrayDataItem()
{
   delete _data;
}


DoubleArrayDataItem::DoubleArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<double>(size);
   _data->assign(size,0);
}

DoubleArrayDataItem::DoubleArrayDataItem(ShallowArray<double> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<double>(size);  
  ShallowArray<double>::const_iterator iter=data.begin(), end=data.end();
  std::vector<double>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}

DoubleArrayDataItem::DoubleArrayDataItem(const DoubleArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<double>(*DI._data);
}


// Utility methods
void DoubleArrayDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new DoubleArrayDataItem(*this)));
}


NumericArrayDataItem& DoubleArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<double>(size);
   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setDouble(coords,DI.getDouble(coords));
   return(*this);
}


const char* DoubleArrayDataItem::getType() const
{
   return _type;
}


// Array methods
std::string DoubleArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   for (std::vector<double>::const_iterator iter = _data->begin(); iter != _data->end(); iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string DoubleArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   std::string rval;
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   std::ostringstream str_value;
   str_value<<(*_data)[offset];
   rval = str_value.str();
   return rval;
}


void DoubleArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   double val;
   std::istringstream str_value(value);
   str_value>>val;
   if (error) {
      if ((val > DBL_MAX) || (val < DBL_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = val;
}


/* * * */

bool DoubleArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (bool)(*_data)[offset];
}


void DoubleArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

char DoubleArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   char conv_val = (char)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > CHAR_MAX) || ((*_data)[offset] < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

unsigned char DoubleArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned char conv_val = (unsigned char)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > UCHAR_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

signed char DoubleArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   signed char conv_val = (signed char)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > SCHAR_MAX) || ((*_data)[offset] < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

short DoubleArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   short conv_val = (short)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > SHRT_MAX) || ((*_data)[offset] < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != double(conv_val)) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

unsigned short DoubleArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned short conv_val = (unsigned short)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > USHRT_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

int DoubleArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   int conv_val = int((*_data)[offset]);
   if (error) {
      if (((*_data)[offset] > INT_MAX) || ((*_data)[offset] < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != double(conv_val)) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

unsigned int DoubleArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   unsigned int conv_val = (unsigned int)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > UINT_MAX) || ((*_data)[offset] < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

long DoubleArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   long conv_val = (long)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > LONG_MAX) || ((*_data)[offset] < LONG_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   double conv_val = (double)value;
   if (error) {
      if (value != (long)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

float DoubleArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   float conv_val = (float)(*_data)[offset];
   if (error) {
      if (((*_data)[offset] > FLT_MAX) || ((*_data)[offset] < FLT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if ((*_data)[offset] != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


/* * * */

double DoubleArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void DoubleArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr << "DoubleArrayDataItem: coordinates out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (double)value;
}


const std::vector<double>* DoubleArrayDataItem::getDoubleVector() const
{
   return _data;
}


std::vector<double>* DoubleArrayDataItem::getModifiableDoubleVector()
{
   return _data;
}


void DoubleArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   unsigned size = getSize(dimensions);
   std::vector<double> *dest = new std::vector<double>(size);
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
   ArrayDataItem::_setDimensions(dimensions);
}


/* * * */
