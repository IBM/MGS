// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "FloatDataItem.h"
#include <sstream>
#include <float.h>
#include <math.h>

// Type
const char* FloatDataItem::_type = "FLOAT";

// Constructors
FloatDataItem::FloatDataItem(float data)
   : _data(data)
{
}


FloatDataItem::FloatDataItem(const FloatDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void FloatDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new FloatDataItem(*this)));
}


NumericDataItem& FloatDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getFloat();
   return(*this);
}


const char* FloatDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string FloatDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value.precision(FLT_DIG);
   str_value.setf(std::ostringstream::scientific, std::ostringstream::floatfield);
   str_value<<_data;
   return str_value.str();
}


void FloatDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   float conv_val = (float)value;
   if (error) {
      if ((value > FLT_MAX) || (value < FLT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool FloatDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void FloatDataItem::setBool(bool i, Error* error)
{
   _data = (float)i;
}


char FloatDataItem::getChar(Error* error) const
{
   char conv_val = (char)_data;
   if (error) {
      if ((_data > CHAR_MAX) || (_data < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setChar(char i, Error* error)
{
   _data = (float)i;
}


unsigned char FloatDataItem::getUnsignedChar(Error* error) const
{
   unsigned char conv_val = (unsigned char)_data;
   if (error) {
      if ((_data > UCHAR_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (float)i;
}


signed char FloatDataItem::getSignedChar(Error* error) const
{
   signed char conv_val = (signed char)_data;
   if (error) {
      if ((_data > SCHAR_MAX) || (_data < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (float)i;
}


short FloatDataItem::getShort(Error* error) const
{
   short conv_val = (short)_data;
   if (error) {
      if ((_data > SHRT_MAX) || (_data < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setShort(short i, Error* error)
{
   _data = (float)i;
}


unsigned short FloatDataItem::getUnsignedShort(Error* error) const
{
   unsigned short conv_val = (unsigned short)_data;
   if (error) {
      if ((_data > USHRT_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   _data = (float)i;
}


int FloatDataItem::getInt(Error* error) const
{
   int conv_val = (int)_data;
   if (error) {
      if ((_data > INT_MAX) || (_data < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setInt(int i, Error* error)
{
   if (error) {
      int mantMax = int(pow(2.0, double(FLT_MANT_DIG)))-1;
      int mantMin = -mantMax - 1;
      if ((i > mantMax) || (i < mantMin)) *error = LOSS_OF_PRECISION;
   }
   _data = (float)i;
}


unsigned int FloatDataItem::getUnsignedInt(Error* error) const
{
   unsigned int conv_val = (unsigned int)_data;
   if (error) {
      if ((_data > UINT_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   if (error) {
      unsigned mantMax = unsigned(pow(2.0, double(FLT_MANT_DIG)))-1;
      if (i > mantMax) *error = LOSS_OF_PRECISION;
   }
   _data = (float)i;
}


long FloatDataItem::getLong(Error* error) const
{
   long conv_val = (long)_data;
   if (error) {
      if ((_data > LONG_MAX) || (_data < LONG_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void FloatDataItem::setLong(long i, Error* error)
{
   if (error) {
      int mantMax = int(pow(2.0, double(FLT_MANT_DIG)))-1;
      int mantMin = -mantMax - 1;
      if ((i > mantMax) || (i < mantMin)) *error = LOSS_OF_PRECISION;
   }
   _data = (float)i;
}


float FloatDataItem::getFloat(Error* error) const
{
   return (float)(_data);
}


void FloatDataItem::setFloat(float i, Error* error)
{
   _data = (float)i;
}


double FloatDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void FloatDataItem::setDouble(double i, Error* error)
{
   float conv_val = (float)i;
   if (error) {
      if ((i > FLT_MAX) || (i < FLT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
