// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "DoubleDataItem.h"
#include <float.h>
#include <climits>
#include <sstream>

// Type
const char* DoubleDataItem::_type = "DOUBLE";

// Constructors
DoubleDataItem::DoubleDataItem(double data)
   : _data(data)
{
}


DoubleDataItem::DoubleDataItem(const DoubleDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void DoubleDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new DoubleDataItem(*this)));
}


NumericDataItem& DoubleDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getDouble();
   return(*this);
}


const char* DoubleDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string DoubleDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value.precision(DBL_DIG);
   str_value.setf(std::ostringstream::scientific, std::ostringstream::floatfield);
   str_value<<_data;
   return str_value.str();
}


void DoubleDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   if (error) {
      if ((value > DBL_MAX) || (value < DBL_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = value;
}


bool DoubleDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void DoubleDataItem::setBool(bool i, Error* error)
{
   _data = (double)i;
}


char DoubleDataItem::getChar(Error* error) const
{
   char conv_val = (char)_data;
   if (error) {
      if ((_data > CHAR_MAX) || (_data < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setChar(char i, Error* error)
{
   _data = (double)i;
}


unsigned char DoubleDataItem::getUnsignedChar(Error* error) const
{
   unsigned char conv_val = (unsigned char)_data;
   if (error) {
      if ((_data > UCHAR_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (double)i;
}


signed char DoubleDataItem::getSignedChar(Error* error) const
{
   signed char conv_val = (signed char)_data;
   if (error) {
      if ((_data > SCHAR_MAX) || (_data < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (double)i;
}


short DoubleDataItem::getShort(Error* error) const
{
   short conv_val = (short)_data;
   if (error) {
      if ((_data > SHRT_MAX) || (_data < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setShort(short i, Error* error)
{
   _data = (double)i;
}


unsigned short DoubleDataItem::getUnsignedShort(Error* error) const
{
   unsigned short conv_val = (unsigned short)_data;
   if (error) {
      if ((_data > USHRT_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   _data = (double)i;
}


int DoubleDataItem::getInt(Error* error) const
{
   int conv_val = (int)_data;
   if (error) {
      if ((_data > INT_MAX) || (_data < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setInt(int i, Error* error)
{
   _data = (double)i;
}


unsigned int DoubleDataItem::getUnsignedInt(Error* error) const
{
   unsigned int conv_val = (unsigned int)_data;
   if (error) {
      if ((_data > UINT_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   _data = (double)i;
}


long DoubleDataItem::getLong(Error* error) const
{
   long conv_val = (long)_data;
   if (error) {
      if ((_data > LONG_MAX) || (_data < LONG_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setLong(long i, Error* error)
{
   double conv_val = (double)i;
   if (error) {
      if (i != (long)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


float DoubleDataItem::getFloat(Error* error) const
{
   float conv_val = (float)_data;
   if (error) {
      if ((_data > FLT_MAX) || (_data < FLT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (_data != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void DoubleDataItem::setFloat(float i, Error* error)
{
   _data = (double)i;
}


double DoubleDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void DoubleDataItem::setDouble(double i, Error* error)
{
   _data = (double)i;
}
