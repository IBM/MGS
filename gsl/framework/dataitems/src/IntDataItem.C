// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "IntDataItem.h"
#include "MaxFloatFullPrecision.h"
#include <sstream>

// Type
const char* IntDataItem::_type = "INT";

// Constructors
IntDataItem::IntDataItem(int data)
   : _data(data)
{
}


IntDataItem::IntDataItem(const IntDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void IntDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new IntDataItem(*this)));
}


NumericDataItem& IntDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getInt();
   return(*this);
}


const char* IntDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string IntDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void IntDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   int conv_val = (int)value;
   if (error) {
      if ((value > INT_MAX) || (value < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool IntDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void IntDataItem::setBool(bool i, Error* error)
{
   _data = (int)i;
}


char IntDataItem::getChar(Error* error) const
{
   if (error) {
      if ((_data > CHAR_MAX) || (_data < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(_data);
}


void IntDataItem::setChar(char i, Error* error)
{
   _data = (int)i;
}


unsigned char IntDataItem::getUnsignedChar(Error* error) const
{
   if (error) {
      if ((_data > UCHAR_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(_data);
}


void IntDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (int)i;
}


signed char IntDataItem::getSignedChar(Error* error) const
{
   if (error) {
      if ((_data > SCHAR_MAX) || (_data < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(_data);
}


void IntDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (int)i;
}


short IntDataItem::getShort(Error* error) const
{
   if (error) {
      if ((_data > SHRT_MAX) || (_data < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (short)(_data);
}


void IntDataItem::setShort(short i, Error* error)
{
   _data = (int)i;
}


unsigned short IntDataItem::getUnsignedShort(Error* error) const
{
   if (error) {
      if ((_data > USHRT_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(_data);
}


void IntDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   _data = (int)i;
}


int IntDataItem::getInt(Error* error) const
{
   return (int)(_data);
}


void IntDataItem::setInt(int i, Error* error)
{
   _data = (int)i;
}


unsigned int IntDataItem::getUnsignedInt(Error* error) const
{
   if (error) {
      if (_data < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned int)(_data);
}


void IntDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   if (error) {
      if (i > INT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (int)i;
}


long IntDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void IntDataItem::setLong(long i, Error* error)
{
   if (error) {
      if ((i > INT_MAX) || (i < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (int)i;
}


float IntDataItem::getFloat(Error* error) const
{
   float max = maxFloatFullPrecision.value();
   if (error) {
      if ((_data > max) || (_data < -max)) *error = LOSS_OF_PRECISION;
   }
   return (float)(_data);
}


void IntDataItem::setFloat(float i, Error* error)
{
   int conv_val = (int)i;
   if (error) {
      if ((i > INT_MAX) || (i < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double IntDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void IntDataItem::setDouble(double i, Error* error)
{
   int conv_val = (int)i;
   if (error) {
      if ((i > INT_MAX) || (i < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
