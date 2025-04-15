// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "LongDataItem.h"
#include "MaxFloatFullPrecision.h"
#include <sstream>

// Type
const char* LongDataItem::_type = "LONG";

// Constructors
LongDataItem::LongDataItem(long data)
   : _data(data)
{
}


LongDataItem::LongDataItem(const LongDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void LongDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new LongDataItem(*this)));
}


NumericDataItem& LongDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getLong();
   return(*this);
}


const char* LongDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string LongDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void LongDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   long conv_val = (long)value;
   if (error) {
      if ((value > LONG_MAX) || (value < LONG_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool LongDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void LongDataItem::setBool(bool i, Error* error)
{
   _data = (long)i;
}


char LongDataItem::getChar(Error* error) const
{
   if (error) {
      if ((_data > CHAR_MAX) || (_data < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(_data);
}


void LongDataItem::setChar(char i, Error* error)
{
   _data = (long)i;
}


unsigned char LongDataItem::getUnsignedChar(Error* error) const
{
   if (error) {
      if ((_data > UCHAR_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(_data);
}


void LongDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (long)i;
}


signed char LongDataItem::getSignedChar(Error* error) const
{
   if (error) {
      if ((_data > SCHAR_MAX) || (_data < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(_data);
}


void LongDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (long)i;
}


short LongDataItem::getShort(Error* error) const
{
   if (error) {
      if ((_data > SHRT_MAX) || (_data < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (short)(_data);
}


void LongDataItem::setShort(short i, Error* error)
{
   _data = (long)i;
}


unsigned short LongDataItem::getUnsignedShort(Error* error) const
{
   if (error) {
      if ((_data > USHRT_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(_data);
}


void LongDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   _data = (long)i;
}


int LongDataItem::getInt(Error* error) const
{
   if (error) {
      if ((_data > INT_MAX) || (_data < INT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (int)(_data);
}


void LongDataItem::setInt(int i, Error* error)
{
   _data = (long)i;
}


unsigned int LongDataItem::getUnsignedInt(Error* error) const
{
   if (error) {
      if ( _data< 0 ||(unsigned long)( _data) > UINT_MAX )
         *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned int)(_data);
}


void LongDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   _data = (long)i;
}


long LongDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void LongDataItem::setLong(long i, Error* error)
{
   _data = (long)i;
}


float LongDataItem::getFloat(Error* error) const
{
   float max = maxFloatFullPrecision.value();
   if (error) {
      if ((_data > max) || (_data < -max)) *error = LOSS_OF_PRECISION;
   }
   return (float)(_data);
}


void LongDataItem::setFloat(float i, Error* error)
{
   long conv_val = (long)i;
   if (error) {
      if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double LongDataItem::getDouble(Error* error) const
{
   double conv_val = (double)_data;
   if (error) {
      if (_data != (long)conv_val) *error = LOSS_OF_PRECISION;
   }
   return conv_val;
}


void LongDataItem::setDouble(double i, Error* error)
{
   long conv_val = (long)i;
   if (error) {
      if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
