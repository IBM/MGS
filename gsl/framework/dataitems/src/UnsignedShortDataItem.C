// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "UnsignedShortDataItem.h"
#include <climits>
#include <sstream>

// Type
const char* UnsignedShortDataItem::_type = "UNSIGNED_SHORT";

// Constructors
UnsignedShortDataItem::UnsignedShortDataItem(unsigned short data)
   : _data(data)
{
}

UnsignedShortDataItem::UnsignedShortDataItem(const UnsignedShortDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void UnsignedShortDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new UnsignedShortDataItem(*this)));
}


NumericDataItem& UnsignedShortDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getUnsignedShort();
   return(*this);
}


const char* UnsignedShortDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string UnsignedShortDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void UnsignedShortDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   unsigned short conv_val = (unsigned short)value;
   if (error) {
      if ((value > USHRT_MAX) || (value < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool UnsignedShortDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void UnsignedShortDataItem::setBool(bool i, Error* error)
{
   _data = (unsigned short)i;
}


char UnsignedShortDataItem::getChar(Error* error) const
{
   if (error) {
      if (_data > CHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(_data);
}


void UnsignedShortDataItem::setChar(char i, Error* error)
{
   _data = (unsigned short)i;
}


unsigned char UnsignedShortDataItem::getUnsignedChar(Error* error) const
{
   if (error) {
      if (_data > UCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(_data);
}


void UnsignedShortDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (unsigned short)i;
}


signed char UnsignedShortDataItem::getSignedChar(Error* error) const
{
   if (error) {
      if (_data > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(_data);
}


void UnsignedShortDataItem::setSignedChar(signed char i, Error* error)
{
   if (error) {
      if (i < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned short)i;
}


short UnsignedShortDataItem::getShort(Error* error) const
{
   if (error) {
      if (_data > SHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (short)(_data);
}


void UnsignedShortDataItem::setShort(short i, Error* error)
{
   if (error) {
      if (i < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned short)i;
}


unsigned short UnsignedShortDataItem::getUnsignedShort(Error* error) const
{
   return (unsigned short)(_data);
}


void UnsignedShortDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   _data = (unsigned short)i;
}


int UnsignedShortDataItem::getInt(Error* error) const
{
   return (int)(_data);
}


void UnsignedShortDataItem::setInt(int i, Error* error)
{
   if (error) {
      if ((i > USHRT_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned short)i;
}


unsigned int UnsignedShortDataItem::getUnsignedInt(Error* error) const
{
   return (unsigned int)(_data);
}


void UnsignedShortDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   if (error) {
      if (i > USHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned short)i;
}


long UnsignedShortDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void UnsignedShortDataItem::setLong(long i, Error* error)
{
   if (error) {
      if ((i > USHRT_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned short)i;
}


float UnsignedShortDataItem::getFloat(Error* error) const
{
   return (float)(_data);
}


void UnsignedShortDataItem::setFloat(float i, Error* error)
{
   unsigned short conv_val = (unsigned short)i;
   if (error) {
      if ((i > USHRT_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double UnsignedShortDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void UnsignedShortDataItem::setDouble(double i, Error* error)
{
   unsigned short conv_val = (unsigned short)i;
   if (error) {
      if ((i > USHRT_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
