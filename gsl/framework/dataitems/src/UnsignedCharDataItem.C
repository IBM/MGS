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

#include "UnsignedCharDataItem.h"
#include <sstream>

// Type
const char* UnsignedCharDataItem::_type = "UNSIGNED_CHAR";

// Constructors
UnsignedCharDataItem::UnsignedCharDataItem(unsigned char data)
   : _data(data)
{
}


UnsignedCharDataItem::UnsignedCharDataItem(const UnsignedCharDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void UnsignedCharDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new UnsignedCharDataItem(*this)));
}


NumericDataItem& UnsignedCharDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getChar();
   return(*this);
}


const char* UnsignedCharDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string UnsignedCharDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void UnsignedCharDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   unsigned char conv_val = (unsigned char)value;
   if (error) {
      if ((value > UCHAR_MAX) || (value < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool UnsignedCharDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void UnsignedCharDataItem::setBool(bool i, Error* error)
{
   _data = (unsigned char)i;
}


char UnsignedCharDataItem::getChar(Error* error) const
{
   return (char)(_data);
}


void UnsignedCharDataItem::setChar(char i, Error* error)
{
   _data = (unsigned char)i;
}


unsigned char UnsignedCharDataItem::getUnsignedChar(Error* error) const
{
   return (unsigned char)(_data);
}


void UnsignedCharDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (unsigned char)i;
}


signed char UnsignedCharDataItem::getSignedChar(Error* error) const
{
   if (error) {
      if (_data > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(_data);
}


void UnsignedCharDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (unsigned char)i;
}


short UnsignedCharDataItem::getShort(Error* error) const
{
   return (short)(_data);
}


void UnsignedCharDataItem::setShort(short i, Error* error)
{
   if (error) {
      if ((i > UCHAR_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned char)i;
}


unsigned short UnsignedCharDataItem::getUnsignedShort(Error* error) const
{
   return (unsigned short)(_data);
}


void UnsignedCharDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   if (error) {
      if (i > UCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned char)i;
}


int UnsignedCharDataItem::getInt(Error* error) const
{
   return (int)(_data);
}


void UnsignedCharDataItem::setInt(int i, Error* error)
{
   if (error) {
      if ((i > UCHAR_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned char)i;
}


unsigned int UnsignedCharDataItem::getUnsignedInt(Error* error) const
{
   return (unsigned int)(_data);
}


void UnsignedCharDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   if (error) {
      if (i > UCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned char)i;
}


long UnsignedCharDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void UnsignedCharDataItem::setLong(long i, Error* error)
{
   if (error) {
      if ((i > UCHAR_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned char)i;
}


float UnsignedCharDataItem::getFloat(Error* error) const
{
   return (float)(_data);
}


void UnsignedCharDataItem::setFloat(float i, Error* error)
{
   unsigned char conv_val = (unsigned char)i;
   if (error) {
      if ((i > UCHAR_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double UnsignedCharDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void UnsignedCharDataItem::setDouble(double i, Error* error)
{
   unsigned char conv_val = (unsigned char)i;
   if (error) {
      if ((i > UCHAR_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
