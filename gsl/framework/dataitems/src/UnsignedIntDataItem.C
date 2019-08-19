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

#include "UnsignedIntDataItem.h"
#include "MaxFloatFullPrecision.h"
#include <sstream>

// Type
const char* UnsignedIntDataItem::_type = "UNSIGNED_INT";

// Constructors
UnsignedIntDataItem::UnsignedIntDataItem(unsigned int data)
   : _data(data)
{
}


UnsignedIntDataItem::UnsignedIntDataItem(const UnsignedIntDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void UnsignedIntDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new UnsignedIntDataItem(*this)));
}


NumericDataItem& UnsignedIntDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getUnsignedInt();
   return(*this);
}


const char* UnsignedIntDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string UnsignedIntDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void UnsignedIntDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   unsigned int conv_val = (unsigned int)value;
   if (error) {
      if ((value > UINT_MAX) || (value < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool UnsignedIntDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void UnsignedIntDataItem::setBool(bool i, Error* error)
{
   _data = (unsigned int)i;
}


char UnsignedIntDataItem::getChar(Error* error) const
{
   if (error) {
      if (_data > CHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(_data);
}


void UnsignedIntDataItem::setChar(char i, Error* error)
{
   _data = (unsigned int)i;
}


unsigned char UnsignedIntDataItem::getUnsignedChar(Error* error) const
{
   if (error) {
      if (_data > UCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(_data);
}


void UnsignedIntDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (unsigned int)i;
}


signed char UnsignedIntDataItem::getSignedChar(Error* error) const
{
   if (error) {
      if (_data > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(_data);
}


void UnsignedIntDataItem::setSignedChar(signed char i, Error* error)
{
   if (error) {
      if (i < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned int)i;
}


short UnsignedIntDataItem::getShort(Error* error) const
{
   if (error) {
      if (_data > SHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (short)(_data);
}


void UnsignedIntDataItem::setShort(short i, Error* error)
{
   if (error) {
      if (i < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned int)i;
}


unsigned short UnsignedIntDataItem::getUnsignedShort(Error* error) const
{
   if (error) {
      if (_data > USHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(_data);
}


void UnsignedIntDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   _data = (unsigned int)i;
}


int UnsignedIntDataItem::getInt(Error* error) const
{
   if (error) {
      if (_data > INT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (int)(_data);
}


void UnsignedIntDataItem::setInt(int i, Error* error)
{
   if (error) {
      if (i < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned int)i;
}


unsigned int UnsignedIntDataItem::getUnsignedInt(Error* error) const
{
   return (unsigned int)(_data);
}


void UnsignedIntDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   _data = (unsigned int)i;
}


long UnsignedIntDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void UnsignedIntDataItem::setLong(long i, Error* error)
{
   if (error) {
      if (i<0 ||((unsigned long) i) > UINT_MAX )
         *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (unsigned int)i;
}


float UnsignedIntDataItem::getFloat(Error* error) const
{
   float max = maxFloatFullPrecision.value();
   if (error) {
      if ((_data > max) || (_data < -max)) *error = LOSS_OF_PRECISION;
   }
   return (float)(_data);
}


void UnsignedIntDataItem::setFloat(float i, Error* error)
{
   unsigned int conv_val = (unsigned int)i;
   if (error) {
      if ((i > UINT_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double UnsignedIntDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void UnsignedIntDataItem::setDouble(double i, Error* error)
{
   if (error) {
      if ((i > UINT_MAX) || (i < 0)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i-((unsigned int)i)) *error = LOSS_OF_PRECISION;
   }
   _data = (unsigned int)i;
}
