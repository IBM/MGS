// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SignedCharDataItem.h"
#include <climits>
#include <sstream>

// Type
const char* SignedCharDataItem::_type = "SIGNED_CHAR";

// Constructors
SignedCharDataItem::SignedCharDataItem(signed char data)
   : _data(data)
{
}


SignedCharDataItem::SignedCharDataItem(const SignedCharDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void SignedCharDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new SignedCharDataItem(*this)));
}


NumericDataItem& SignedCharDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getChar();
   return(*this);
}


const char* SignedCharDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string SignedCharDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void SignedCharDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   signed char conv_val = (signed char)value;
   if (error) {
      if ((value > SCHAR_MAX) || (value < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool SignedCharDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void SignedCharDataItem::setBool(bool i, Error* error)
{
   _data = (signed char)i;
}


char SignedCharDataItem::getChar(Error* error) const
{
  /*
   if (error) {
      if (_data < CHAR_MIN) *error = CONVERSION_OUT_OF_RANGE;
   }
  */
   return (char)(_data);
}


void SignedCharDataItem::setChar(char i, Error* error)
{
  /*
   if (error) {
      if (i > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
  */
   _data = (signed char)i;
}


unsigned char SignedCharDataItem::getUnsignedChar(Error* error) const
{
   if (error) {
      if (_data < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(_data);
}


void SignedCharDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   if (error) {
      if (i > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (signed char)i;
}


signed char SignedCharDataItem::getSignedChar(Error* error) const
{
   return (signed char)(_data);
}


void SignedCharDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (signed char)i;
}


short SignedCharDataItem::getShort(Error* error) const
{
   return (short)(_data);
}


void SignedCharDataItem::setShort(short i, Error* error)
{
   if (error) {
      if ((i > SCHAR_MAX) || (i < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (signed char)i;
}


unsigned short SignedCharDataItem::getUnsignedShort(Error* error) const
{
   if (error) {
      if (_data < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(_data);
}


void SignedCharDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   if (error) {
      if (i > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (signed char)i;
}


int SignedCharDataItem::getInt(Error* error) const
{
   return (int)(_data);
}


void SignedCharDataItem::setInt(int i, Error* error)
{
   if (error) {
      if ((i > SCHAR_MAX) || (i < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (signed char)i;
}


unsigned int SignedCharDataItem::getUnsignedInt(Error* error) const
{
   if (error) {
      if (_data < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned int)(_data);
}


void SignedCharDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   if (error) {
      if (i > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (signed char)i;
}


long SignedCharDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void SignedCharDataItem::setLong(long i, Error* error)
{
   if (error) {
      if ((i > SCHAR_MAX) || (i < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (signed char)i;
}


float SignedCharDataItem::getFloat(Error* error) const
{
   return (float)(_data);
}


void SignedCharDataItem::setFloat(float i, Error* error)
{
   signed char conv_val = (signed char)i;
   if (error) {
      if ((i > SCHAR_MAX) || (i < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double SignedCharDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void SignedCharDataItem::setDouble(double i, Error* error)
{
   signed char conv_val = (signed char)i;
   if (error) {
      if ((i > SCHAR_MAX) || (i < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
