// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "ShortDataItem.h"
#include <sstream>

// Type
const char* ShortDataItem::_type = "SHORT";

// Constructors
ShortDataItem::ShortDataItem(short data)
   : _data(data)
{
}


ShortDataItem::ShortDataItem(const ShortDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void ShortDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new ShortDataItem(*this)));
}


NumericDataItem& ShortDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getShort();
   return(*this);
}


const char* ShortDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string ShortDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void ShortDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   short conv_val = (short)value;
   if (error) {
      if ((value > SHRT_MAX) || (value < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool ShortDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void ShortDataItem::setBool(bool i, Error* error)
{
   _data = (short)i;
}


char ShortDataItem::getChar(Error* error) const
{
   if (error) {
      if ((_data > CHAR_MAX) || (_data < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (char)(_data);
}


void ShortDataItem::setChar(char i, Error* error)
{
   _data = (short)i;
}


unsigned char ShortDataItem::getUnsignedChar(Error* error) const
{
   if (error) {
      if ((_data > UCHAR_MAX) || (_data < 0)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned char)(_data);
}


void ShortDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (short)i;
}


signed char ShortDataItem::getSignedChar(Error* error) const
{
   if (error) {
      if ((_data > SCHAR_MAX) || (_data < SCHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (signed char)(_data);
}


void ShortDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (short)i;
}


short ShortDataItem::getShort(Error* error) const
{
   return (short)(_data);
}


void ShortDataItem::setShort(short i, Error* error)
{
   _data = (short)i;
}


unsigned short ShortDataItem::getUnsignedShort(Error* error) const
{
   if (error) {
      if (_data < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned short)(_data);
}


void ShortDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   if (error) {
      if (i > SHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (short)i;
}


int ShortDataItem::getInt(Error* error) const
{
   return (int)(_data);
}


void ShortDataItem::setInt(int i, Error* error)
{
   if (error) {
      if ((i > SHRT_MAX) || (i < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (short)i;
}


unsigned int ShortDataItem::getUnsignedInt(Error* error) const
{
   if (error) {
      if (_data < 0) *error = CONVERSION_OUT_OF_RANGE;
   }
   return (unsigned int)(_data);
}


void ShortDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   if (error) {
      if (i > SHRT_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (short)i;
}


long ShortDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void ShortDataItem::setLong(long i, Error* error)
{
   if (error) {
      if ((i > SHRT_MAX) || (i < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (short)i;
}


float ShortDataItem::getFloat(Error* error) const
{
   return (float)(_data);
}


void ShortDataItem::setFloat(float i, Error* error)
{
   short conv_val = (short)i;
   if (error) {
      if ((i > SHRT_MAX) || (i < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double ShortDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void ShortDataItem::setDouble(double i, Error* error)
{
   short conv_val = (short)i;
   if (error) {
      if ((i > SHRT_MAX) || (i < SHRT_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
