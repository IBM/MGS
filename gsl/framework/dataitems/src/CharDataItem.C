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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "CharDataItem.h"
#include <sstream>

// Type
const char* CharDataItem::_type = "CHAR";

// Constructors
CharDataItem::CharDataItem(char data)
   : _data(data)
{
}


CharDataItem::CharDataItem(const CharDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void CharDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new CharDataItem(*this)));
}


NumericDataItem& CharDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getChar();
   return(*this);
}


const char* CharDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string CharDataItem::getString(Error* error) const
{
   std::ostringstream str_value;
   str_value<<_data;
   return str_value.str();
}


void CharDataItem::setString(std::string i, Error* error)
{
   double value;
   std::istringstream str_value(i);
   str_value>>value;
   char conv_val = (char)value;
   if (error) {
      if ((value > CHAR_MAX) || (value < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


bool CharDataItem::getBool(Error* error) const
{
   return(bool)(_data);
}


void CharDataItem::setBool(bool i, Error* error)
{
   _data = (char)i;
}


char CharDataItem::getChar(Error* error) const
{
   return (_data);
}


void CharDataItem::setChar(char i, Error* error)
{
   _data = (char)i;
}


unsigned char CharDataItem::getUnsignedChar(Error* error) const
{
   return (unsigned char)(_data);
}


void CharDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (char)i;
}


signed char CharDataItem::getSignedChar(Error* error) const
{
  /*
   if (error) {
      if (_data > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
  */
   return (signed char)(_data);
}


void CharDataItem::setSignedChar(signed char i, Error* error)
{
  /*
   if (error) {
      if (i < CHAR_MIN) *error = CONVERSION_OUT_OF_RANGE;
   }
  */
   _data = (char)i;
}


short CharDataItem::getShort(Error* error) const
{
   return (short)(_data);
}


void CharDataItem::setShort(short i, Error* error)
{
   if (error) {
      if ((i > CHAR_MAX) || (i < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (char)i;
}


unsigned short CharDataItem::getUnsignedShort(Error* error) const
{
   return (unsigned short)(_data);
}


void CharDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   if (error) {
      if (i > CHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (char)i;
}


int CharDataItem::getInt(Error* error) const
{
   return (int)(_data);
}


void CharDataItem::setInt(int i, Error* error)
{
   if (error) {
      if ((i > CHAR_MAX) || (i < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (char)i;
}


unsigned int CharDataItem::getUnsignedInt(Error* error) const
{
   return (unsigned int)(_data);
}


void CharDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   if (error) {
      if (i > CHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (char)i;
}


long CharDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void CharDataItem::setLong(long i, Error* error)
{
   if (error) {
      if ((i > CHAR_MAX) || (i < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   _data = (char)i;
}


float CharDataItem::getFloat(Error* error) const
{
   return (float)(_data);
}


void CharDataItem::setFloat(float i, Error* error)
{
   char conv_val = (char)i;
   if (error) {
      if ((i > CHAR_MAX) || (i < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}


double CharDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void CharDataItem::setDouble(double i, Error* error)
{
   char conv_val = (char)i;
   if (error) {
      if ((i > CHAR_MAX) || (i < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (i != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   _data = conv_val;
}
