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

#include "BoolDataItem.h"
#include <stdio.h>
#include <stdlib.h>

// Type
const char* BoolDataItem::_type = "BOOL";

// Constructors
BoolDataItem::BoolDataItem(bool data)
   : _data(data)
{
}


BoolDataItem::BoolDataItem(const BoolDataItem& DI)
{
   _data = DI._data;
}


// Utility methods
void BoolDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(new BoolDataItem(*this));
}


NumericDataItem& BoolDataItem::assign(const NumericDataItem& DI)
{
   _data = DI.getBool();
   return(*this);
}


const char* BoolDataItem::getType() const
{
   return _type;
}


// Singlet methods
std::string BoolDataItem::getString(Error* error) const
{
   std::string rval;
   if (_data == true) rval = "1";
   if (_data == false) rval = "0";
   return rval;
}


void BoolDataItem::setString(std::string i, Error* error)
{
   if ((i=="true") || (i=="TRUE") || (i=="True") || (atoi(i.c_str())!=0)) {
      _data = true;
   }
   else if ((i=="false") || (i=="FALSE") || (i=="False") || (atoi(i.c_str())==0)) {
      _data = false;
   }
   else if (error) {
      *error = CONVERSION_OUT_OF_RANGE;
      _data = false;
   }
   return;
}


bool BoolDataItem::getBool(Error* error) const
{
   return(_data);
}


void BoolDataItem::setBool(bool i, Error* error)
{
   _data = (bool)i;
}


char BoolDataItem::getChar(Error* error) const
{
   return (char)(_data);
}


void BoolDataItem::setChar(char i, Error* error)
{
   _data = (bool)i;
}


unsigned char BoolDataItem::getUnsignedChar(Error* error) const
{
   return (unsigned char)(_data);
}


void BoolDataItem::setUnsignedChar(unsigned char i, Error* error)
{
   _data = (bool)i;
}


signed char BoolDataItem::getSignedChar(Error* error) const
{
   return (signed char)(_data);
}


void BoolDataItem::setSignedChar(signed char i, Error* error)
{
   _data = (char)i;
}


short BoolDataItem::getShort(Error* error) const
{
   return (short)(_data);
}


void BoolDataItem::setShort(short i, Error* error)
{
   _data = (bool)i;
}


unsigned short BoolDataItem::getUnsignedShort(Error* error) const
{
   return (unsigned short)(_data);
}


void BoolDataItem::setUnsignedShort(unsigned short i, Error* error)
{
   _data = (bool)i;
}


int BoolDataItem::getInt(Error* error) const
{
   return (int)(_data);
}


void BoolDataItem::setInt(int i, Error* error)
{
   _data = (bool)i;
}


unsigned int BoolDataItem::getUnsignedInt(Error* error) const
{
   return (unsigned int)(_data);
}


void BoolDataItem::setUnsignedInt(unsigned int i, Error* error)
{
   _data = (bool)i;
}


long BoolDataItem::getLong(Error* error) const
{
   return (long)(_data);
}


void BoolDataItem::setLong(long i, Error* error)
{
   _data = (bool)i;
}


float BoolDataItem::getFloat(Error* error) const
{
   return (float)(_data);
}


void BoolDataItem::setFloat(float i, Error* error)
{
   _data = (bool)i;
}


double BoolDataItem::getDouble(Error* error) const
{
   return (double)(_data);
}


void BoolDataItem::setDouble(double i, Error* error)
{
   _data = (bool)i;
}
