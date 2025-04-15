// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NUMERICDATAITEM_H
#define NUMERICDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class NumericDataItem : public DataItem
{
   public:
      virtual NumericDataItem& operator=(const NumericDataItem& DI);
      virtual NumericDataItem & assign(const NumericDataItem &) =0;

      virtual std::string getString(Error* error=0) const=0;
      virtual void setString(std::string i, Error* error=0)=0;

      virtual bool getBool(Error* error=0) const=0;
      virtual void setBool(bool i, Error* error=0)=0;

      virtual char getChar(Error* error = 0) const=0;
      virtual void setChar(char i, Error* error = 0)=0;

      virtual unsigned char getUnsignedChar(Error* error = 0) const=0;
      virtual void setUnsignedChar(unsigned char i, Error* error = 0)=0;

      virtual short getShort(Error* error = 0) const=0;
      virtual void setShort(short i, Error* error = 0)=0;

      virtual unsigned short getUnsignedShort(Error* error = 0) const=0;
      virtual void setUnsignedShort(unsigned short i, Error* error = 0)=0;

      virtual int getInt(Error* error = 0) const=0;
      virtual void setInt(int i, Error* error = 0)=0;

      virtual unsigned int getUnsignedInt(Error* error = 0) const=0;
      virtual void setUnsignedInt(unsigned int i, Error* error = 0)=0;

      virtual long getLong(Error* error = 0)const=0;
      virtual void setLong(long i, Error* error = 0)=0;

      virtual float getFloat(Error* error = 0) const=0;
      virtual void setFloat(float i, Error* error = 0)=0;

      virtual double getDouble(Error* error = 0) const=0;
      virtual void setDouble(double i, Error* error = 0)=0;
};
#endif
