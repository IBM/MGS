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

// Common characteristics of all array data items
// Created		: 09/27/2001 ~ Charles Peck
#include "Copyright.h"

#ifndef NUMERICARRAYDATAITEM_H
#define NUMERICARRAYDATAITEM_H

#include "ArrayDataItem.h"
#include "ShallowArray.h"

class NumericArrayDataItem: public ArrayDataItem
{
   public:
      NumericArrayDataItem();
      NumericArrayDataItem(std::vector<int> const &dimensions);

      NumericArrayDataItem& operator=(const NumericArrayDataItem& DI);
      virtual NumericArrayDataItem & assign(const NumericArrayDataItem &) =0;

      virtual std::string getString(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setString(std::vector<int> coords, std::string value, Error* error = 0)=0;

      virtual bool getBool(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setBool(std::vector<int> coords, bool value, Error* error = 0)=0;

      virtual char getChar(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setChar(std::vector<int> coords, char value, Error* error = 0)=0;

      virtual signed char getSignedChar(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setSignedChar(std::vector<int> coords, signed char value, Error* error = 0)=0;

      virtual unsigned char getUnsignedChar(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error = 0)=0;

      virtual short getShort(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setShort(std::vector<int> coords, short value, Error* error = 0)=0;

      virtual unsigned short getUnsignedShort(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error = 0)=0;

      virtual int getInt(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setInt(std::vector<int> coords, int value, Error* error = 0)=0;

      virtual unsigned int getUnsignedInt(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error = 0)=0;

      virtual long getLong(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setLong(std::vector<int> coords, long value, Error* error = 0)=0;

      virtual float getFloat(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setFloat(std::vector<int> coords, float value, Error* error = 0)=0;

      virtual double getDouble(std::vector<int> coords, Error* error = 0) const=0;
      virtual void setDouble(std::vector<int> coords, double value, Error* error = 0)=0;

      virtual ~NumericArrayDataItem();
};
#endif
