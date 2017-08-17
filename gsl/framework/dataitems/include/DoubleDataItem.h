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

#ifndef DOUBLEDATAITEM_H
#define DOUBLEDATAITEM_H
#include "Copyright.h"

#include "NumericDataItem.h"

class DoubleDataItem : public NumericDataItem
{
   private:
      NumericDataItem & assign(const NumericDataItem &);
      double _data;

   public:
      static char const* _type;

      // Constructors
      DoubleDataItem(double data = 0);
      DoubleDataItem(const DoubleDataItem& DI);

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Singlet Methods
      std::string getString(Error* error=0) const;
      void setString(std::string i, Error* error=0);

      bool getBool(Error* error=0) const;
      void setBool(bool i, Error* error=0);

      char getChar(Error* error=0) const;
      void setChar(char i, Error* error=0);

      unsigned char getUnsignedChar(Error* error=0) const;
      void setUnsignedChar(unsigned char i, Error* error=0);

      signed char getSignedChar(Error* error=0) const;
      void setSignedChar(signed char i, Error* error=0);

      short getShort(Error* error=0) const;
      void setShort(short i, Error* error=0);

      unsigned short getUnsignedShort(Error* error=0) const;
      void setUnsignedShort(unsigned short i, Error* error=0);

      int getInt(Error* error=0) const;
      void setInt(int i, Error* error=0);

      unsigned int getUnsignedInt(Error* error=0) const;
      void setUnsignedInt(unsigned int i, Error* error=0);

      long getLong(Error* error=0)const;
      void setLong(long i, Error* error=0);

      float getFloat(Error* error=0) const;
      void setFloat(float i, Error* error=0);

      double getDouble(Error* error=0) const;
      void setDouble(double i, Error* error=0);

};
#endif
