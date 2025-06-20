// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SIGNEDCHARDATAITEM_H
#define SIGNEDCHARDATAITEM_H
#include "Copyright.h"

#include "NumericDataItem.h"

class SignedCharDataItem : public NumericDataItem
{
   private:
      NumericDataItem & assign(const NumericDataItem &);
      signed char _data;

   public:
      static const char* _type;

      // Constructors
      SignedCharDataItem(signed char data = 0);
      SignedCharDataItem(const SignedCharDataItem& DI);

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
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
