// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FLOATARRAYDATAITEM_H
#define FLOATARRAYDATAITEM_H
#include "Copyright.h"

#include "NumericArrayDataItem.h"
#include "ShallowArray.h"

class FloatArrayDataItem : public NumericArrayDataItem
{
   private:
      NumericArrayDataItem & assign(const NumericArrayDataItem &);
      std::vector<float> *_data;

   public:
      static char const * _type;

      // Constructors
      FloatArrayDataItem();
      FloatArrayDataItem(const FloatArrayDataItem& DI);
      FloatArrayDataItem(std::vector<int> const & dimensions);
      FloatArrayDataItem(ShallowArray<float> const & data);
      ~FloatArrayDataItem();

      // Utility methods
      void setDimensions(std::vector<int> const &dimensions);
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      // Array Methods
      std::string getString(Error* error=0) const;
      std::string getString(std::vector<int> coords, Error* error=0) const;
      void setString(std::vector<int> coords, std::string value, Error* error=0);

      bool getBool(std::vector<int> coords, Error* error=0) const;
      void setBool(std::vector<int> coords, bool value, Error* error=0);

      char getChar(std::vector<int> coords, Error* error=0) const;
      void setChar(std::vector<int> coords, char value, Error* error=0);

      unsigned char getUnsignedChar(std::vector<int> coords, Error* error=0) const;
      void setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error=0);

      signed char getSignedChar(std::vector<int> coords, Error* error=0) const;
      void setSignedChar(std::vector<int> coords, signed char value, Error* error=0);

      short getShort(std::vector<int> coords, Error* error=0) const;
      void setShort(std::vector<int> coords, short value, Error* error=0);

      unsigned short getUnsignedShort(std::vector<int> coords, Error* error=0) const;
      void setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error=0);

      int getInt(std::vector<int> coords, Error* error=0) const;
      void setInt(std::vector<int> coords, int value, Error* error=0);

      unsigned int getUnsignedInt(std::vector<int> coords, Error* error=0) const;
      void setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error=0);

      long getLong(std::vector<int> coords, Error* error=0) const;
      void setLong(std::vector<int> coords, long value, Error* error=0);

      float getFloat(std::vector<int> coords, Error* error=0) const;
      void setFloat(std::vector<int> coords, float value, Error* error=0);
      const std::vector<float>*  getFloatVector() const;
      std::vector<float>* getModifiableFloatVector();

      double getDouble(std::vector<int> coords, Error* error=0) const;
      void setDouble(std::vector<int> coords, double value, Error* error=0);

};
#endif
