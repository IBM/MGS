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

#ifndef UNSIGNEDSHORTARRAYDATAITEM_H
#define UNSIGNEDSHORTARRAYDATAITEM_H
#include "Copyright.h"

#include "NumericArrayDataItem.h"
#include "ShallowArray.h"

class UnsignedShortArrayDataItem : public NumericArrayDataItem
{
   private:
      NumericArrayDataItem & assign(const NumericArrayDataItem &);
      std::vector<unsigned short> *_data;

   public:
      static const char* _type;

      // Constructors
      UnsignedShortArrayDataItem();
      UnsignedShortArrayDataItem(const UnsignedShortArrayDataItem& DI);
      UnsignedShortArrayDataItem(std::vector<int> const & dimensions);
      UnsignedShortArrayDataItem(ShallowArray<unsigned short> const & data);
      ~UnsignedShortArrayDataItem();

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
      const std::vector<unsigned short>* getUnsignedShortVector() const;
      std::vector<unsigned short>* getModifiableUnsignedShortVector();

      int getInt(std::vector<int> coords, Error* error=0) const;
      void setInt(std::vector<int> coords, int value, Error* error=0);

      unsigned int getUnsignedInt(std::vector<int> coords, Error* error=0) const;
      void setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error=0);

      long getLong(std::vector<int> coords, Error* error=0) const;
      void setLong(std::vector<int> coords, long value, Error* error=0);

      float getFloat(std::vector<int> coords, Error* error=0) const;
      void setFloat(std::vector<int> coords, float value, Error* error=0);

      double getDouble(std::vector<int> coords, Error* error=0) const;
      void setDouble(std::vector<int> coords, double value, Error* error=0);

};
#endif
