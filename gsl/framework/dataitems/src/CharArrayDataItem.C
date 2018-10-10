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
#include "CharArrayDataItem.h"
#include "VolumeOdometer.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

// Type
const char* CharArrayDataItem::_type = "CHAR_ARRAY";

// Constructors
CharArrayDataItem::CharArrayDataItem()
{
   _data = new std::vector<char>;
}


CharArrayDataItem::~CharArrayDataItem()
{
   delete _data;
}


CharArrayDataItem::CharArrayDataItem(std::vector<int> const &dimensions)
{
   ArrayDataItem::_setDimensions(dimensions);
   unsigned size = getSize();
   _data = new std::vector<char>(size);
   _data->assign(size,0);
}

CharArrayDataItem::CharArrayDataItem(ShallowArray<char> const & data)
{
  std::vector<int> dimensions(1);
  unsigned size=dimensions[0]=data.size();
  ArrayDataItem::_setDimensions(dimensions);
  _data = new std::vector<char>(size);  
  ShallowArray<char>::const_iterator iter=data.begin(), end=data.end();
  std::vector<char>::iterator iter2=_data->begin();
  for (; iter!=end; ++iter, ++iter2) *iter2=*iter;
}

CharArrayDataItem::CharArrayDataItem(const CharArrayDataItem& DI)
{
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   _data = new std::vector<char>(*DI._data);
}


// Utility methods
void CharArrayDataItem::duplicate(std::unique_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new CharArrayDataItem(*this)));
}


NumericArrayDataItem& CharArrayDataItem::assign(const NumericArrayDataItem& DI)
{
   delete _data;
   ArrayDataItem::_setDimensions(*DI.getDimensions());
   unsigned size = getSize();
   _data = new std::vector<char>(size);
   std::vector<int> begin(size);
   begin.assign(size,0);
   VolumeOdometer odmtr(begin, _dimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() )
      setChar(coords,DI.getChar(coords));
   return(*this);
}


const char* CharArrayDataItem::getType() const
{
   return _type;
}


void CharArrayDataItem::setDimensions(std::vector<int> const &dimensions)
{
   std::unique_ptr<DataItem*> diap;
   unsigned size = getOffset(dimensions);
   std::vector<char> *dest = new std::vector<char>(size);
   dest->assign(dest->size(),0);

   // find overlap between old dimensions and new
   unsigned oldsize = _dimensions.size();
   unsigned newsize = dimensions.size();
   int minNumDimensions = (oldsize> newsize)?oldsize:newsize;
   std::vector<int> minDimensions(minNumDimensions);
   std::vector<int> begin(minNumDimensions);
   for(int i = 0;i<minNumDimensions;++i) {
      begin[i] = 0;
      minDimensions[i] = (_dimensions[i]>dimensions[i])? _dimensions[i]:dimensions[i];
   }

   // Copy appropriate values to new vector
   VolumeOdometer odmtr(begin, minDimensions);
   std::vector<int> & coords = odmtr.look();
   for (; !odmtr.isRolledOver(); odmtr.next() ) {
      unsigned sourceOffset = getOffset(coords);
      unsigned destOffset = getOffset(dimensions,coords);
      (*dest)[destOffset] = (*_data)[sourceOffset];
   }

   // replace old vector with new
   delete _data;
   _data = dest;
   ArrayDataItem::_setDimensions(dimensions);
}


// Array methods
std::string CharArrayDataItem::getString(Error* error) const
{
   std::string rval;
   std::ostringstream str_value;
   str_value<<"{";
   for (std::vector<char>::const_iterator iter = _data->begin(); iter != _data->end(); iter++) {
      if (iter != _data->begin()) str_value<<",";
      str_value<<(*iter);
   }
   str_value<<"}";
   rval = str_value.str();
   return rval;
}


std::string CharArrayDataItem::getString(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   std::string rval;
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   std::ostringstream str_value;
   str_value<<(*_data)[offset];
   rval = str_value.str();
   return rval;
}


void CharArrayDataItem::setString(std::vector<int> coords, std::string value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   double val;
   std::istringstream str_value(value);
   str_value>>val;
   char conv_val = (char)val;
   if (error) {
      if ((val > CHAR_MAX) || (val < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (val != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

bool CharArrayDataItem::getBool(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (bool)(*_data)[offset];
}


void CharArrayDataItem::setBool(std::vector<int> coords, bool value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (char)value;
}


/* * * */

char CharArrayDataItem::getChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (char)(*_data)[offset];
}


void CharArrayDataItem::setChar(std::vector<int> coords, char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (char)value;
}


const std::vector<char>* CharArrayDataItem::getCharVector() const
{
   return _data;
}


std::vector<char>* CharArrayDataItem::getModifiableCharVector()
{
   return _data;
}


/* * * */

unsigned char CharArrayDataItem::getUnsignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned char)(*_data)[offset];
}


void CharArrayDataItem::setUnsignedChar(std::vector<int> coords, unsigned char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   (*_data)[offset] = (char)value;
}


/* * * */

signed char CharArrayDataItem::getSignedChar(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   /*
   if (error) {
      if ((*_data)[offset] > SCHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   */
   return (signed char)(*_data)[offset];
}


void CharArrayDataItem::setSignedChar(std::vector<int> coords, signed char value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   /*
   if (error) {
      if (value < CHAR_MIN) *error = CONVERSION_OUT_OF_RANGE;
   }
   */
   (*_data)[offset] = (char)value;
}


/* * * */

short CharArrayDataItem::getShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (short)(*_data)[offset];
}


void CharArrayDataItem::setShort(std::vector<int> coords, short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > CHAR_MAX) || (value < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (char)value;
}


/* * * */

unsigned short CharArrayDataItem::getUnsignedShort(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned short)(*_data)[offset];
}


void CharArrayDataItem::setUnsignedShort(std::vector<int> coords, unsigned short value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (value > CHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (char)value;
}


/* * * */

int CharArrayDataItem::getInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (int)(*_data)[offset];
}


void CharArrayDataItem::setInt(std::vector<int> coords, int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > CHAR_MAX) || (value < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (char)value;
}


/* * * */

unsigned int CharArrayDataItem::getUnsignedInt(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (unsigned int)(*_data)[offset];
}


void CharArrayDataItem::setUnsignedInt(std::vector<int> coords, unsigned int value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if (value > CHAR_MAX) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (char)value;
}


/* * * */

long CharArrayDataItem::getLong(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (long)(*_data)[offset];
}


void CharArrayDataItem::setLong(std::vector<int> coords, long value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   if (error) {
      if ((value > CHAR_MAX) || (value < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
   }
   (*_data)[offset] = (char)value;
}


/* * * */

float CharArrayDataItem::getFloat(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (float)(*_data)[offset];
}


void CharArrayDataItem::setFloat(std::vector<int> coords, float value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   char conv_val = (char)value;
   if (error) {
      if ((value > CHAR_MAX) || (value < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (float)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */

double CharArrayDataItem::getDouble(std::vector<int> coords, Error* error) const
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   return (double)(*_data)[offset];
}


void CharArrayDataItem::setDouble(std::vector<int> coords, double value, Error* error)
{
   unsigned offset = getOffset(coords);
   if (offset>_data->size()) {
      std::cerr<<"CharArrayDataItem: coords out of range!"<<std::endl;
      exit(-1);
   }
   char conv_val = (char)value;
   if (error) {
      if ((value > CHAR_MAX) || (value < CHAR_MIN)) *error = CONVERSION_OUT_OF_RANGE;
      else if (value != (double)conv_val) *error = LOSS_OF_PRECISION;
   }
   (*_data)[offset] = conv_val;
}


/* * * */
