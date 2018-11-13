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

#include "CustomAttribute.h"
#include "AccessType.h"
#include "DataType.h"
#include "Constants.h"
#include <string>
#include <vector>
#include <sstream>
#include <cassert>

CustomAttribute::CustomAttribute()
   : Attribute(), _name(""), _type(""), _basic(false), _pointer(false), 
     _owned(false), _cArray(false), _customDelete(false), _cArraySize(""), _reference(false), 
     _parameterName("")
{
}

CustomAttribute::CustomAttribute(const std::string& name, 
				 const std::string& type
	  , AccessType accessType)
   : Attribute(accessType), _name(name), _type(type), _basic(false), 
     _pointer(false), _owned(false), _cArray(false), _customDelete(false), _cArraySize(""), 
     _reference(false), _parameterName("")
{
}

void CustomAttribute::duplicate(std::auto_ptr<Attribute>& dup) const
{
   dup.reset(new CustomAttribute(*this));
}

CustomAttribute::~CustomAttribute()
{
}

std::string CustomAttribute::getName() const
{
   return _name;
}

void CustomAttribute::setName(const std::string& name)
{
   _name = name;
}

std::string CustomAttribute::getType() const
{
   if (isReference()) {
      return _type + "&";
   } else {
      return _type;
   }
}

void CustomAttribute::setType(const std::string& type)
{
   _type = type;
}

bool CustomAttribute::isBasic() const
{
   return _basic;
}

void CustomAttribute::setBasic(bool basic)
{
   _basic = basic;
}

bool CustomAttribute::isPointer() const
{
   return _pointer;
}

void CustomAttribute::setPointer(bool pointer)
{
   _pointer = pointer;
}

bool CustomAttribute::isOwned() const
{
   return _owned;
}

void CustomAttribute::setOwned(bool owned)
{
   _owned = owned;
}

bool CustomAttribute::isCArray() const
{
   return _cArray;
}

void CustomAttribute::setCArray(bool cArray)
{
   _cArray = cArray;
}

std::string CustomAttribute::getCArraySize() const
{
   return _cArraySize;
}

void CustomAttribute::setCArraySize(const std::string& cArraySize)
{
   _cArraySize = cArraySize;
}

bool CustomAttribute::isReference() const
{
   return _reference;
}

void CustomAttribute::setReference(bool reference)
{
   _reference = reference;
}

const std::string& CustomAttribute::getInitializeString() const
{
   return _initializeString;
}

void CustomAttribute::setInitializeString(const std::string& init)
{
   _initializeString = init;
}

std::string CustomAttribute::getConstructorParameter(
   const std::string& className) const
{
   std::string retVal = "";
   if (getConstructorParameterNameExtra() != "") {
      retVal = getType();
      if (isPointer()) {
	 retVal += "*";
      }
      retVal += " " + getConstructorParameterNameExtra();
   } else if (isReference()) {
      retVal = getType() + " " + PREFIX + className + getName();
   }
   return retVal;
}

std::string CustomAttribute::getConstructorParameterName(
   const std::string& className) const
{
   std::string retVal = "";
   if (getConstructorParameterNameExtra() != "") {
      retVal = getConstructorParameterNameExtra();
   } else if (isReference()) {
      retVal = PREFIX + className + getName();
   }
   return retVal;
}

bool CustomAttribute::isDontCopy() const
{
   if (isCArray()) {
      // not implemented yet
      assert(0);
      return true;
   }
   if (isReference()) { // references don't get copied
      return true;
   }

   return false;
}

void CustomAttribute::fillInitializer(std::string& init) const
{
   if (_initializeString != "") {
      init = _initializeString;
   } else {
      Attribute::fillInitializer(init);
   }
}

std::string CustomAttribute::getDeleteString() 
{
  std::string rval="";
  if (_customDelete) {
    rval = _customDeleteString;
  }
  rval += Attribute::getDeleteString();
  return rval;
}

void CustomAttribute::setCustomDeleteString(std::string deleteString)
{
  _customDeleteString=deleteString;
  _customDelete=true;
}

std::string CustomAttribute::getCustomDeleteString() 
{
  return _customDeleteString;
}
