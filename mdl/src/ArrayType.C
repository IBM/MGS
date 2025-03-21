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

#include "ArrayType.h"
#include "DataType.h"
#include "Constants.h"
#include "InternalException.h"
#include "Utility.h"
#include "Attribute.h"
#include "DataTypeAttribute.h"
#include "Class.h"
#include "AccessType.h"
#include <string>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>

ArrayType::ArrayType() 
   : DataType(), _type(0), _blockSize(0), _incrementSize(0)
{
}

ArrayType::ArrayType(DataType* dt) 
   : DataType(), _type(dt), _blockSize(0), _incrementSize(0)
{
}

ArrayType::ArrayType(const ArrayType& rv)
   : DataType(rv), _type(0), _blockSize(rv._blockSize),
     _incrementSize(rv._incrementSize)
{
   copyOwnedHeap(rv);
}

ArrayType& ArrayType::operator=(const ArrayType& rv)
{
   if (this != &rv) {
      DataType::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _blockSize = rv._blockSize;
      _incrementSize = rv._incrementSize;
   }
   return *this;
}

void ArrayType::duplicate(std::unique_ptr<DataType>&& rv) const
{
   rv.reset(new ArrayType(*this));
}

std::string ArrayType::getDataItemString() const
{
   return _type->getArrayDataItemString();
}

std::string ArrayType::getPrefixArrayType() const
{
   std::string arrayType;
   if (_type->isPointer()) {
      if (_type->shouldBeOwned()) {
	 if (_type->isBasic()) {
	    arrayType = "DeepPointer"; 
	 } else {
	    arrayType = "DuplicatePointer";
	 }    
      } else {
	 arrayType = "Shallow";
      }
   } else {
      arrayType = "Shallow";
   }
   return arrayType;
}

std::string ArrayType::getDescriptor() const
{
   std::ostringstream os;
   os << getPrefixArrayType() <<"Array< " << _type->getDescriptor();
   // !_type->shouldBeOwned() is due to these type of arrays having ptr in 
   // theit definition, * is not embedded in the type.
   if (_type->isPointer() && !_type->shouldBeOwned()) { 
      os << "*";
   }
   if (_blockSize > 0) {
      os  << ", " << _blockSize;
      if (_incrementSize > 0) {
	 os  << ", " << _incrementSize;
      }
   }

   os << " >";
   return os.str();
}

std::string ArrayType::getHeaderString(
   std::vector<std::string>& arrayTypeVec) const
{
   std::string retVal = _type->getHeaderString(arrayTypeVec);
   arrayTypeVec.push_back(getPrefixArrayType() + "Array");
   return retVal;
}

std::string ArrayType::getHeaderDataItemString() const
{
   return _type->getDataItemString();   
}

std::string ArrayType::getInitializerString(
   const std::string& diArg, int level, bool isIterator, bool forPSet) const
{
   std::string tab;
   setTabWithLevel(tab, level);
   if (isPointer() || !(_type->isLegitimateDataItem())) {
      if (forPSet) {
	 return getNotLegitimateDataItemString(tab);
      } else {
	 return "";
      }
   }
   std::ostringstream os;
   std::ostringstream diName;
   std::ostringstream diSize;
   std::string strippedName = getName();
   mdl::stripNameForCG(strippedName);
   diName << "" + PREFIX + "" << strippedName << "DI" << level;
   std::string referencedDiArg;
   if (isIterator) {
      referencedDiArg = "*" + diArg;
   } else {
      referencedDiArg = diArg;
   }   
   os << getArgumentCheckerString(diName.str(), referencedDiArg, level)
      << _type->getArrayInitializerString(diName.str(), getName(), level);

   if (isIterator) {
      os << tab << diArg + "++;\n";
   }
   return os.str();
}

// @TODO : implement this property, have to insantiate arrays and
// insert them to the higher level array.
std::string ArrayType::getArrayInitializerString(const std::string& name
						 , const std::string& arrayName
						 , int level) const
{
   std::cerr 
      << "\nCareful Arrays in arrays have not been implemented fully yet for MPI-communication.";
//  throw InternalException("Arrays in arrays have not been implemented yet.");
   return "/* Careful not Implemented... */\n";
}

bool ArrayType::shouldBeOwned() const
{   
   return false;
}

bool ArrayType::isSuitableForInterface() const
{
   return _type->isSuitableForInterface();
}


bool ArrayType::anythingToCopy()
{
   return (isPointer() && shouldBeOwned()) || _type->anythingToCopy();
}

ArrayType::~ArrayType() 
{
   destructOwnedHeap();
}

void ArrayType::copyOwnedHeap(const ArrayType& rv)
{
   if (rv._type) {
      std::unique_ptr<DataType> dup;
      rv._type->duplicate(std::move(dup));
      _type = dup.release();
   }
}

void ArrayType::destructOwnedHeap()
{
   delete _type;
}

void ArrayType::addProxyAttribute(std::unique_ptr<Class>&& instance) const
{
   std::unique_ptr<DataType> dup;
   duplicate(std::move(dup));
   dup->setPointer(false);
   if (_type->isPointer()) {
      ArrayType* dup2 = new ArrayType(*this);
      dup2->setPointer(false);
      dup2->_type->setPointer(false);
      dup2->setName(PREFIX + getName());
      
      std::unique_ptr<DataType> dup2ptr(dup2);
      std::unique_ptr<Attribute> att(new DataTypeAttribute(std::move(dup2ptr)));
      att->setAccessType(AccessType::PROTECTED);
      instance->addAttribute(std::move(att));
   }
   std::unique_ptr<Attribute> att(new DataTypeAttribute(std::move(dup)));
   att->setAccessType(AccessType::PROTECTED);
   instance->addAttribute(att);
}

void ArrayType::setArrayCharacteristics(
   unsigned blockSize, unsigned incrementSize)
{
   if (_type->isArray()) {
      _type->setArrayCharacteristics(blockSize, incrementSize);
   } else {
      _blockSize = blockSize;
      _incrementSize = incrementSize;
   }
}

void ArrayType::getSubStructDescriptors(std::set<std::string>& subStructTypes) const
{
  _type->getSubStructDescriptors(subStructTypes);
} 
