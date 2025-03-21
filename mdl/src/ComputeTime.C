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

#include "ComputeTime.h"
#include "ComputeTimeType.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include "SyntaxErrorException.h"
#include "NotFoundException.h"
#include "DataType.h"
#include "InterfaceImplementorBase.h"
#include <memory>
#include <string>
#include <sstream>

ComputeTime::ComputeTime(const std::string& name, std::unique_ptr<ComputeTimeType>&& computeTimeType,
	     const std::vector<std::string>& pvn)
   : _name(name), _packedVariableNames(pvn)
{
   _computeTimeType = computeTimeType.release();
}

ComputeTime::ComputeTime(const ComputeTime& rv)
   : _name(rv._name), //_computeTimeType(0), 
     _packedVariableNames(rv._packedVariableNames), 
     _packedVariables(rv._packedVariables)
{
   copyOwnedHeap(rv);
}

ComputeTime& ComputeTime::operator=(const ComputeTime& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _name = rv._name;
      _packedVariableNames = rv._packedVariableNames;
      _packedVariables = rv._packedVariables;
   }
   return *this;
}

ComputeTime::~ComputeTime()
{
   destructOwnedHeap();
}

std::string ComputeTime::getGenerateString() const
{
   std::ostringstream os;
   os << getType() << " " << getName() << "(";

   std::vector<const DataType*>::const_iterator it, 
      end = _packedVariables.end();
   bool first = true;
   for (it = _packedVariables.begin(); it != end; ++it) {
      if (!first) {
	 os << ", ";
      } else {
	 first = false;
      }
      os << (*it)->getName();
   }

   os << ")";

   return os.str();
}

void ComputeTime::generateVirtualUserMethod(Class& c) const
{
   generateInternalUserMethod(c, true);
}

void ComputeTime::generateUserMethod(Class& c) const
{
   generateInternalUserMethod(c, false);
}

void ComputeTime::generateInternalUserMethod(Class& c, bool pureVirtual) const
{
   std::unique_ptr<Method> method(new Method(_name, "void"));
   method->setVirtual();
   method->setPureVirtual(pureVirtual);
   c.addMethod(std::move(method));
}    

void ComputeTime::generateInstanceComputeTimeMethod(
   Class& c, const std::string& instanceType, 
   const std::string& componentType) const
{
   _computeTimeType->generateInstanceComputeTimeMethod(c, _name, instanceType, 
					   componentType);
}

void ComputeTime::copyOwnedHeap(const ComputeTime& rv)
{
   if (rv._computeTimeType) {
      std::unique_ptr<ComputeTimeType> dup;
      rv._computeTimeType->duplicate(std::move(dup));
      _computeTimeType = dup.release();
   } else {
      _computeTimeType = 0;
   }
}

void ComputeTime::destructOwnedHeap()
{
   delete _computeTimeType;
}

std::string ComputeTime::getType() const
{
   return getInternalType() + "ComputeTime " + _computeTimeType->getType();
}

std::string ComputeTime::getWorkUnitsMethodBody(
   const std::string& tab, const std::string& workUnits,
   const std::string& instanceType, const std::string& componentType) const
{
   return _computeTimeType->getWorkUnitsMethodBody(tab, workUnits, instanceType,
					     _name, componentType);
}

std::string ComputeTime::getInitializeComputeTimeMethodBody() const
{
   return TAB + "initializeComputeTime(\"" + getName() + "\", \"" + 
      getInternalType() + "\");\n";
}

void ComputeTime::setPackedVariables(const InterfaceImplementorBase& base)
{
   const DataType* tmp;
   std::vector<std::string>::iterator it, end = _packedVariableNames.end();
   for(it = _packedVariableNames.begin(); it != end; ++it) {
      try {
	 tmp = base.getInstances().getMember(*it);
	 if (! base.isMemberToInterface(*tmp)) {
	    std::ostringstream os;
	    os << "For computeTime " << getName() << " " << *it << 
	       " does not implement an interface";
	    throw SyntaxErrorException(os.str());	    
	 }
	 _packedVariables.push_back(tmp);
      } catch (NotFoundException& e) {
	 std::ostringstream os;
	 os << "For computeTime " << getName() << " " << *it << 
	    " is not a instance variable";
	 throw SyntaxErrorException(os.str());
      }
   }
}

std::string ComputeTime::getAddVariableNamesForComputeTime(const std::string& tab) const
{
   if (!hasPackedVariables()) {
      return "";
   }
   std::vector<std::string>::const_iterator it, 
      end = _packedVariableNames.end();

   std::ostringstream os;
   os << tab << "if (" << PHASE << " == \"" << getName() << "\") {\n";
   for (it = _packedVariableNames.begin(); it != end; ++it) {
      os << tab << TAB << NAMESSET << ".insert(\"" << *it << "\");\n";
   }
   os << tab << "}\n";
   return os.str();
}
