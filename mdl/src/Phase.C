// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Phase.h"
#include "PhaseType.h"
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

Phase::Phase(const std::string& name, std::auto_ptr<PhaseType>& phaseType,
	     const std::vector<std::string>& pvn)
   : _name(name), _packedVariableNames(pvn)
{
   _phaseType = phaseType.release();
}

Phase::Phase(const Phase& rv)
   : _name(rv._name), _phaseType(0), 
     _packedVariableNames(rv._packedVariableNames), 
     _packedVariables(rv._packedVariables)
{
   copyOwnedHeap(rv);
}

Phase& Phase::operator=(const Phase& rv)
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

Phase::~Phase()
{
   destructOwnedHeap();
}

std::string Phase::getGenerateString() const
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

void Phase::generateVirtualUserMethod(Class& c) const
{
   std::auto_ptr<Method> method(new Method(_name, "void"));
   method->setVirtual();
   method->setPureVirtual(true);
   method->addParameter("RNG& rng");
   c.addMethod(method);
}

void Phase::generateUserMethod(Class& c) const
{
   generateInternalUserMethod(c);
}

void Phase::generateInternalUserMethod(Class& c) const
{
   std::auto_ptr<Method> method(new Method(_name, "void"));
   //method->setInline();
   method->addParameter("RNG& rng");
   c.addMethod(method);
}    


void Phase::generateInstancePhaseMethod(
   Class& c, const std::string& instanceType, 
   const std::string& componentType) const
{
   _phaseType->generateInstancePhaseMethod(c, _name, instanceType, 
					   componentType);
}

void Phase::copyOwnedHeap(const Phase& rv)
{
   if (rv._phaseType) {
      std::auto_ptr<PhaseType> dup;
      rv._phaseType->duplicate(dup);
      _phaseType = dup.release();
   } else {
      _phaseType = 0;
   }
}

void Phase::destructOwnedHeap()
{
   delete _phaseType;
}

std::string Phase::getType() const
{
   return getInternalType() + "Phase " + _phaseType->getType();
}

std::string Phase::getWorkUnitsMethodBody(
   const std::string& tab, const std::string& workUnits,
   const std::string& instanceType, const std::string& componentType) const
{
   return _phaseType->getWorkUnitsMethodBody(tab, workUnits, instanceType,
					     _name, componentType);
}

std::string Phase::getInitializePhaseMethodBody() const
{
   return TAB + "initializePhase(\"" + getName() + "\", \"" + 
      getInternalType() + "\", " + (hasPackedVariables() ? "true" : "false") + ");\n";
}

void Phase::setPackedVariables(const InterfaceImplementorBase& base)
{
   const DataType* tmp;
   std::vector<std::string>::iterator it, end = _packedVariableNames.end();
   for(it = _packedVariableNames.begin(); it != end; ++it) {
      try {
	 tmp = base.getInstances().getMember(*it);
	 if (! base.isMemberToInterface(*tmp)) {
	    std::ostringstream os;
	    os << "For phase " << getName() << " " << *it << 
	       " does not implement an interface";
	    throw SyntaxErrorException(os.str());	    
	 }
	 _packedVariables.push_back(tmp);
      } catch (NotFoundException& e) {
	 std::ostringstream os;
	 os << "For phase " << getName() << " " << *it << 
	    " is not a instance variable";
	 throw SyntaxErrorException(os.str());
      }
   }
}

std::string Phase::getAddVariableNamesForPhase(const std::string& tab) const
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
