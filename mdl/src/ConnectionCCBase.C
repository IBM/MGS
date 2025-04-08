// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-14-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "ConnectionCCBase.h"
#include "CompCategoryBase.h"
#include "RegularConnection.h"
#include "Generatable.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include "UserFunctionCall.h"
#include "BaseClass.h"
#include "ArrayType.h"
#include "Utility.h"
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <sstream>
#include <iostream>

ConnectionCCBase::ConnectionCCBase(const std::string& fileName) 
   : CompCategoryBase(fileName), _userFunctions(0), _predicateFunctions(0)
{
}

ConnectionCCBase::ConnectionCCBase(const ConnectionCCBase& rv)
   : CompCategoryBase(rv) 
{
   copyOwnedHeap(rv);
}  

ConnectionCCBase& ConnectionCCBase::operator=(const ConnectionCCBase& rv)
{
   if (this != &rv) {
      CompCategoryBase::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}

void ConnectionCCBase::addConnection(std::unique_ptr<RegularConnection>&& con) 
{
   _connections.push_back(con.release());
}

ConnectionCCBase::~ConnectionCCBase() 
{
   destructOwnedHeap();
}

std::string ConnectionCCBase::generateExtra() const
{
   std::ostringstream os;
   os << CompCategoryBase::generateExtra();
   if (_userFunctions) {
      os << "\n";
      std::vector<UserFunction*>::const_iterator it, 
	 end = _userFunctions->end();
      for (it = _userFunctions->begin(); it != end; ++it) {
	 os << (*it)->getString() << "\n";
      }
   }
   if (_predicateFunctions) {
      os << "\n";
      std::vector<PredicateFunction*>::const_iterator it, 
	 end = _predicateFunctions->end();
      for (it = _predicateFunctions->begin(); it != end; ++it) {
	 os << (*it)->getString() << "\n";
      }
   }
   if (_connections.size() > 0) {
      os << "\n";
      std::vector<RegularConnection*>::const_iterator it, 
	 end = _connections.end();
      for (it = _connections.begin(); it != end; it++) {
	 os << (*it)->getString() << "\n";
      }
   }
   return os.str();
}

void ConnectionCCBase::destructOwnedHeap() 
{
   std::vector<RegularConnection*>::iterator it, end = _connections.end();
   for (it = _connections.begin(); it != end; it++) {
      delete *it;
   }
   _connections.clear();
   if (_userFunctions) {
      std::vector<UserFunction*>::iterator it, 
	 end = _userFunctions->end();
      for (it = _userFunctions->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _userFunctions;
      _userFunctions = 0;
   }
   if (_predicateFunctions) {
      std::vector<PredicateFunction*>::iterator it, 
	 end = _predicateFunctions->end();
      for (it = _predicateFunctions->begin(); it != end; ++it) {
	 delete *it;
      }
      delete _predicateFunctions;
      _predicateFunctions = 0;
   }
}

void ConnectionCCBase::copyOwnedHeap(const ConnectionCCBase& rv) 
{
   std::vector<RegularConnection*>::const_iterator it, 
      end = rv._connections.end();
   std::unique_ptr<RegularConnection> dup;   
   for (it = rv._connections.begin(); it != end; it++) {
      (*it)->duplicate(std::move(dup));
      _connections.push_back(dup.release());
   }
   if (rv._userFunctions) {
      _userFunctions = new std::vector<UserFunction*>();
      std::vector<UserFunction*>::const_iterator it
	 , end = rv._userFunctions->end();
      std::unique_ptr<UserFunction> dup;   
      for (it = rv._userFunctions->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _userFunctions->push_back(dup.release());
      }
   } else {
      _userFunctions = 0;
   }
   if (rv._predicateFunctions) {
      _predicateFunctions = new std::vector<PredicateFunction*>();
      std::vector<PredicateFunction*>::const_iterator it
	 , end = rv._predicateFunctions->end();
      std::unique_ptr<PredicateFunction> dup;   
      for (it = rv._predicateFunctions->begin(); it != end; ++it) {
	 (*it)->duplicate(std::move(dup));
	 _predicateFunctions->push_back(dup.release());
      }
   } else {
      _predicateFunctions = 0;
   }
}

std::string ConnectionCCBase::getAddPreEdgeFunctionBody() const
{
   return getAddConnectionFunctionBody(RegularConnection::_EDGE, 
				       RegularConnection::_PRE);
}

std::string ConnectionCCBase::getAddPreNodeFunctionBody() const
{
   return getAddConnectionFunctionBody(RegularConnection::_NODE, 
				       RegularConnection::_PRE);
}
std::string ConnectionCCBase::getAddPreNode_DummyFunctionBody() const
{
   bool dummy=1;
   return getAddConnectionFunctionBody(RegularConnection::_NODE, 
				       RegularConnection::_PRE, dummy);
}

std::string ConnectionCCBase::getAddPreConstantFunctionBody() const
{
   return getAddConnectionFunctionBody(RegularConnection::_CONSTANT, 
				       RegularConnection::_PRE);
}

std::string ConnectionCCBase::getAddPreVariableFunctionBody() const
{
   return getAddConnectionFunctionBody(RegularConnection::_VARIABLE, 
				       RegularConnection::_PRE);
}

/* add the new argument 'dummy' 
 *  as this is evoked by 'getAddConnectionFunction(...)'
 *  to add the body to :addPreNode_Dummy(...)
 */
std::string ConnectionCCBase::getAddConnectionFunctionBodyExtra(
   Connection::ComponentType componentType, 
   Connection::DirectionType directionType,
   const std::string& componentName, const std::string& psetType, 
   const std::string& psetName,
   bool dummy
   ) const
{
   std::ostringstream os;
   std::string subBody = "";
   
   if (dummy)
   {
      // the parameters that will be passed to Predicate and User functions
      std::string functionParameters;
      functionParameters += "\"" + Connection::getStringForDirectionType(
	    directionType) + "\", "
	 + "\"" + Connection::getStringForComponentType(componentType) + "\", "
	 + Connection::getParametersForComponentType(componentType) + ", " 
	 + Connection::getParametersForDirectionType(directionType);
      std::vector<const RegularConnection*> connections;   
      std::vector<RegularConnection*>::const_iterator it, 
	 end = _connections.end();
      for (it = _connections.begin(); it != end; ++it) {
	 if ((componentType == (*it)->getComponentType()) &&
	       (directionType == (*it)->getDirectionType())) {
	    connections.push_back(*it);
	 }
      }
      std::set<std::string> interfaceCasts;
      std::vector<const RegularConnection*>::const_iterator it2, 
	 end2 = connections.end();
      // evaluation of predicates
      std::set<std::string> predicateFunctions;
      for (it2 = connections.begin(); it2 != end2; ++it2) {
	 (*it2)->getFunctionPredicateNames(predicateFunctions);
      }   
      // ConnectionCodes
      subBody += TAB + "bool noPredicateMatch= true; \n";
      subBody += TAB + "bool matchPredicateAndCast= false; \n";
      for (it2 = connections.begin(); it2 != end2; ++it2) {
	 std::set<std::string> tmpInterfaceCasts= 
	    (*it2)->getInterfaceCasts(componentName);
	 interfaceCasts.insert(
	       tmpInterfaceCasts.begin(), tmpInterfaceCasts.end());

	 if (isSupportedMachineType(MachineType::GPU) and componentType == RegularConnection::_NODE)
	 {
	    subBody += STR_GPU_CHECK_START;
	    subBody += "// CPU-GPU code\n";
	    subBody += (*it2)->getConnectionCode(getName(), 
		  functionParameters, MachineType::GPU, dummy) + "\n"; 
	    subBody += "#else\n";
	 }
	 /* CPU code */
	 subBody += "// CPU-only code\n";
	 subBody += TAB + "assert(0);\n";
	 //subBody += (*it2)->getConnectionCode(getName(), 
	 //      functionParameters) + "\n"; 
	 if (isSupportedMachineType(MachineType::GPU) and componentType == RegularConnection::_NODE )
	 {
	    subBody += STR_GPU_CHECK_END;
	 }
      }
      std::set<std::string>::iterator it3, end3 = interfaceCasts.end();
      //for (it3 = interfaceCasts.begin(); it3 != end3; ++it3) {
      //   os << TAB << *it3;
      //}
      if ((subBody != "") || predicateFunctions.size() > 0) { 
	 os << TAB << psetType << "* " << psetName << " = dynamic_cast <" 
	    << psetType << "*>(" << PREFIX << "pset);\n";
      }
      end3 = predicateFunctions.end();
      for (it3 = predicateFunctions.begin(); it3 != end3; ++it3) {
	 os << TAB << "bool " << PREDICATEFUNCTIONPREFIX << *it3 << " = " 
	    << *it3 << "(" << functionParameters << ");\n";
      }   
      if ((subBody != "") || predicateFunctions.size() > 0) { 
	 os << subBody;
      }

   }
   else{
      // the parameters that will be passed to Predicate and User functions
      std::string functionParameters;
      functionParameters += "\"" + Connection::getStringForDirectionType(
	    directionType) + "\", "
	 + "\"" + Connection::getStringForComponentType(componentType) + "\", "
	 + Connection::getParametersForComponentType(componentType) + ", " 
	 + Connection::getParametersForDirectionType(directionType);
      std::vector<const RegularConnection*> connections;   
      std::vector<RegularConnection*>::const_iterator it, 
	 end = _connections.end();
      for (it = _connections.begin(); it != end; ++it) {
	 if ((componentType == (*it)->getComponentType()) &&
	       (directionType == (*it)->getDirectionType())) {
	    connections.push_back(*it);
	 }
      }
      std::set<std::string> interfaceCasts;
      std::vector<const RegularConnection*>::const_iterator it2, 
	 end2 = connections.end();
      // evaluation of predicates
      std::set<std::string> predicateFunctions;
      for (it2 = connections.begin(); it2 != end2; ++it2) {
	 (*it2)->getFunctionPredicateNames(predicateFunctions);
      }   
      // ConnectionCodes
      subBody += TAB + "bool noPredicateMatch= true; \n";
      subBody += TAB + "bool matchPredicateAndCast= false; \n";
      for (it2 = connections.begin(); it2 != end2; ++it2) {
	 std::set<std::string> tmpInterfaceCasts= 
	    (*it2)->getInterfaceCasts(componentName);
	 interfaceCasts.insert(
	       tmpInterfaceCasts.begin(), tmpInterfaceCasts.end());

	 if (isSupportedMachineType(MachineType::GPU) and componentType == RegularConnection::_NODE)
	 {
	    subBody += STR_GPU_CHECK_START;
	    subBody += (*it2)->getConnectionCode(getName(), 
		  functionParameters, MachineType::GPU) + "\n"; 
	    subBody += "#else\n";
	 }
	 subBody += (*it2)->getConnectionCode(getName(), 
	       functionParameters) + "\n"; 
	 if (isSupportedMachineType(MachineType::GPU) and componentType == RegularConnection::_NODE )
	 {
	    subBody += STR_GPU_CHECK_END;
	 }
      }
      std::set<std::string>::iterator it3, end3 = interfaceCasts.end();
      for (it3 = interfaceCasts.begin(); it3 != end3; ++it3) {
	 os << TAB << *it3;
      }
      if ((subBody != "") || predicateFunctions.size() > 0) { 
	 os << TAB << psetType << "* " << psetName << " = dynamic_cast <" 
	    << psetType << "*>(" << PREFIX << "pset);\n";
      }
      end3 = predicateFunctions.end();
      for (it3 = predicateFunctions.begin(); it3 != end3; ++it3) {
	 os << TAB << "bool " << PREDICATEFUNCTIONPREFIX << *it3 << " = " 
	    << *it3 << "(" << functionParameters << ");\n";
      }   
      if ((subBody != "") || predicateFunctions.size() > 0) { 
	 os << subBody;
      }

   }
   return os.str();
}

void ConnectionCCBase::addExtraInstanceBaseMethods(Class& instance) const
{
   CompCategoryBase::addExtraInstanceBaseMethods(instance);

   instance.addClass("Constant");
   instance.addHeader("\"CustomString.h\"");
   instance.addHeader("\"Service.h\"");
   instance.addHeader("\"SyntaxErrorException.h\"");
   instance.addHeader("\"VariableDescriptor.h\"");

   // Classes that make a connection also accept services.
   // These are connections extablished using publishers.
   // Modified, the base classes inherit from ServiceAcceptor now.
//    std::unique_ptr<BaseClass> base(new BaseClass("ServiceAcceptor"));
//    instance.addBaseClass(base);
//    instance.addHeader("\"ServiceAcceptor.h\"");

   // add acceptService method for ServiceAcceptor base class
   std::unique_ptr<Method> acceptServiceMethod(new Method("acceptService", 
							"void"));
   acceptServiceMethod->setVirtual();
   acceptServiceMethod->addParameter("Service* service");
   acceptServiceMethod->addParameter("const std::string& name");
   acceptServiceMethod->setFunctionBody(getAcceptServiceBody());
   instance.addMethod(std::move(acceptServiceMethod));

   addExtraInstanceMethodsCommon(instance, true);

   std::vector<RegularConnection*>::const_iterator it, 
      end = _connections.end();
   for (it = _connections.begin(); it != end; ++it) {
      (*it)->addInterfaceHeaders(instance);
   }
}

void ConnectionCCBase::addExtraInstanceProxyMethods(Class& instance) const
{
   CompCategoryBase::addExtraInstanceProxyMethods(instance);

   instance.addClass("Constant");
   instance.addHeader("\"CustomString.h\"");
   instance.addHeader("<cassert>");

   // @TODO Implement acceptService method
   // add acceptService method for ServiceAcceptor base class
//    std::unique_ptr<Method> acceptServiceMethod(new Method("acceptService", 
// 							"void"));
//    acceptServiceMethod->setVirtual();
//    acceptServiceMethod->addParameter("Service* service");
//    acceptServiceMethod->addParameter("const std::string& name");
//    acceptServiceMethod->setFunctionBody(
//       TAB + FALSEASSERT);
//    instance.addMethod(acceptServiceMethod);


#if 0
   // add acceptService method for ServiceAcceptor base class
   std::unique_ptr<Method> acceptServiceMethod(new Method("acceptService", 
							"void"));
   acceptServiceMethod->setVirtual();
   acceptServiceMethod->addParameter("Service* service");
   acceptServiceMethod->addParameter("const std::string& name");
   acceptServiceMethod->setFunctionBody(getAcceptServiceBody());
   instance.addMethod(acceptServiceMethod);
#endif
}

void ConnectionCCBase::addExtraInstanceMethods(Class& instance) const
{
   addExtraInstanceMethodsCommon(instance, false);
}

void ConnectionCCBase::addExtraInstanceMethodsCommon(Class& instance, 
					       bool pureVirtual) const
{
   if (_userFunctions) {
      std::vector<UserFunction*>::iterator it, 
	 end = _userFunctions->end();
      for (it = _userFunctions->begin(); it != end; ++it) {
	 (*it)->generateInstanceMethod(instance, pureVirtual, *this);
      }
   }
   if (_predicateFunctions) {
      std::vector<PredicateFunction*>::iterator it, 
	 end = _predicateFunctions->end();
      for (it = _predicateFunctions->begin(); it != end; ++it) {
	 (*it)->generateInstanceMethod(instance, pureVirtual, *this);
      }
   }
}

bool ConnectionCCBase::userFunctionCallExists(const std::string& name) const
{
   if (_userFunctions) {
      std::vector<UserFunction*>::const_iterator it, 
	 end = _userFunctions->end();
      for (it = _userFunctions->begin(); it != end; ++it) {
	 if (name == (*it)->getName()) {
	    return true;
	 }
      }      
   } 
   return false;
}

bool ConnectionCCBase::predicateFunctionCallExists(
   const std::string& name) const
{
   if (_predicateFunctions) {
      std::vector<PredicateFunction*>::const_iterator it, 
	 end = _predicateFunctions->end();
      for (it = _predicateFunctions->begin(); it != end; ++it) {
	 if (name == (*it)->getName()) {
	    return true;
	 }
      }      
   } 
   return false;
}

void ConnectionCCBase::setUserFunctions(
   std::unique_ptr<std::vector<UserFunction*> >& userFunction) 
{
   delete _userFunctions;
   _userFunctions = userFunction.release();
}

void ConnectionCCBase::setPredicateFunctions(
   std::unique_ptr<std::vector<PredicateFunction*> >& predicateFunction) 
{
   delete _predicateFunctions;
   _predicateFunctions = predicateFunction.release();
}

std::string ConnectionCCBase::getAcceptServiceBody() const
{
   std::ostringstream os;
   MemberContainer<DataType>::const_iterator it, end = getInstances().end();   
   for (it = getInstances().begin(); it != end; ++it) {
      if (it->second->isArray()) {
	 const ArrayType* array = dynamic_cast<const ArrayType*>(it->second);
	 os << getArrayConnectionAccept(
	    array, array->getType()->isPointer());
      } else { 
	 os << getNonArrayConnectionAccept(
	    it->second, it->second->isPointer());
      }
   }
   os << TAB << "throw SyntaxErrorException(name + \" is not an " 
      << "acceptable service\");\n";
   return os.str();
}

std::string ConnectionCCBase::getAcceptServiceBodyExtra() const
{
   return "";
}

std::string ConnectionCCBase::getNonArrayConnectionAccept(
   const DataType* elem, bool pointer) const
{
   std::ostringstream os;
   std::string local = PREFIX + "local";
   std::string localType = "GenericService< " + 
      elem->getDescriptor() + " >*";
   
   os << TAB << "if (name == \"" << elem->getName() << "\") {\n"
      << TAB << TAB << localType << " " << local 
      << " = dynamic_cast<" << localType << ">(service);\n"
      << TAB << TAB << "if (" << local << " == 0) {\n"
      << TAB << TAB << TAB << "throw SyntaxErrorException("
      << "\"Expected a " << elem->getDescriptor() << " service for " 
      << elem->getName() << "\");\n"
      << TAB << TAB << "}\n";

   if (isSupportedMachineType(MachineType::GPU))
   {
      os << STR_GPU_CHECK_START;
      os  << TAB << TAB << elem->getName(MachineType::GPU) << " = ";
      if (!pointer) {
	 os << "*";
      }
      os << local << "->getData();\n"
	 << "#else\n";
   }
   os  << TAB << TAB << elem->getName() << " = ";
   if (!pointer) {
      os << "*";
   }
   os << local << "->getData();\n";
   if (isSupportedMachineType(MachineType::GPU))
   {
      os << STR_GPU_CHECK_END;
   }
   os   << TAB << TAB << "return;\n"
      << TAB << "}\n";            
   return os.str();
}

std::string ConnectionCCBase::getArrayConnectionAccept(
   const ArrayType* elem, bool pointer) const
{
   std::ostringstream os;

   std::string local = PREFIX + "local";
   std::string descriptor = elem->getType()->getDescriptor();
   std::string localType = "GenericService< " + descriptor + " >*";
   
   /* added by Jizhu Lu on 12/17/2005 to fix a bug */
   std::string insert;
   if (elem->isPointer()) {
      insert = "->insert(";
   } else {
      insert = ".insert(";
   }
   /****/

   os << TAB << "if (name == \"" << elem->getName() << "\") {\n"
      << TAB << TAB << localType << " " << local 
      << " = dynamic_cast<" << localType << ">(service);\n"
      << TAB << TAB << "if (" << local << " == 0) {\n"
      << TAB << TAB << TAB << "throw SyntaxErrorException("
      << "\"Expected a " << descriptor << " service for "
      << elem->getName() << "\");\n"
      << TAB << TAB << "}\n";

   if (isSupportedMachineType(MachineType::GPU))
   {
      os << STR_GPU_CHECK_START;
      os << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	 //<< TAB << REF_CC_OBJECT << "->" << elem->getName(MachineType::GPU) 
	 << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME <<elem->getName() 
	 << "[" << REF_INDEX << "]"
	 << insert;
      if (!pointer) {
	 os << "*";
      }
      os << local 
	 << "->getData());\n";
      os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n";
      std::string tmpVarName = PREFIX_MEMBERNAME + elem->getName() + "_index"; 
      os << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << elem->getName() << "_num_elements["
	 << REF_INDEX << "] +=1;\n"
         << TAB << "int " << tmpVarName << " = " 
	 << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME <<elem->getName() << "_offset["  
	 << REF_INDEX << "] + " << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME 
	 << elem->getName() << "_num_elements[" << REF_INDEX << "]-1;\n"
          << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << elem->getName() 
	  << ".replace(" << tmpVarName << ", "; 
      if (!pointer) {
	 os << "*";
      }
      os  << local 
	  << "->getData());\n";
          //<< TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << elem->getName() 
	  //<< "[" << tmpVarName << "] = " 
	  //<< local 
	  //<< "->getData();\n";
      os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
          << TAB <<REF_CC_OBJECT <<  "->" << PREFIX_MEMBERNAME << elem->getName() << "_num_elements[" 
	  << REF_INDEX << "] +=1;\n"
          << TAB << "int " << tmpVarName << " = " << REF_INDEX << " * " << REF_CC_OBJECT << "->" 
	  << PREFIX_MEMBERNAME << elem->getName() << "_max_elements + " 
	  << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << elem->getName() << "_num_elements["
	  << REF_INDEX << "]-1;\n"
          << TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << elem->getName() << ".replace(" 
	  << tmpVarName << ", "; 
      if (!pointer) {
	 os << "*";
      }
      os << local 
	  << "->getData());\n";
          //<< TAB << REF_CC_OBJECT << "->" << PREFIX_MEMBERNAME << elem->getName() << "[" 
	  //<< tmpVarName << "] = " 
	  //<< local 
	  //<< "->getData();\n";
      os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	  << TAB << "assert(0);\n"
	  << "#endif\n";

	 os << "#else\n";
   }
   os   << TAB << TAB << elem->getName() << insert;
   if (!pointer) {
      os << "*";
   }
   os << local 
      << "->getData());\n";
   if (isSupportedMachineType(MachineType::GPU))
   {
      os << STR_GPU_CHECK_END;
   }
   os << TAB << TAB << "return;\n"
      << TAB << "}\n";            
   return os.str();
}
