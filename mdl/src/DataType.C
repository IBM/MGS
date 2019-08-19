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

#include "DataType.h"
#include "Constants.h"
#include "Utility.h"
#include "Attribute.h"
#include "DataTypeAttribute.h"
#include "Class.h"
#include "AccessType.h"
#include "InternalException.h"
#include "Method.h"
#include "MacroConditional.h"
#include <string>
#include <memory>
#include <sstream>
#include <cassert>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <cxxabi.h>
#include <regex>

DataType::DataType() 
   : _pointer(false), _derived(false), _shared(false), _name(""), _comment("")
{
}

std::string DataType::getString() const
{
   std::ostringstream os;  
   if (_derived) {
      os << "Derived ";
   }
   os << getDescriptor();
   if (_pointer) {
      os << "*";
   }
   os << " " << getName();
   if (_comment != "") {
      os << " : " << '"' << _comment << '"';
   }
   return os.str();
}

std::string DataType::getTypeString() const
{
   std::ostringstream os;
   os << getDescriptor();
   if (_pointer) {
      os << "*";
   }
   return os.str();
}

std::string DataType::getCapitalDescriptor() const
{
   return getDescriptor();
}

std::string DataType::getHeaderString(std::vector<std::string>& arrayTypeVec) const
{
   // Basic types don't have any header strings.
   return "";
}

std::string DataType::getHeaderDataItemString() const
{
   assert(0);   
   return "";
}

std::string DataType::getInitializerString(const std::string& diArg, int level,
					   bool isIterator, bool forPSet) const
{
   std::string tab;
   setTabWithLevel(tab, level);
   if (!isLegitimateDataItem()) {
      if (forPSet) {
	 return getNotLegitimateDataItemString(tab);
      } else {
	 return "";
      }
   } 
   std::ostringstream os;
   std::ostringstream diNameStr;
   std::string strippedName = getName();
   mdl::stripNameForCG(strippedName);
   diNameStr << "" + PREFIX + "" << strippedName << "DI" << level;
   std::string diName = diNameStr.str();
   std::string referencedDiArg;
   std::string newName = "";
   if (isIterator) {
      referencedDiArg = "*" + diArg;
   } else {
      referencedDiArg = diArg;
   }   
   os << getArgumentCheckerString(diName, referencedDiArg, level)
      << tab << "else {\n" 
      // else has meaning if struct type is 
      // specialized int init from vector<DataItem*>
      << checkIfStruct(diName, tab, newName);
   // This looks silly, however, if I just use newName (which should be == to
   // diName, or the modified version, in other words always the 
   // desired value, newName is sometimes equal to "", due to a gcc bug.
   std::string newForDuplicate;
   if (newName != diName) {
      os << duplicateIfOwned(newName, tab, newForDuplicate);
   } else {
      os << duplicateIfOwned(diName, tab, newForDuplicate);
   }
   os << tab << TAB << getName() << " = "
      << getDataFromVariable(newForDuplicate)
      << ";\n"
      << tab << "}\n";
   if (isIterator) {
      os << tab << diArg + "++;\n";
   }
   return os.str();

}

std::string DataType::getPSetString(const std::string& diArg, bool first) const
{
   std::ostringstream os;
   os << TAB << TAB ;
   if (!first) {
      os << "else ";
   }
   os << "if (" << diArg << "->getName() == \"" << getName() << "\") {\n"
      << TAB << TAB << TAB << FOUND << " = true;\n"
      << getInitializerString(diArg + "->getDataItem()", 2, false, true)
      << TAB << TAB << "}\n";
   return os.str();
}

std::string DataType::checkIfStruct(const std::string& name, 
				    const std::string& tab,
				    std::string& newName) const
{
   newName = name;
   return "";
}

std::string DataType::duplicateIfOwned(const std::string& name, 
				       const std::string& tab,
				       std::string& newName) const
{
   if (shouldBeOwned()) {
      std::ostringstream os;
      newName = name + "ap";
      os << tab << TAB << "std::unique_ptr< " 
	 << getDescriptor() << " > " << newName << ";\n"
	 << tab << TAB << name << "->" << getDataItemFunctionString() 
	 << "->duplicate("
	 << newName << ");\n";
      return os.str();	 
   } else {
      newName = name;
   }
   return "";
}

std::string DataType::getDataFromVariable(const std::string& name) const
{
   std::ostringstream os;
   if (shouldBeOwned()) {
      os << name << ".release()";
   } else {
      os << name << "->" << getDataItemFunctionString();
   }
   return os.str();
}

std::string DataType::getDataItemString() const
{
   return getDescriptor() + "DataItem";
}

std::string DataType::getInitializationDataItemString() const
{
   return getDataItemString();
}

std::string DataType::getArrayDataItemString() const
{
   return "DataItemArrayDataItem";
}

std::string DataType::getArgumentCheckerString(const std::string& name,
					       const std::string& diArg,
					       int level) const
{
   std::string tab;
   setTabWithLevel(tab, level);
   std::ostringstream os;
   os << tab << getInitializationDataItemString() << "* " << name 
      << " = dynamic_cast<" << getInitializationDataItemString() 
      << "*>(" << diArg << ");\n" 
      << tab << "if (" << name << " == 0) {\n"
      << tab << TAB << "throw SyntaxErrorException(\"Expected a " 
      << getInitializationDataItemString();
   if (!strcmp(getName().c_str(), "")) { // means it is in an array  // modified by Jizhu Lu on 02/23/2006
      os << " as the array element\");\n";
   } else {
      os << " for " << getName() << "\");\n";
   }
   os << tab << "}\n";
   return os.str();
}

std::string DataType::getDataItemFunctionString() const
{
   return "get" + getDescriptor() + "()";
}

bool DataType::isLegitimateDataItem() const
{
   return !isPointer();
}

std::string DataType::getArrayInitializerString(const std::string& name,
						const std::string& arrayName,
						int level) const
{
   std::string tab;
   setTabWithLevel(tab, level);
   std::string nameVec = name + "Vec";
   std::string nameVecEntity = name + "Entity";
   std::string nameVecIt = nameVec + "It";
   std::string nameVecEnd = nameVec + "End";
   std::string newName = "";
   std::ostringstream body;
   body << tab << "std::vector<DataItem*>* " << nameVec << " = " 
	<< name << "->getModifiableDataItemVector();\n"
	<< tab << "std::vector<DataItem*>::iterator " << nameVecIt 
	<< ", " << nameVecEnd << " = "
	<< nameVec << "->end();\n"
	<< tab << "for (" << nameVecIt << " = " << nameVec << "->begin(); " 
	<< nameVecIt
	<< " != " << nameVecEnd << "; " << nameVecIt << "++) {\n"
	<< getArgumentCheckerString(nameVecEntity, "(*" + nameVecIt + ")", 
				    level + 1)
	<< checkIfStruct(nameVecEntity, tab, newName);
   // This looks silly, however, if I just use newName (which should be == to
   // nameVecEntity, or the modified version, in other words always the 
   // desired value, newName is sometimes equal to "", due to a gcc bug.
   std::string newForDuplicate;
   if (newName != nameVecEntity) {
      body << duplicateIfOwned(newName, tab, newForDuplicate);
   } else {
      body << duplicateIfOwned(nameVecEntity, tab, newForDuplicate);
   }
   body << tab << TAB << arrayName << ".insert(" 
      << getDataFromVariable(newForDuplicate)
      << ");\n"
      << tab << "}\n";
   return body.str();
}

std::string DataType::getCustomArrayInitializerString(
   const std::string& name, const std::string& arrayName, int level,
   const std::string& diTypeName, const std::string& dataTypeName) const
{
   std::string tab;
   setTabWithLevel(tab, level);
   std::string nameVec = name + "Vec";
   std::string nameVecIt = nameVec + "It";
   std::string nameVecEnd = nameVec + "End";
   std::ostringstream body;
   body << tab << "std::vector<" << dataTypeName << ">* " << nameVec << " = " 
	<< name << "->getModifiable" << diTypeName << "Vector();\n"
	<< tab << "std::vector<" << dataTypeName <<  ">::iterator " 
	<< nameVecIt << ", " << nameVecEnd << " = "
	<< nameVec << "->end();\n"
	<< tab << "for (" << nameVecIt << " = " << nameVec << "->begin(); " 
	<< nameVecIt << " != " << nameVecEnd << "; " << nameVecIt << "++) {\n"
	<< tab << TAB << arrayName << ".insert((" << getDescriptor() 
	<< ") *" << nameVecIt << ");\n"
	<< tab << "}\n";
   return body.str();
}

void DataType::setTabWithLevel(std::string& tab, int level) const
{
   tab = "";
   for (int i = 0; i < (level+1); i++) {
      tab += TAB;
   }
}

bool DataType::shouldBeOwned() const
{
   return false;
}

bool DataType::anythingToCopy()
{
   return isPointer() && shouldBeOwned();
}

std::string DataType::getServiceString(const std::string& tab) const
{
   return getServiceString(tab, MachineType::CPU);
}
std::string DataType::getServiceString(const std::string& tab, MachineType mach_type) const
{
   // No services for pointers that are not optional
   if (isPointer()) {
      return "";
   }
   std::string open_parenthesis="", close_parenthesis="";
   if (mach_type == MachineType::GPU)
   {
      open_parenthesis="(", close_parenthesis=")";
   }
   std::ostringstream os;
   os << tab << "if (" << SERVICEREQUESTED << " == \"" << getName() 
      << "\") {\n";

   if (mach_type == MachineType::GPU and ! _shared)
   {
      if (isArray())
      {
	 std::string  type = getTypeString(); 
	 std::string from = "ShallowArray<";
	 std::string to = "ShallowArray_Flat<";
	 type = type.replace(type.find(from),from.length(),to);
	 std::size_t start = type.find_first_of("<");
	 std::size_t last = type.find_first_of(">");
	 std::string element_datatype = type.substr(start+1, last-start-1);
	 type = type.replace(start+1, last-start-1, element_datatype + ", " + MEMORY_LOCATION);
	 os << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n";
	 os << tab << TAB << "rval = new GenericService< " << type
	    << " >(" << DATA << ", " << "&("; 
	 os<< open_parenthesis << DATA << "->";
	 if (_shared) {
	    os << "getNonConstSharedMembers().";
	 } 
	 os << GETCOMPCATEGORY_FUNC_NAME << "()->" << PREFIX_MEMBERNAME << getName() << close_parenthesis << "[" << DATA << "->" << REF_INDEX << "]" << ")" << ");\n";
	 os << "#else\n"
	    << " // ignore it as GenericService has >> operator that does not accept pointer\n"
	    << " // and we most likely won't use this service\n"
	    << "#endif\n";
	 std::string comment = "// ";
	 /*
	 os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n";
	 os << tab << TAB << "int offset = " << DATA << "->" << GETCOMPCATEGORY_FUNC_NAME << "()->" << PREFIX_MEMBERNAME 
	    << getName() << "_start_offset[" << DATA << "->" << REF_INDEX << "];\n"
	    //" + " << DATA << "->" << GETCOMPCATEGORY_FUNC_NAME << "()->" << PREFIX_MEMBERNAME 
	    << getName() << "_num_elements[" << DATA << "->" << REF_INDEX << "];\n";
	 os << tab << TAB << "rval = new GenericService< " << element_datatype << 
	    " >(" << DATA << ", &("  << open_parenthesis
	    << DATA << "->" << GETCOMPCATEGORY_FUNC_NAME << "()->" << PREFIX_MEMBERNAME 
	    << getName() << close_parenthesis << "[offset]));\n";
	 os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n";
	 os << tab << TAB << "int offset = " << DATA << "->" << REF_INDEX << " * " 
	    << DATA << "->" << GETCOMPCATEGORY_FUNC_NAME << "()->" << 
	    PREFIX_MEMBERNAME << getName() << "_max_elements;\n"; 
	    //<< " + " << DATA << "->" << GETCOMPCATEGORY_FUNC_NAME << "()->" 
	    //<< PREFIX_MEMBERNAME << getName() << "_num_elements[" << DATA << "->" << REF_INDEX << "];\n";
	 os << tab << TAB << "rval = new GenericService< " << element_datatype << 
	    " >(" << DATA << ", &("  << open_parenthesis
	    << DATA << "->" << GETCOMPCATEGORY_FUNC_NAME << "()->" << PREFIX_MEMBERNAME 
	    << getName() << close_parenthesis << "[offset]));\n";
	 os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n";
	 os << tab << TAB << "assert(0);\n";
	 os << "#endif\n";
	 */
      }else{
	 os << tab << TAB << "rval = new GenericService< " << getTypeString() 
	    << " >(" << DATA << ", " << "&("; 
	 os<< open_parenthesis << DATA << "->";
	 if (_shared) {
	    os << "getNonConstSharedMembers().";
	 } 
	 os << GETCOMPCATEGORY_FUNC_NAME << "()->" << PREFIX_MEMBERNAME << getName() << close_parenthesis << "[" << DATA << "->" << REF_INDEX << "]" << ")" << ");\n";
      }
   }
   else{
      os << tab << TAB << "rval = new GenericService< " << getTypeString() 
	 << " >(" << DATA << ", " << "&("; 
      os //<< open_parenthesis 
	 << DATA << "->";
      if (_shared) {
	 os << "getNonConstSharedMembers().";
      } 
      os << getName() << ")" << ");\n";
   }
   os   << tab << TAB << "_services.push_back(rval);\n"
      << tab << TAB << "return rval;\n"
      << tab << "}\n";
   return os.str();
}

std::string DataType::getOptionalServiceString(const std::string& tab) const
{
   std::ostringstream os;
   os << tab << "if (" << SERVICEREQUESTED << " == \"" << getName() 
      << "\") {\n"
      << tab << TAB << "rval = new GenericService< " << getDescriptor() 
      << " >(" << DATA << ", " << DATA << "->";
   if (_shared) {
      os << "getNonConstSharedMembers().";
   } 
   os << PREFIX << GETSERVICE << getName();
   os << "());\n"
      << tab << TAB << "_services.push_back(rval);\n"
      << tab << TAB << "return rval;\n"
      << tab << "}\n";
   return os.str();
}

std::string DataType::getServiceNameString(const std::string& tab,
				     MachineType mach_type
      ) const
{
   return getServiceInfoString(tab, getName(), mach_type);
}

std::string DataType::getServiceDescriptionString(
	 const std::string& tab,
	 MachineType mach_type
	 ) const
{
   return getServiceInfoString(tab, getComment(), mach_type);
}

std::string DataType::getOptionalServiceNameString(
   const std::string& tab) const
{
   return getOptionalServiceInfoString(tab, getName());
}

std::string DataType::getOptionalServiceDescriptionString(
	 const std::string& tab) const
{
   return getOptionalServiceInfoString(tab, getComment());
}

std::string DataType::getServiceInfoString(
   const std::string& tab, const std::string& info,
   MachineType mach_type
   ) const
{
   // No services for pointers
   if (isPointer()) {
      return "";
   }
   std::ostringstream os;
   if (_shared) {
      os << tab << "if (" << PUBDATANAME << " == &(";
      os << "getSharedMembers().";
      os << getName();
      os << ")) {\n"
	 << tab << TAB << "return \"" << info << "\";\n"
	 << tab << "}\n";
   }
   else{
      //int status;
      //char * demangled = abi::__cxa_demangle(typeid(*this).name(),0,0,&status);
      //std::string datatype(demangled);
      //free(demangled);
      //if (datatype.find("ArrayType") != std::string::npos 
      if (this->isArray() 
	    and mach_type == MachineType::GPU)
      {
	 os << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n";
	 os << tab << "if (" << PUBDATANAME << " == &(";
	 os << REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME + _name + "[" + REF_INDEX + "]";
	 os << ")) {\n"
	    << tab << TAB << "return \"" << info << "\";\n"
	    << tab << "}\n";

	 os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n";
	 os << tab << "if (" << PUBDATANAME << " == &(";
	 os << REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME + _name + "[" + REF_CC_OBJECT +"->" + PREFIX_MEMBERNAME + _name + SUFFIX_MEMBERNAME_ARRAY + "[" + REF_INDEX + "]" + "]";
	 os << ")) {\n"
	    << tab << TAB << "return \"" << info << "\";\n"
	    << tab << "}\n";

	 os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n";
	 os << tab << "if (" << PUBDATANAME << " == &(";
	 os << REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME + _name + "[" + REF_INDEX + "*" + REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME + _name + SUFFIX_MEMBERNAME_ARRAY_MAXELEMENTS + "]";
	 os << ")) {\n"
	    << tab << TAB << "return \"" << info << "\";\n"
	    << tab << "}\n";

	 os << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n";
	 os << tab << "assert(0);\n";
	 os << "#endif\n";
      }
      else{
	 if (mach_type == MachineType::GPU)
	 {
	    os << tab << "if (" << PUBDATANAME << " == &(";
	    os << REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME + _name + "[" + REF_INDEX + "]";
	    os << ")) {\n"
	       << tab << TAB << "return \"" << info << "\";\n"
	       << tab << "}\n";
	 }
	 else{
	    os << tab << "if (" << PUBDATANAME << " == &(";
	    os << getName();
	    os << ")) {\n"
	       << tab << TAB << "return \"" << info << "\";\n"
	       << tab << "}\n";
	 }
      }

   }
   return os.str();
}

std::string DataType::getOptionalServiceInfoString(
   const std::string& tab, const std::string& info) const
{

// Always pointers don't do this   
//    // No services for pointers
//    if (isPointer()) {
//       return "";
//    }
   std::ostringstream os;
   os << tab << "if (" << PUBDATANAME << " == ";
   if (_shared) {
      os << "getSharedMembers().";
   } 

   // Do not use os << PREFIX << GETSERVICE << getName();
   // because it'll create the optional service if there isn't one
   os << getName();
   os << ") {\n"
      << tab << TAB << "return \"" << info << "\";\n"
      << tab << "}\n";
   return os.str();
}

std::string DataType::getServiceDescriptorString(const std::string& tab) const
{
   // No services for pointers
   if (isPointer()) {
      return "";
   }
   std::ostringstream os;
   os << tab << TAB << SERVICEDESCRIPTORS << ".push_back(ServiceDescriptor(\""
      << getName() << "\", \"" << getComment() << "\", \"" 
      << getDescriptor() << "\"));\n";
   return os.str();
}

std::string DataType::getOptionalServiceDescriptorString(
   const std::string& tab) const
{
// Always pointers don't do this   
//    // No services for pointers
//    if (isPointer()) {
//       return "";
//    }
   std::ostringstream os;
   os << tab << TAB << SERVICEDESCRIPTORS << ".push_back(ServiceDescriptor(\""
      << getName() << "\", \"" << getComment() << "\", \"" 
      << getDescriptor() << "\"));\n";
   return os.str();
}

DataType::~DataType() {
}

std::string DataType::getNotLegitimateDataItemString(
   const std::string& tab) const
{
   std::ostringstream os;
   os << tab << "throw SyntaxErrorException(\"" << getName() 
      << " can not be initialized with NDPairList.\\n\");\n";
   return os.str();
}

void DataType::addProxyAttribute(std::auto_ptr<Class>& instance) const
{
   std::auto_ptr<DataType> dup;
   duplicate(dup);
   dup->setPointer(false);
   instance->addDataTypeHeader(dup.get());
   instance->addDataTypeDataItemHeader(dup.get());
   std::auto_ptr<Attribute> att(new DataTypeAttribute(dup));
   att->setAccessType(AccessType::PROTECTED);
   instance->addAttribute(att);
}

void DataType::addSenderMethod(Class& instance) const
{
   std::auto_ptr<Method> sender(
      new Method(getSenderMethodName(), "void"));
   sender->setAccessType(AccessType::PROTECTED);
   MacroConditional mpiConditional(MPICONDITIONAL);
   sender->setMacroConditional(mpiConditional);
   sender->addParameter(OUTPUTSTREAM + "& stream");
   sender->setConst();
   std::string funBody;
   funBody = TAB + "stream << ";
   if (isPointer()) {
      funBody += "*";
   }
   funBody += _name + ";\n";
   sender->setFunctionBody(funBody);
   instance.addMethod(sender);
}

std::string DataType::getSenderMethodName() const
{
   return PREFIX + "send_" + _name;
}

void DataType::setArrayCharacteristics(
   unsigned blockSize, unsigned incrementSize)
{
   throw InternalException(
      "Characteristics are being set on a non-array data type.");
}
