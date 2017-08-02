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

#include "StructType.h"
#include "DataType.h"
#include "Generatable.h"
#include "Class.h"
#include "Method.h"
#include "CustomAttribute.h"
#include "MemberContainer.h"
#include "InternalException.h"
#include "Constants.h"
#include "BaseClass.h"
#include "ConstructorMethod.h"

#include <string>
#include <memory>
#include <iostream>
#include <sstream>

StructType::StructType(const std::string& fileName) 
   : DataType(), Generatable(fileName), _typeName("") 
{
}

void StructType::duplicate(std::auto_ptr<StructType>& rv) const
{
   rv.reset(new StructType(*this));
}

void StructType::duplicate(std::auto_ptr<DataType>& rv) const
{
   rv.reset(new StructType(*this));
}

void StructType::duplicate(std::auto_ptr<Generatable>& rv) const
{
   rv.reset(new StructType(*this));
}

bool StructType::isSuitableForInterface() const
{
   MemberContainer<DataType>::const_iterator it, end = _members.end();
   for (it = _members.begin(); it != end; ++it) {
      if (!it->second->isSuitableForInterface()) {
	 return false;
      }
   }
   return true;
}

bool StructType::isSuitableForFlatDemarshaller() const
{
   bool rval = true;
   MemberContainer<DataType>::const_iterator it, end = _members.end();
   for (it = _members.begin(); it != end; ++it) {
     rval &= (it->second->isBasic());
   }
   return rval;
}

void StructType::generate() const
{
   std::cout << "Struct " << _typeName << " { " << std::endl;
   MemberContainer<DataType>::const_iterator end = _members.end();
   MemberContainer<DataType>::const_iterator it;
   for (it = _members.begin(); it != end; it ++) {
      std::cout << "\t" << it->second->getString() << ";" << std::endl;
   }
   std::cout << "}\n" << std::endl;
}

std::string StructType::getInAttrPSetStr() const
{
   return getPSetStr("InAttrPSet");
}

std::string StructType::getOutAttrPSetStr() const
{
   return getPSetStr("OutAttrPSet");
}

std::string StructType::getModuleName() const
{
   return getTypeName();
}

std::string StructType::getModuleTypeName() const
{
   return "struct";
}

void StructType::internalGenerateFiles() 
{
   if (getTypeName() == "") {
      throw InternalException(
	 "getTypeName() is empty in StructType::generateFiles");
   }  
   generateInstance();
   if (isSuitableForFlatDemarshaller()) generateFlatDemarshaller();
   else generateDemarshaller();
   if (isSuitableForFlatMarshaller()) generateFlatMarshaller();
   else generateMarshaller();
   generateType();
   generateFactory();
}

const std::string& StructType::getTypeName() const
{
   return _typeName;
}

void StructType::setTypeName(const std::string& type) 
{
   _typeName = type;
}

std::string StructType::getDescriptor() const
{
   return _typeName;
}

std::string StructType::getHeaderString(
   std::vector<std::string>& arrayTypeVec) const
{
   return "\"" + _typeName + ".h\"";
}

StructType::~StructType() 
{
}

std::string StructType::getDataItemString() const
{
   return "StructDataItem";
}

std::string StructType::getDataItemFunctionString() const
{
   return "getStruct()";
}

std::string StructType::checkIfStruct(const std::string& name,
				      const std::string& tab,
				      std::string& newName) const
{
   std::ostringstream os;
   newName = name + "Struct";
   os << tab << TAB << getTypeName() << "* " << newName 
      << " = dynamic_cast<" << getTypeName() << "*>(" << name 
      << "->" << getDataItemFunctionString() << ");\n" 
      << tab << TAB << "if (" << newName << " == 0) {\n" 
      << tab << TAB << TAB << "throw SyntaxErrorException(\"Expected a " 
      << getTypeName() 
      << " for " << getName() << ".\");\n"
      << tab << TAB << "}\n";
   return os.str();
}

std::string StructType::getDataFromVariable(const std::string& name) const
{
   std::ostringstream os;
   os << "*(" << name << ")";
   return os.str();
}

void StructType::generateInstance() 
{
   std::auto_ptr<Class> instance(new Class(getTypeName()));
   
   std::auto_ptr<BaseClass> structBase(new BaseClass("Struct"));

   instance->addBaseClass(structBase);
   instance->addHeader("\"OutputStream.h\"",MPICONDITIONAL);
   instance->addHeader("\"Struct.h\"");
   instance->addHeader("\"SyntaxErrorException.h\"");
   instance->addHeader("<iostream>");
   instance->addHeader("<memory>");
   instance->addHeader("<cassert>");
   //   instance->addExtraSourceHeader("\"Marshall.h\"");
   //   instance->addExtraSourceHeader("\"CG_" + getDescriptor() + "MarshallerInstance.h\"");
   instance->addAttributes(_members);

   std::auto_ptr<Method> ostreamOpMethod(new Method("operator<<", 
						    "std::ostream&") );
   ostreamOpMethod->setExternCPP();
   ostreamOpMethod->addParameter("std::ostream& os");
   ostreamOpMethod->addParameter("const " + getDescriptor() + "& inp");
   ostreamOpMethod->setFunctionBody(
      TAB + "os << \"N/A\";\n" +
      TAB + "return os;\n");
   instance->addMethod(ostreamOpMethod);

   std::auto_ptr<Method> istreamOpMethod(new Method("operator>>", 
						    "std::istream&") );
   istreamOpMethod->setExternCPP();
   istreamOpMethod->addParameter("std::istream& is");
   istreamOpMethod->addParameter(getDescriptor() + "& inp");
   istreamOpMethod->setFunctionBody(
      TAB + FALSEASSERT +
      TAB + "return is;\n");
   instance->addMethod(istreamOpMethod);

   addDoInitializeMethods(*(instance.get()), _members);

   instance->addStandardMethods();
   _classes.push_back(instance.release());
}

void StructType::generateFlatDemarshaller() 
{
   MacroConditional mpiConditional(MPICONDITIONAL);
   std::auto_ptr<Class> demarshallerInstance(new Class("CG_" + getTypeName() + "Demarshaller")); 
   demarshallerInstance->setAlternateFileName("CG_" + getTypeName() + "Demarshaller");
   demarshallerInstance->setMacroConditional(mpiConditional);
   demarshallerInstance->addHeader("\"DemarshallerInstance.h\"");
   demarshallerInstance->addHeader("\"" + getTypeName() + ".h\"");
   std::auto_ptr<BaseClass> demarshallerBase(new BaseClass("Demarshaller"));
   demarshallerInstance->addBaseClass(demarshallerBase);

   // Add member class for flat data structure
   std::auto_ptr<Class> flatDataInstance(new Class(getTypeName()+"Data_LensReserved")); 
   flatDataInstance->addAttributes(_members);
   demarshallerInstance->addMemberClass(flatDataInstance, AccessType::PRIVATE);

   // Constructors

   std::ostringstream constructorFB;
   std::ostringstream base2InitString;

   std::auto_ptr<ConstructorMethod> baseConstructor1(new ConstructorMethod("CG_" + getTypeName() + "Demarshaller"));
   std::auto_ptr<ConstructorMethod> baseConstructor2(new ConstructorMethod("CG_" + getTypeName() + "Demarshaller"));
   baseConstructor1->setInitializationStr("_destination(0)");
   baseConstructor2->setInitializationStr("_destination(reinterpret_cast<char*>(&s->"+_members.begin()->second->getName()+"))");
   baseConstructor2->addParameter(getTypeName() + "* s");

   std::auto_ptr<Method> baseConsToIns1(baseConstructor1.release());
   std::auto_ptr<Method> baseConsToIns2(baseConstructor2.release());
   baseConsToIns1->setInline();
   baseConsToIns2->setInline();

   baseConsToIns1->setFunctionBody(constructorFB.str());
   baseConsToIns2->setFunctionBody(constructorFB.str());
   demarshallerInstance->addMethod(baseConsToIns1);
   demarshallerInstance->addMethod(baseConsToIns2); 
   
   std::auto_ptr<Method> setDestinationMethod(new Method("setDestination", "void"));
   setDestinationMethod->setInline();
   setDestinationMethod->addParameter(getTypeName()+" *s");
   std::ostringstream setDestinationMethodFB;
   setDestinationMethodFB << TAB << TAB << TAB << "_destination=reinterpret_cast<char*>(&s->"+_members.begin()->second->getName()+");\n";
   setDestinationMethodFB << TAB << TAB << TAB << "reset();\n";
   setDestinationMethod->setFunctionBody(setDestinationMethodFB.str());
   demarshallerInstance->addMethod(setDestinationMethod);

   std::auto_ptr<Method> resetMethod(new Method("reset", "void"));
   resetMethod->setInline();
   std::ostringstream resetMethodFB;
   resetMethodFB << TAB << TAB << TAB << "_offset = 0;\n";
   resetMethod->setFunctionBody(resetMethodFB.str());
   demarshallerInstance->addMethod(resetMethod);

   std::auto_ptr<Method> doneMethod(new Method("done", "bool"));
   doneMethod->setInline();
   std::ostringstream doneMethodFB;
   doneMethodFB << TAB << TAB << TAB << "return (_offset == sizeof("+getTypeName()+"Data_LensReserved));\n";
   doneMethod->setFunctionBody(doneMethodFB.str());
   demarshallerInstance->addMethod(doneMethod);

   std::auto_ptr<Method> getBlocksMethod(new Method("getBlocks", "void"));
   getBlocksMethod->setInline();
   getBlocksMethod->addParameter("std::vector<int>& blengths");
   getBlocksMethod->addParameter("std::vector<MPI_Aint>& blocs");
   std::ostringstream getBlocksMethodFB;
   getBlocksMethodFB << TAB << TAB << TAB << "blengths.push_back(sizeof(" <<getTypeName() << "Data_LensReserved));\n";
   getBlocksMethodFB << TAB << TAB << TAB << "MPI_Aint blockAddress;\n";
   getBlocksMethodFB << TAB << TAB << TAB << "MPI_Get_address(_destination, &blockAddress);\n"
		     << TAB << TAB << TAB << "blocs.push_back(blockAddress);\n";
   getBlocksMethod->setFunctionBody(getBlocksMethodFB.str());
   demarshallerInstance->addMethod(getBlocksMethod);

   std::auto_ptr<Method> demarshallMethod(new Method("demarshall", "int"));
   demarshallMethod->setInline();
   demarshallMethod->addParameter("const char * buffer");
   demarshallMethod->addParameter("int size");
   std::ostringstream demarshallMethodFB;
   demarshallMethodFB << TAB << TAB << TAB << "int retval = size;\n";
   demarshallMethodFB << TAB << TAB << TAB << "if (!done()) {\n";
   demarshallMethodFB << TAB << TAB << TAB << TAB << "int bytesRemaining = sizeof("+getTypeName()+"Data_LensReserved) - _offset;\n";
   demarshallMethodFB << TAB << TAB << TAB << TAB << "int toTransfer = (bytesRemaining<size)?bytesRemaining:size;\n";
   //demarshallMethodFB << TAB << TAB << TAB << TAB << "memcpy(_destination+_offset, buffer, toTransfer);\n";
   demarshallMethodFB << TAB << TAB << TAB << TAB << "std::copy(buffer, buffer + toTransfer,  _destination+_offset);\n";
   demarshallMethodFB << TAB << TAB << TAB << TAB << "_offset += toTransfer;\n";
   demarshallMethodFB << TAB << TAB << TAB << TAB << "retval = size - toTransfer;\n";
   demarshallMethodFB << TAB << TAB << TAB << "}\n";
   demarshallMethodFB << TAB << TAB << TAB << "return retval;\n";
   demarshallMethod->setFunctionBody(demarshallMethodFB.str());
   demarshallerInstance->addMethod(demarshallMethod);

   CustomAttribute* offsetAttr = new CustomAttribute("_offset", "int", AccessType::PRIVATE);
   CustomAttribute* destinationAttr = new CustomAttribute("_destination", "char", AccessType::PRIVATE);
   destinationAttr->setPointer();   
   std::auto_ptr<Attribute> offsetAttrAp(offsetAttr);
   std::auto_ptr<Attribute> destinationAttrAp(destinationAttr);
   demarshallerInstance->addAttribute(offsetAttrAp);
   demarshallerInstance->addAttribute(destinationAttrAp);

   demarshallerInstance->addBasicInlineDestructor();
   
   _classes.push_back(demarshallerInstance.release());  
}

void StructType::generateDemarshaller() 
{
   MacroConditional mpiConditional(MPICONDITIONAL);
   std::auto_ptr<Class> demarshallerInstance(new Class("CG_" + getTypeName() + "Demarshaller")); 
   demarshallerInstance->setAlternateFileName("CG_" + getTypeName() + "Demarshaller");
   demarshallerInstance->setMacroConditional(mpiConditional);
   demarshallerInstance->addHeader("\"DemarshallerInstance.h\"");
   demarshallerInstance->addHeader("\"StructDemarshallerBase.h\"");
   demarshallerInstance->addHeader("\"" + getTypeName() + ".h\"");
   std::auto_ptr<BaseClass> structDemarshallerBase(new BaseClass("StructDemarshallerBase"));
   demarshallerInstance->addBaseClass(structDemarshallerBase);

   // Constructors

   std::ostringstream constructorFB;
   std::ostringstream setDestinationMethodFB;
   std::ostringstream base2InitString;
   base2InitString<<"_struct(s)";
   setDestinationMethodFB << TAB << TAB << TAB << "_struct = s;\n";
   MemberContainer<DataType>::const_iterator end = _members.end();
   MemberContainer<DataType>::const_iterator it;
   for (it = _members.begin(); it != end; it ++) {
     std::string varName = it->second->getName();
     std::string varType = it->second->getDescriptor();
     constructorFB << TAB << TAB << TAB << "_demarshallers.push_back(&" << varName << "Demarshaller);\n";
     setDestinationMethodFB << TAB << TAB << TAB << varName << "Demarshaller.setDestination(&(_struct->" << varName << "));\n";
     CustomAttribute* demarshaller;
     if (it->second->isTemplateDemarshalled()) demarshaller = new CustomAttribute(varName + "Demarshaller", 
							   "DemarshallerInstance< " + varType + " >", AccessType::PRIVATE);
     else demarshaller = new CustomAttribute(varName + "Demarshaller", 
					     "CG_" + varType + "Demarshaller", AccessType::PRIVATE);
     std::auto_ptr<Attribute> demarshallerAp(demarshaller);
     demarshallerInstance->addAttribute(demarshallerAp);
     base2InitString<<",\n" << TAB << TAB << varName << "Demarshaller(&(s->" << varName <<"))";
   }
   setDestinationMethodFB << TAB << TAB << TAB << "reset();\n";

   std::auto_ptr<ConstructorMethod> baseConstructor1(new ConstructorMethod("CG_" + getTypeName() + "Demarshaller"));
   std::auto_ptr<ConstructorMethod> baseConstructor2(new ConstructorMethod("CG_" + getTypeName() + "Demarshaller"));
   baseConstructor1->setInitializationStr("_struct(0)");
   baseConstructor2->setInitializationStr(base2InitString.str());
   baseConstructor2->addParameter(getTypeName() + "* s");

   std::auto_ptr<Method> baseConsToIns1(baseConstructor1.release());
   std::auto_ptr<Method> baseConsToIns2(baseConstructor2.release());
   baseConsToIns1->setInline();
   baseConsToIns2->setInline();

   baseConsToIns1->setFunctionBody(constructorFB.str());
   baseConsToIns2->setFunctionBody(constructorFB.str());
   demarshallerInstance->addMethod(baseConsToIns1);
   demarshallerInstance->addMethod(baseConsToIns2); 
   
   std::auto_ptr<Method> setDestinationMethod(new Method("setDestination", "void"));
   setDestinationMethod->setInline();
   setDestinationMethod->addParameter(getTypeName()+" *s");
   setDestinationMethod->setFunctionBody(setDestinationMethodFB.str());
   demarshallerInstance->addMethod(setDestinationMethod);

   CustomAttribute* structDestination = new CustomAttribute("_struct", getTypeName(), AccessType::PRIVATE);
   structDestination->setPointer();   
   std::auto_ptr<Attribute> structDestinationAp(structDestination);
   demarshallerInstance->addAttribute(structDestinationAp);

   demarshallerInstance->addBasicInlineDestructor();
   
   _classes.push_back(demarshallerInstance.release());  
}

void StructType::generateFlatMarshaller() 
{
   MacroConditional mpiConditional(MPICONDITIONAL);
   std::auto_ptr<Class> marshallerInstance(new Class("CG_" + getDescriptor() + "MarshallerInstance")); 
   marshallerInstance->setAlternateFileName("CG_" + getDescriptor() + "MarshallerInstance");
   marshallerInstance->setMacroConditional(mpiConditional);
   marshallerInstance->addHeader("\"Marshall.h\"");
   marshallerInstance->addHeader("\"OutputStream.h\"");
   marshallerInstance->addHeader("\"" + getDescriptor() + ".h\"");
   marshallerInstance->addHeader("<vector>");

   if (_members.size() != 0) {
     // Add member class for flat data structure
     std::auto_ptr<Class> flatDataInstance(new Class(getTypeName()+"Data_LensReserved")); 
     flatDataInstance->addAttributes(_members);
     marshallerInstance->addMemberClass(flatDataInstance, AccessType::PRIVATE);

     std::auto_ptr<Method> marshallMethod(new Method("marshall", "void"));
     marshallMethod->setInline();
     marshallMethod->addParameter("OutputStream* stream");
     marshallMethod->addParameter(getDescriptor() + " const& data");
     std::ostringstream os;
     os << TAB << TAB << TAB << "*stream << *(reinterpret_cast<" 
	<< getTypeName() << "Data_LensReserved*>(const_cast<" 
	<< _members.begin()->second->getTypeString() << "*>(&data." 
	<< _members.begin()->second->getName() << ")));\n";
     
     marshallMethod->setFunctionBody(os.str());
     marshallerInstance->addMethod(marshallMethod);
     
     std::auto_ptr<Method> getBlocksMethod(new Method("getBlocks", "void"));
     getBlocksMethod->setInline();
     getBlocksMethod->addParameter("std::vector<int>& blengths");
     getBlocksMethod->addParameter("std::vector<MPI_Aint>& blocs");
     getBlocksMethod->addParameter(getDescriptor() + " const& data");
     os.str("");
     os.clear();
     os << TAB << TAB << TAB << "blengths.push_back(sizeof(" <<getTypeName() << "Data_LensReserved));\n";
     os << TAB << TAB << TAB << "MPI_Aint blockAddress;\n";
     os << TAB << TAB << TAB << "MPI_Get_address(const_cast<" 
	<< _members.begin()->second->getTypeString() << "*>(&data." 
	<< _members.begin()->second->getName() << "), &blockAddress);\n"
        << TAB << TAB << TAB << "blocs.push_back(blockAddress);\n";
     getBlocksMethod->setFunctionBody(os.str());
     marshallerInstance->addMethod(getBlocksMethod);
   }

   _classes.push_back(marshallerInstance.release());  
}

void StructType::generateMarshaller() 
{
   MacroConditional mpiConditional(MPICONDITIONAL);
   std::auto_ptr<Class> marshallerInstance(new Class("MarshallerInstance")); 
   marshallerInstance->setAlternateFileName("CG_" + getDescriptor() + "MarshallerInstance");
   marshallerInstance->setMacroConditional(mpiConditional);
   marshallerInstance->addHeader("\"Marshall.h\"");
   marshallerInstance->addHeader("\"OutputStream.h\"");
   marshallerInstance->addHeader("\"" + getDescriptor() + ".h\"");
   marshallerInstance->addHeader("<vector>");
   marshallerInstance->setTemplateClass();
   marshallerInstance->addTemplateClassSpecialization(getDescriptor());

   std::auto_ptr<Method> marshallMethod(new Method("marshall", "void"));
   marshallMethod->setInline();
   marshallMethod->addParameter("OutputStream* stream");
   marshallMethod->addParameter(getDescriptor() + " const& data");
   marshallMethod->setFunctionBody(getMarshallMethodFunctionBody());
   marshallerInstance->addMethod(marshallMethod);

   std::auto_ptr<Method> getBlocksMethod(new Method("getBlocks", "void"));
   getBlocksMethod->setInline();
   getBlocksMethod->addParameter("std::vector<int>& blengths");
   getBlocksMethod->addParameter("std::vector<MPI_Aint>& blocs");
   getBlocksMethod->addParameter(getDescriptor() + " const& data");
   getBlocksMethod->setFunctionBody(getGetBlocksMethodFunctionBody());
   marshallerInstance->addMethod(getBlocksMethod);

   _classes.push_back(marshallerInstance.release());  
}

std::string StructType::getMarshallMethodFunctionBody() const
{
   if (_members.size() == 0) {
      return "";
   }

   std::ostringstream os;

   MemberContainer<DataType>::const_iterator it, end = _members.end();
   std::map<std::string,int> typeMarshaller;
   std::map<std::string,int>::iterator typeMarshallerIter;
   int miSN, typeSN = 0;
   
   for (it = _members.begin(); it != end; it++) {
     std::string type=it->second->getTypeString();
     if ((typeMarshallerIter=typeMarshaller.find(type)) == typeMarshaller.end()) {
       miSN = typeSN++;
       typeMarshaller[type] = miSN;
       os << TAB << TAB << TAB << "MarshallerInstance<" << type << " > mi" << miSN << ";\n";
     } else
       miSN = (*typeMarshallerIter).second;
     os << TAB << TAB << TAB << "mi" << miSN << ".marshall(stream, data." << it->second->getName() << ");\n";
   }   
   return os.str();
}

std::string StructType::getGetBlocksMethodFunctionBody() const
{
   if (_members.size() == 0) {
      return "";
   }

   std::ostringstream os;

   MemberContainer<DataType>::const_iterator it, end = _members.end();
   std::map<std::string,int> typeMarshaller;
   std::map<std::string,int>::iterator typeMarshallerIter;
   int miSN, typeSN = 0;
   
   for (it = _members.begin(); it != end; it++) {
     std::string type=it->second->getTypeString();
     if ((typeMarshallerIter=typeMarshaller.find(type)) == typeMarshaller.end()) {
       miSN = typeSN++;
       typeMarshaller[type] = miSN;
       os << TAB << TAB << TAB << "MarshallerInstance<" << type << " > mi" << miSN << ";\n";
     } else
       miSN = (*typeMarshallerIter).second;
     os << TAB << TAB << TAB << "mi" << miSN << ".getBlocks(blengths, blocs, data." << it->second->getName() << ");\n";
   }   
   return os.str();
}

std::string StructType::getPSetStr(const std::string& which) const
{
   if (_members.size() == 0) {
      return "";
   }
   std::ostringstream os;
   os << "\n\t" << which << " {\n";
   MemberContainer<DataType>::const_iterator end = _members.end();
   MemberContainer<DataType>::const_iterator it;
   for (it = _members.begin(); it != end; it ++) {
      os << "\t\t" << it->second->getString() << ";\n";
   }
   os << "\t" << "}\n";  
   return os.str();
}

void StructType::getSubStructDescriptors(std::set<std::string>& subStructTypes) const
{
  subStructTypes.insert(getDescriptor());
  MemberContainer<DataType>::const_iterator end = _members.end();
  MemberContainer<DataType>::const_iterator it;
  for (it = _members.begin(); it != end; it ++) {
    it->second->getSubStructDescriptors(subStructTypes);
  }
}
