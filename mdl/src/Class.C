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

#include "Class.h"
#include "Method.h"
#include "CopyConstructorMethod.h"
#include "DefaultConstructorMethod.h"
#include "Attribute.h"
#include "CustomAttribute.h"
#include "DataTypeAttribute.h"
#include "Constants.h"
#include "AccessType.h"
#include "MemberContainer.h"
#include "DataType.h"
#include "VoidType.h"
#include "BaseClass.h"
#include "FriendDeclaration.h"

#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <time.h>
#include <cstring>

Class::Class()
   : _name(""), _fileOutput(true), _userCode(false),
     _sourceFileBeginning(""), _copyingDisabled(false), _copyingRemoved(false), _templateClass(false), _generateSourceFile(false),
     _memberClass(false), _alternateFileName("")
{
}

Class::Class(const std::string& name)
   : _name(name), _fileOutput(true), _userCode(false),
     _sourceFileBeginning(""), _copyingDisabled(false), _copyingRemoved(false), _templateClass(false), _generateSourceFile(false),
     _memberClass(false), _alternateFileName("")
{
}

Class::Class(const Class& rv)
   : _name(rv._name), _duplicateTypes(rv._duplicateTypes), 
     _headers(rv._headers),
     _extraSourceHeaders(rv._extraSourceHeaders), _classes(rv._classes),
     _extraSourceStrings(rv._extraSourceStrings), _fileOutput(rv._fileOutput), 
     _userCode(rv._userCode), _sourceFileBeginning(rv._sourceFileBeginning), 
     _copyingDisabled(rv._copyingDisabled), 
     _copyingRemoved(rv._copyingRemoved), 
     _templateClass(rv._templateClass),
     _generateSourceFile(rv._generateSourceFile),
     _memberClass(rv._memberClass),
     _alternateFileName(rv._alternateFileName),
     _friendDeclarations(rv._friendDeclarations), 
     _macroConditional(rv._macroConditional), 
     _typeDefinitions(rv._typeDefinitions)
{
   copyOwnedHeap(rv);
}

void Class::duplicate(std::auto_ptr<Class>& dup) const
{
   dup.reset(new Class(*this));
}

Class& Class::operator=(const Class& rv)
{
   if (this != &rv) {
      destructOwnedHeap();
      copyOwnedHeap(rv);
      _name = rv._name;
      _duplicateTypes = rv._duplicateTypes;
      _headers = rv._headers;
      _extraSourceHeaders = rv._extraSourceHeaders;
      _classes = rv._classes;
      _extraSourceStrings = rv._extraSourceStrings;
      _fileOutput = rv._fileOutput;
      _userCode = rv._userCode;
      _sourceFileBeginning = rv._sourceFileBeginning;
      _copyingDisabled = rv._copyingDisabled;
      _copyingRemoved = rv._copyingRemoved;
      _alternateFileName = rv._alternateFileName,
      _friendDeclarations = rv._friendDeclarations;
      _macroConditional = rv._macroConditional;
      _typeDefinitions = rv._typeDefinitions;
   }
   return *this;
}


Class::~Class()
{
   destructOwnedHeap();
}

void Class::addDataTypeHeader(const DataType* member)
{
   if (dynamic_cast<const VoidType*>(member)) { // Void Type
      return;
   }
   std::vector<std::string> arrayTypeVec;
   std::string curStr = member->getHeaderString(arrayTypeVec);
   std::vector<std::string>::iterator it, end = arrayTypeVec.end();
   for (it = arrayTypeVec.begin(); it != end; ++it) {
      _headers.insert("\"" + *it + ".h\"");
   }
   if (curStr != "") {
      _headers.insert(curStr);
   }
}

void Class::addDataTypeHeaders(const MemberContainer<DataType>& members) {
   if (members.size() > 0) {
      MemberContainer<DataType>::const_iterator end = members.end();
      MemberContainer<DataType>::const_iterator it;
      for (it = members.begin(); it != end; ++it) {
	 addDataTypeHeader(it->second);
      }
   } 
}

void Class::addDataTypeDataItemHeader(const DataType* member)
{
   if (dynamic_cast<const VoidType*>(member)) { // Void Type
      return;
   }
   std::vector<std::string> arrayTypeVec;   
   member->getHeaderString(arrayTypeVec);
   _headers.insert("\"" + member->getDataItemString() + ".h\"");
   if (arrayTypeVec.size() != 0) {
      addHeader("\"" + member->getArrayDataItemString() + ".h\"");
      addHeader("\"" + member->getHeaderDataItemString() + ".h\"");
      addHeader("\"DataItem.h\"");
   }
}

void Class::addDataTypeDataItemHeaders(
   const MemberContainer<DataType>& members) {
   if (members.size() > 0) {
      MemberContainer<DataType>::const_iterator end = members.end();
      MemberContainer<DataType>::const_iterator it;
      for (it = members.begin(); it != end; ++it) {
	 addDataTypeDataItemHeader(it->second);
      }
   } 
}

void Class::addAttributes(const MemberContainer<DataType>& members
			  , AccessType accessType, bool suppressPointers,
			  bool add_gpu_attributes)
{
   if (members.size() > 0) {
      addDataTypeHeaders(members);
      addDataTypeDataItemHeaders(members);
      MemberContainer<DataType>::const_iterator it, end = members.end();
      for (it = members.begin(); it != end; ++it) {
	 std::auto_ptr<DataType> dup;
	 it->second->duplicate(dup);
	 if (dup->isPointer() && suppressPointers) dup->setPointer(false);
 	 std::auto_ptr<Attribute> att(new DataTypeAttribute(dup));
	 att->setAccessType(accessType);
	 if (add_gpu_attributes)
	 {//make these data members 'disappear' in GPU
	   MacroConditional gpuConditional(GPUCONDITIONAL);
	   gpuConditional.setNegateCondition();
	   att->setMacroConditional(gpuConditional);
	 }
	 addAttribute(att);
      }
   }
   //extension for GPU 
   if (add_gpu_attributes)
   {
     //we need to add two data elements
     //  int index;
     //  static CG_"name"CompCategory* REF_CC_OBJECT;
     //std::unique_ptr<DataType> dup;
     //dup(new IntType());
     //std::auto_ptr<Attribute> att_index(new DataTypeAttribute(dup));
     //CustomAttribute* att_index= new CustomAttribute(REF_INDEX, "int*");
     MacroConditional gpuConditional(GPUCONDITIONAL);
     std::unique_ptr<Attribute> att_index(new CustomAttribute(REF_INDEX, "int", accessType));
     att_index->setMacroConditional(gpuConditional);
     addAttribute(att_index, MachineType::GPU);
     //_instances_GPU.addMember("index", dup);
     //_instances_GPU.addMember(att_index);
     std::unique_ptr<Attribute> att_ccAccessors(new CustomAttribute(REF_CC_OBJECT, _name + COMPCATEGORY + "*", accessType));
     att_ccAccessors->setMacroConditional(gpuConditional);
     att_ccAccessors->setStatic();
     auto ptr = dynamic_cast<CustomAttribute&>(*att_ccAccessors);
     ptr.setPointer();
     //_instances_GPU.addMember(att_ccAccessors);
     addAttribute(att_ccAccessors, MachineType::GPU);
   }
}

void Class::getTemplateClassParametersString(std::string& s) 
{
  s = "<";
  std::vector<std::string>::iterator iter = _templateClassParameters.begin();
  std::vector<std::string>::iterator end = _templateClassParameters.end();
  while (iter != end) {
    s += (*iter);
    if (++iter != end) s += ", ";
  }
  s += ">";
}

void Class::getTemplateClassSpecializationsString(std::string& s)
{
  s="";
  if (_templateClassSpecializations.size()) {
    s = "<";
    std::vector<std::string>::iterator iter = _templateClassSpecializations.begin();
    std::vector<std::string>::iterator end = _templateClassSpecializations.end();
    while (iter != end) {
      s += (*iter);
      if (++iter != end) s += ", ";
    }
    s += ">";
  }
}

void Class::generate(const std::string& moduleName)
{
   generateHeader(moduleName);
   if (_generateSourceFile) generateSource(moduleName);
}

void Class::destructOwnedHeap()
{
   for (std::vector<BaseClass*>::iterator it = _baseClasses.begin();
	it != _baseClasses.end(); ++it) {
      delete *it;
   }
   _baseClasses.clear();
   for (std::vector<Attribute*>::iterator it = _attributes.begin();
	it != _attributes.end(); ++it) {
      delete *it;
   }
   _attributes.clear();
   for (std::vector<Attribute*>::iterator it = _attributes_gpu.begin();
	it != _attributes_gpu.end(); ++it) {
      delete *it;
   }
   _attributes_gpu.clear();
   for (std::vector<Method*>::iterator it = _methods.begin();
	it != _methods.end(); ++it) {
      delete *it;
   }
   _methods.clear();
}

void Class::copyOwnedHeap(const Class& rv)
{   
   for (std::vector<BaseClass*>::const_iterator it = rv._baseClasses.begin();
	it != rv._baseClasses.end(); ++it) {
      std::auto_ptr<BaseClass> dup;
      (*it)->duplicate(dup);
      _baseClasses.push_back(dup.release());
   }
   for (std::vector<Attribute*>::const_iterator it = rv._attributes.begin();
	it != rv._attributes.end(); ++it) {
      std::auto_ptr<Attribute> dup;
      (*it)->duplicate(dup);
      _attributes.push_back(dup.release());
   }
   for (std::vector<Attribute*>::const_iterator it = rv._attributes_gpu.begin();
	it != rv._attributes_gpu.end(); ++it) {
      std::auto_ptr<Attribute> dup;
      (*it)->duplicate(dup);
      _attributes_gpu.push_back(dup.release());
   }
   for (std::vector<Method*>::const_iterator it = rv._methods.begin();
	it != rv._methods.end(); ++it) {
      std::auto_ptr<Method> dup;
      (*it)->duplicate(dup);
      _methods.push_back(dup.release());
   }
}

void Class::printBeginning(std::ostringstream& os)
{
  std::string name = _alternateFileName;
  if (name == "") name = _name;
   os << "#ifndef " << name << "_H" << "\n"
      << "#define " << name << "_H" << "\n"
      << "\n";
//      << "using namespace std;" << "\n"
//      << "\n";
}


void Class::printCopyright(std::ostringstream& os)
{
  std::string current_date; 
  std::chrono::system_clock::time_point tp = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(tp);
  std::tm* timetm = std::localtime(&time);
  char date_time_format[] = "%m-%d-%Y";
  char time_str[] = "mm-dd-yyyyaa";
  strftime(time_str, strlen(time_str), date_time_format, timetm);
  char year_format[] = "%Y";
  char year[] = "mm-dd-yyyya";
  strftime(year, strlen(year), year_format, timetm);
  os << "// =================================================================\n"
    << "// Licensed Materials - Property of IBM\n"
    << "//\n"
    << "// \"Restricted Materials of IBM\n"
    << "//\n"
    //<< "// BCM-YKT-07-18-2017\n"
    << "// BCM-YKT-"
    << time_str << "\n"
    << "//\n"
    << "//  (C) Copyright IBM Corp. 2005-"
    << year << "  All rights reserved   .\n"
    //<< "// (C) Copyright IBM Corp. 2005-2017  All rights reserved\n"
    << "// US Government Users Restricted Rights -\n"
    << "// Use, duplication or disclosure restricted by\n"
    << "// GSA ADP Schedule Contract with IBM Corp.\n"
    << "//\n"
    << "// =================================================================\n\n";
}

void Class::printHeaders(const std::set<IncludeHeader>& headers, 
			 std::ostringstream& os)
{
   for (std::set<IncludeHeader>::const_iterator it = headers.begin();
	it != headers.end(); ++it) {
      os << it->getHeaderCode();
   }  
}

void Class::printClasses(std::ostringstream& os)
{
   if (_classes.size() > 0) {
      for (std::set<IncludeClass>::const_iterator it = _classes.begin();
	   it != _classes.end(); ++it) {
	 os << it->getClassCode();
      }
      os << "\n";
   }
}

void Class::printClassHeaders(std::ostringstream& os)
{
   for (std::set<IncludeClass>::const_iterator it = _classes.begin();
	it != _classes.end(); ++it) {
      os << it->getHeaderCode();
   }
}

void Class::printTypeDefs(AccessType type, std::ostringstream& os)
{
   for (std::vector<TypeDefinition>::const_iterator it = 
	   _typeDefinitions.begin();
	it != _typeDefinitions.end(); ++it) {
     it->printTypeDef(type, os);
   }
}

void Class::printMethodDefinitions(AccessType type, std::ostringstream& os)
{
   for (std::vector<Method*>::const_iterator it = _methods.begin();
	it != _methods.end(); ++it) {
     (*it)->printDefinition(type, os);
   }
}

void Class::printExternCDefinitions(std::ostringstream& os)
{
   bool found = false;
   for (std::vector<Method*>::const_iterator it = _methods.begin();
	it != _methods.end(); ++it) {
      if ((!found) && (*it)->isExternC()) {
	 os << "extern \"C\"\n"
	    << "{\n";    
	 found = true;
      }
      (*it)->printExternCDefinition(os);
   }      
   if (found) {
      os << "}\n\n";
   }
}

void Class::printExternCPPDefinitions(std::ostringstream& os)
{
   bool found = false;
   for (std::vector<Method*>::const_iterator it = _methods.begin();
	it != _methods.end(); ++it) {
      if ((!found) && (*it)->isExternCPP() && !(*it)->isTemplate() ) {
	 os << "extern ";
	 found = true;
      }
      (*it)->printExternCPPDefinition(os);
   }      
   if (found) {
      os << "\n";
   }
}

void Class::printExtraSourceStrings(std::ostringstream& os)
{
   for (std::vector<std::string>::const_iterator 
	   it = _extraSourceStrings.begin();
	it != _extraSourceStrings.end(); ++it) {
      os << *it;
   }
}

void Class::printAttributeStaticInstances(std::ostringstream& os)
{
   for (std::vector<Attribute*>::const_iterator it = _attributes.begin();
	it != _attributes.end(); ++it) {
      os << (*it)->getStaticInstanceCode(_name);
   }
   for (std::vector<Attribute*>::const_iterator it = _attributes_gpu.begin();
        it != _attributes_gpu.end(); ++it) {
      os << (*it)->getStaticInstanceCode(_name);
   }
}


void Class::printMethods(std::ostringstream& os)
{
   for (std::vector<Method*>::const_iterator it = _methods.begin();
	it != _methods.end(); ++it) {
      (*it)->printSource(_name, os);
   }
}


void Class::printAttributes(AccessType type, std::ostringstream& os, MachineType mach_type)
{
  if (mach_type == MachineType::CPU)
  {
    for (std::vector<Attribute*>::const_iterator it = _attributes.begin();
	it != _attributes.end(); ++it) {
      os << (*it)->getDefinition(type);
    }
  }
  else if (mach_type == MachineType::GPU)
  {
    for (std::vector<Attribute*>::const_iterator it = _attributes_gpu.begin();
	it != _attributes_gpu.end(); ++it) {
      os << (*it)->getDefinition(type);
    }
  }
  else{
    assert(0);
  }
}

void Class::printPartnerClasses(std::ostringstream& os)
{
   for (std::vector<Class*>::iterator it = _partnerClasses.begin();
	it != _partnerClasses.end(); ++it) {
     os << (*it)->getMacroConditional().getBeginning();     
     (*it)->generateClassDefinition(os);
     os <<  (*it)->getMacroConditional().getEnding();     
   }
}


void Class::printAccess(AccessType type, const std::string& name, 
			std::ostringstream& os)
{
   if (isAccessRequired(type)) {
      os << TAB << name << ":\n";
      printTypeDefs(type, os);
      printMethodDefinitions(type, os);
      if (name == "protected" and _attributes_gpu.size() > 0)
      {
	os  << STR_GPU_CHECK_START;
	printAttributes(type, os, MachineType::GPU);
	os << "#else\n";
      }
      printAttributes(type, os);
      if (name == "protected")
      {
	os << "#endif\n";
      }
   }
}

void Class::printAccessMemberClasses(AccessType type, const std::string& name, 
			std::ostringstream& os)
{
  std::map<AccessType, std::vector<Class*> >::iterator classVec = _memberClasses.find(type);
  if ( classVec != _memberClasses.end() ) {
    os << TAB << name << ":\n";
    std::vector<Class*>& classes = _memberClasses[type];
    std::vector<Class*>::iterator iter = classes.begin();
    std::vector<Class*>::iterator end = classes.end();    
    for (; iter!=end; ++iter) {
      if ((*iter)->_memberClass) os << "\n";
      os << (*iter)->getMacroConditional().getBeginning();
      if ((*iter)->_memberClass) os << TAB;
      (*iter)->generateClassDefinition(os);
      os <<  (*iter)->getMacroConditional().getEnding();
      if ((*iter)->_memberClass) os << "\n";
    }   
  }
}

bool Class::isAccessRequired(AccessType type)
{
   for (std::vector<TypeDefinition>::const_iterator it = 
	   _typeDefinitions.begin();
	it != _typeDefinitions.end(); ++it) {
      if (it->getAccessType() == type) {
	 return true;
      }
   }
   for (std::vector<Attribute*>::const_iterator it = _attributes.begin();
	it != _attributes.end(); ++it) {
      if ((*it)->getAccessType() == type) {
	 return true;
      }
   }
   for (std::vector<Method*>::const_iterator it = _methods.begin();
	it != _methods.end(); ++it) {
      if ((*it)->getAccessType() == type) {
	 return true;
      }
   }
   return false;
}

void Class::generateOutput(const std::string& modifier, 
			   const std::string& directory,
			   std::ostringstream& os)
{
   if (_fileOutput) {
     std::string name = _alternateFileName;
     if (name == "") name = _name;
     /* TUAN TODO: consider using Boost.FileSystem for cross-platform */
      std::string fName = directory + "/" + name + modifier;
      if (_userCode) {
	 fName += ".gen";
      }
      //else return; // hack for MBL
      std::ofstream fs(fName.c_str());
      fs << os.str();
      fs.close();
   } 
}

void Class::generateHeader(const std::string& moduleName)
{
   std::ostringstream os;
   if (!_userCode) printCopyright(os);
   printBeginning(os); 

   os << "#include \"Lens.h\"\n";          // added by Jizhu Lu on 01/06/2006
   os << _macroConditional.getBeginning(); // #ifndef FOO_H, for example

   printHeaders(_headers, os); // #include "Foo.h", for example
   if (_headers.size() > 0) {
      os << "\n";
   }
   printClasses(os); // class Foo; for example
   
   generateClassDefinition(os); // the class!

   printPartnerClasses(os); // classes defined in same header as this one

   printExternCDefinitions(os);   // Extern C Foo(), for example
   printExternCPPDefinitions(os); 

   os << _macroConditional.getEnding(); // #endif

   os << "#endif\n";
   generateOutput(".h", moduleName + "/include", os);  // write to file
}

void Class::generateClassDefinition(std::ostringstream& os) 
{
   std::string s; 
   if (_templateClass) {
     getTemplateClassParametersString(s);
     os << "template " << s << " ";
   }
   os << "class " << _name;
   
   if (_templateClass) {
     getTemplateClassSpecializationsString(s);
     os << s;
   } 
   if (_baseClasses.size() > 0) {
      os << " : ";
      bool first = true;
      for (std::vector<BaseClass*>::const_iterator it = _baseClasses.begin();
	   it != _baseClasses.end(); ++it) {
	 std::string conditional=(*it)->getConditional();
	 if (first) {
	    first = false;
	 } else if (conditional=="") {
	    os << ", ";
	 }
	 if (conditional!="") os << "\n#ifdef "<<conditional<<"\n"<< TAB<<", ";
	 os << "public " << (*it)->getName();
	 if (conditional!="") os << "\n#endif\n";
      }     
   }
   os << "\n";
   if (_memberClass) os << TAB;
   os <<"{\n";
   if (_friendDeclarations.size() > 0) {
      std::vector<FriendDeclaration>::iterator it, 
	 end = _friendDeclarations.end();
      for (it = _friendDeclarations.begin(); it != end; ++it) {
	 if (_memberClass) os << TAB;
	 os << it->getCodeString();
      }
   }

   printAccessMemberClasses(AccessType::PUBLIC, "public", os);
   printAccessMemberClasses(AccessType::PROTECTED, "protected", os);
   printAccessMemberClasses(AccessType::PRIVATE, "private", os);

   printAccess(AccessType::PUBLIC, "public", os);
   printAccess(AccessType::PROTECTED, "protected", os);
   printAccess(AccessType::PRIVATE, "private", os);
   if (_memberClass) os << TAB;
   os << "};\n\n";
}

void Class::generateSource(const std::string& moduleName)
{
   std::ostringstream os;

   if (!_userCode) printCopyright(os);
   os << _macroConditional.getBeginning();

   os << "#include \"Lens.h\"\n";            // added by Jizhu Lu on 01/06/2006
   os << _sourceFileBeginning;
   addExtraSourceHeader("\"" + getName() + ".h\"");
   printHeaders(_extraSourceHeaders, os);
   printClassHeaders(os);
   printHeaders(_headers, os);
   os << "\n";
   printMethods(os);
   printExtraSourceStrings(os);
   printAttributeStaticInstances(os);

   os << _macroConditional.getEnding();

   generateOutput(".C", moduleName + "/src", os);
}

// This method should be called after all the attributes are added.
void Class::addStandardMethods()
{
   addConstructor();
   if (hasOwnedHeapData() && !getCopyingRemoved()){
      addCopyConstructor();
      addEqualOperator();
      addCopyOwnedHeap();
      addDestructOwnedHeap();
   }
   addDestructor();
   addDuplicate();
}

void Class::addBasicDestructor() 
{
   std::auto_ptr<Method> destructor(new Method("~" + getName()));
   std::ostringstream fb; // functionBody
   if (_attributes.size() > 0) {
      std::vector<Attribute*>::const_iterator it, end = _attributes.end();
      for (it = _attributes.begin(); it != end; ++it) {
	 fb << (*it)->getDeleteString();
      }
   }
   destructor->setFunctionBody(fb.str());  
   if (_baseClasses.size() > 0) {
      destructor->setVirtual();
   }
   addMethod(destructor);
}

void Class::addBasicInlineDestructor(bool isVirtual) 
{
   std::auto_ptr<Method> destructor(new Method("~" + getName()));
   std::ostringstream fb; // functionBody
   if (_attributes.size() > 0) {
      std::vector<Attribute*>::const_iterator it, end = _attributes.end();
      for (it = _attributes.begin(); it != end; ++it) {
	 fb << (*it)->getDeleteString();
      }
   }
   destructor->setFunctionBody(fb.str());  
   destructor->setInline();
   if (_baseClasses.size() > 0 || isVirtual) {
      destructor->setVirtual();
   }
   addMethod(destructor);
}

void Class::prepareBaseString(std::string& bases, const std::string& arg)
{
   if (_baseClasses.size() > 0) {
      bases = "";
      bool first = true;
      for (std::vector<BaseClass*>::const_iterator it = _baseClasses.begin()
	      ; it != _baseClasses.end(); ++it) {
	 if (first) {
	    first = false;
	    bases += "";
	 } else {
	    bases += ", ";
	 }
	 std::string initStr;
	 if (arg == "") { // don't override a request
	    initStr = (*it)->getInitString();
	 } else {
	    initStr = arg;
	 }
	 bases += (*it)->getName() + "(" + initStr + ")"; 
      }
   }
}

void Class::addConstructor()
{
   std::auto_ptr<DefaultConstructorMethod> constructor(
      new DefaultConstructorMethod(getName()));
   std::string bases = "";
   prepareBaseString(bases);

   std::vector<BaseClass*>::const_iterator it, end = _baseClasses.end();
   for (it = _baseClasses.begin(); it != end; ++it) {
      constructor->addDefaultConstructorParameters(
	 (*it)->getAttributes(), (*it)->getName());
   }  

   constructor->addDefaultConstructorParameters(_attributes);
   constructor->addDefaultConstructorInitializers(_attributes, bases);

   std::auto_ptr<Method> consToIns(constructor.release());
   addMethod(consToIns);
}

void Class::addCopyConstructor()
{
   std::string bases = "";
   prepareBaseString(bases, "rv");

   std::auto_ptr<CopyConstructorMethod> constructor(
      new CopyConstructorMethod(getName()));
   constructor->addCopyConstructorInitializers(_attributes, bases, "rv.");
   constructor->addParameter("const " + getName() + "& rv"); 
   if (_copyingDisabled) {
      constructor->setFunctionBody(TAB + "// Copying Disabled. \n");
      constructor->setAccessType(AccessType::PRIVATE);
   } else {   
      constructor->setFunctionBody(TAB + "copyOwnedHeap(rv);\n");
   }
   std::auto_ptr<Method> consToIns(constructor.release());
   addMethod(consToIns);
}

void Class::addEqualOperator()
{
   std::auto_ptr<Method> equalOperator(
      new Method("operator=", getName() + "&"));
   equalOperator->addParameter("const " + getName() + "& rv"); 
   std::ostringstream fb; // functionBody
   if (_copyingDisabled) {
      fb << TAB << "// Copying Disabled. \n";
      equalOperator->setAccessType(AccessType::PRIVATE);
   } else {   
      fb << TAB << "if (this != &rv) {\n";
      if (_baseClasses.size() > 0) {
	 for (std::vector<BaseClass*>::const_iterator it = _baseClasses.begin()
		 ; it != _baseClasses.end(); ++it) {
	    fb << TAB << TAB << (*it)->getName() << "::operator=(rv);\n";
	 }
      }
      fb << TAB << TAB << "destructOwnedHeap();\n"
	 << TAB << TAB << "copyOwnedHeap(rv);\n";
      if (_attributes.size() > 0) {
	 std::vector<Attribute*>::const_iterator it, end = _attributes.end();
	 for (it = _attributes.begin(); it != end; ++it) {
	    if (!((*it)->isPointer() && (*it)->isOwned())) {
	       fb << (*it)->getCopyString(TAB + TAB);
	    }
	 }
      } 
      fb << TAB << "}\n";
   }
   fb << TAB << "return *this;\n";
   equalOperator->setFunctionBody(fb.str());
   addMethod(equalOperator);

}

void Class::addCopyOwnedHeap()
{
   addHeader("<memory>");
   std::auto_ptr<Method> copyOwnedHeap(new Method("copyOwnedHeap", "void"));
   copyOwnedHeap->addParameter("const " + getName() + "& rv"); 
   std::ostringstream fb; // functionBody
   if (_copyingDisabled) {
      fb << TAB << "// Copying Disabled. \n";
   } else {
      if (_attributes.size() > 0) {
	 std::vector<Attribute*>::const_iterator it, end = _attributes.end();
	 for (it = _attributes.begin(); it != end; ++it) {
	    if ((*it)->isPointer() && (*it)->isOwned()) {
	       fb << (*it)->getCopyString(TAB);
	    }
	 }
      }
   }
   copyOwnedHeap->setFunctionBody(fb.str());
   copyOwnedHeap->setAccessType(AccessType::PRIVATE);
   addMethod(copyOwnedHeap);
}

void Class::addDestructOwnedHeap()
{
   std::auto_ptr<Method> destructOwnedHeap(
      new Method("destructOwnedHeap", "void"));
   std::ostringstream fb; // functionBody
   if (_attributes.size() > 0) {
      std::vector<Attribute*>::const_iterator it, end = _attributes.end();
      for (it = _attributes.begin(); it != end; ++it) {
	 fb << (*it)->getDeleteString();
      }
   }
   destructOwnedHeap->setFunctionBody(fb.str());  
   destructOwnedHeap->setAccessType(AccessType::PRIVATE);
   addMethod(destructOwnedHeap);
}

void Class::addDestructor()
{
   std::auto_ptr<Method> destructor(new Method("~" + getName()));
   if (hasOwnedHeapData() && !getCopyingRemoved()) {
      destructor->setFunctionBody(TAB + "destructOwnedHeap();\n");  
   }
   destructor->setVirtual();
   addMethod(destructor);
}


// If there is a pure virtual function, no duplicate will be added as a method.
void Class::addDuplicate()
{
   // Return if copying is disabled.
   if (getCopyingDisabled() || getCopyingRemoved()) {
      return;
   }
   addHeader("<memory>");
   std::string commonBody = TAB + "dup.reset(new " + getName() + "(*this));\n";

   bool pureVirtualExists = false;

   std::vector<Method*>::const_iterator it, end = _methods.end();
   for (it = _methods.begin(); it != end; ++it) {
      if ((*it)->isPureVirtual()) {
	 pureVirtualExists = true;
	 break;
      }
   }

   if (!pureVirtualExists) {
      // duplicate for self
      std::auto_ptr<Method> dupSelf(new Method("duplicate", "void"));
      dupSelf->addParameter("std::unique_ptr<" + getName() + ">& dup");
      dupSelf->setFunctionBody(commonBody);
      dupSelf->setVirtual();
      dupSelf->setConst();
      addMethod(dupSelf);

      // duplicate for the indicated
      std::vector<std::string>::iterator sit, send = _duplicateTypes.end();
      for (sit = _duplicateTypes.begin(); sit != send; ++sit) {
	 std::auto_ptr<Method> dupInd(new Method("duplicate", "void"));
	 dupInd->addParameter(
	    "std::unique_ptr<" + *sit + ">& dup");
	 dupInd->setFunctionBody(commonBody);
	 dupInd->setVirtual();
	 dupInd->setConst();
	 addMethod(dupInd);
      }

      // duplicate for the base classes
      if (_baseClasses.size() > 0) {
	 for (std::vector<BaseClass*>::const_iterator 
		 it = _baseClasses.begin(); it != _baseClasses.end(); ++it) {
	    std::auto_ptr<Method> dupBase(new Method("duplicate", "void"));
	    dupBase->addParameter("std::unique_ptr<" 
				   + (*it)->getName() + ">& dup");
	    dupBase->setFunctionBody(commonBody);
	    dupBase->setVirtual();
	    dupBase->setConst();
	    addMethod(dupBase);
	 }
      }
   }
}

bool Class::hasOwnedHeapData()
{
   bool retVal = false;
   if (_attributes.size() > 0) {
      std::vector<Attribute*>::const_iterator it, end = _attributes.end();
      for (it = _attributes.begin(); it != end; ++it) {
	 if ((*it)->isPointer() && (*it)->isOwned()) {
	    retVal = true;
	    break;
	 }
      }
   }
   return retVal;
}

void Class::setFileOutput(bool fileOutput)
{
  _fileOutput=fileOutput;
}

std::string Class::getFileName() 
{
  if (_alternateFileName == "") return _name;
  else return _alternateFileName;
}

