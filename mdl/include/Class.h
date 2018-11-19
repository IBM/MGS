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

#ifndef Class_H
#define Class_H
#include "Mdl.h"

#include "MemberContainer.h"
#include "AccessType.h"
#include "FriendDeclaration.h"
#include "MacroConditional.h"
#include "TypeDefinition.h"
#include "IncludeHeader.h"
#include "IncludeClass.h"
#include "Attribute.h"
#include "Method.h"
#include "ArrayType.h"
#include <string>
#include <vector>
#include <set>
#include <map>

class DataType;
class BaseClass;

class Class
{
   public:
      enum class PrimeType{ UN_SET, Node, Variable };
      enum class SubType{ UN_SET, BaseClass, Class, BaseCompCategory, CompCategory, BaseClasFactory, BaseClassGridLayerData, BaseClassInAttrPSet, BaseClassNodeAccessor, BaseClassOutAttrPSet, BaseClassPSet, BaseClassProxy };
      void setClassInfo(std::pair<PrimeType, SubType> _pair){ _classInfo = _pair; };
      PrimeType getClassInfoPrimeType(){ return _classInfo.first; };
      SubType getClassInfoSubType(){ return _classInfo.second; };
      Class();
      Class(const std::string& name);
      Class(const Class& rv);
      void duplicate(std::auto_ptr<Class>& dup) const;
      Class& operator=(const Class& rv);
      ~Class();

      void addHeader(const std::string& header, 
		     const std::string& conditional = "") {
	 _headers.insert(IncludeHeader(header, conditional));
      }

      void addExtraSourceHeader(const std::string& header, 
		     const std::string& conditional = "") {
	 _extraSourceHeaders.insert(IncludeHeader(header, conditional));
      }

      void addDataTypeHeader(const DataType* member);
      void addDataTypeHeaders(const MemberContainer<DataType>& members);
      void addDataTypeDataItemHeader(const DataType* member);
      void addDataTypeDataItemHeaders(
	 const MemberContainer<DataType>& members);
      void addAttributes(const MemberContainer<DataType>& members
			 , AccessType accessType = AccessType::PUBLIC, bool suppressPointers=false,
			 bool add_gpu_attributes=false,
			 Class* compcat_ptr = nullptr);

      void addClass(const std::string& cl, 
		     const std::string& conditional = "") {
	 _classes.insert(IncludeClass(cl, conditional));
      }

      void addMemberClass(std::auto_ptr<Class>& cl, 
		     AccessType accessType, const std::string& conditional = "") {
	 cl->setMemberClass();
	 cl->setParentClassName(_name);
	 _memberClasses[accessType].push_back(cl.release());
      }

      void addPartnerClass(std::auto_ptr<Class>& cl, 
		     const std::string& conditional = "") {
	 _partnerClasses.push_back(cl.release());
      }

      void addBaseClass(std::auto_ptr<BaseClass>& bc) {
	 _baseClasses.push_back(bc.release());
      }

      /* to be removed when we convert from auto_ptr to unique_ptr */
      void addAttribute(std::auto_ptr<Attribute>& att) {
	 if (att->getStatic() ) _generateSourceFile=true;
	 _attributes.push_back(att.release());
      }
      void addAttribute(std::unique_ptr<Attribute>& att, MachineType mach_type=MachineType::CPU) {
	 if (att->getStatic() ) _generateSourceFile=true;
	 if (mach_type == MachineType::CPU)
	    _attributes.push_back(att.release());
	 else if (mach_type == MachineType::GPU)
	    _attributes_gpu.push_back(att.release());
	 else 
	    assert(0);
      }

      void addMethod(std::auto_ptr<Method>& mt) {
	if (!(mt->isInline())) _generateSourceFile=true;
	 _methods.push_back(mt.release());
      }
      void addMethodToExternalFile(std::string external_filename, std::auto_ptr<Method>& mt) {
	 assert(mt->isInline() == false);
	//if (!(mt->isInline())) _generateSourceFile=true;
	 _methodsInDifferentFile[external_filename].push_back(mt.release());
      }

      void addTemplateClassParameter(const std::string& str) {
	 _templateClassParameters.push_back(str);
      }

      void addTemplateClassSpecialization(const std::string& str) {
	 _templateClassSpecializations.push_back(str);
      }

      void addExtraSourceString(const std::string& str) {
	 _generateSourceFile=true;
	 _extraSourceStrings.push_back(str);
      }

      const std::string& getName() const {
	 return _name;
      }

      void setName(const std::string& name) {
	 _name = name;
      }

      void addDuplicateType(const std::string& duplicateType) {
	 _duplicateTypes.push_back(duplicateType);
      }

      bool getFileOutput() const {
	 return _fileOutput;
      }

      bool getUserCode() const {
	 return _userCode;
      }

      void setUserCode(bool userCode = true) {
	 _userCode = userCode;
      }

      const std::string& getSourceFileBeginning() const {
	 return _sourceFileBeginning;
      }

      void setSourceFileBeginning(const std::string& sourceFileBeginning) {
	 _sourceFileBeginning = sourceFileBeginning;
      }

      bool getCopyingDisabled() const {
	 return _copyingDisabled;
      }

      void setCopyingDisabled(bool copyingDisabled = true) {
	 _copyingDisabled = copyingDisabled;
      }

      bool getCopyingRemoved() const {
	 return _copyingRemoved;
      }

      void setCopyingRemoved(bool copyingRemoved = true) {
	 _copyingRemoved = copyingRemoved;
      }

      void addFriendDeclaration(const FriendDeclaration& friendDeclaration) {
	 _friendDeclarations.push_back(friendDeclaration);
      }

      const MacroConditional& getMacroConditional() const {
	 return _macroConditional;
      }

      void setMacroConditional(const MacroConditional& macroConditional) {
	 _macroConditional = macroConditional;
      }      

      void addTypeDefinition(const TypeDefinition& typeDefinition) {
	 _typeDefinitions.push_back(typeDefinition);
      }

      void getTemplateClassParametersString(std::string&);
      void getTemplateClassSpecializationsString(std::string&);

      void generate(const std::string& moduleName);
      void addStandardMethods();
      void addBasicDestructor();
      void addBasicInlineDestructor(bool isVirtual=false); 
      void setFileOutput(bool);
      void setTemplateClass(bool templateClass=true) {_templateClass=templateClass;}
      bool generateSourceFile() {return _generateSourceFile;}
      void setMemberClass(bool memberClass=true) {_memberClass=memberClass; if (memberClass == false) _nameParentClass="";}
      void setParentClassName(std::string nameParentClass) {_nameParentClass=nameParentClass;}

      bool isMemberClass() {return _memberClass;}
      void setAlternateFileName(std::string s) {_alternateFileName=s;}
      std::string getFileName();
      /*
       * arg (as called from CPU-side)= um_value.getDataRef()
       * param (for definition)= value
       * typeStr (for definition)=  'int*'
       */
      void addKernelArgs(std::string arg, std::string param, std::string typeStr){
         if (_gpuKernelArgs.empty())
         {
            _gpuKernelArgsAsCalledFromCPU = TAB + TAB + arg + "\n";
            _gpuKernelArgs = TAB + typeStr + " " + param + "\n";
         }
         else{
            _gpuKernelArgsAsCalledFromCPU += TAB + TAB + ", " + arg + "\n";
            _gpuKernelArgs += TAB + ", " + typeStr + " " + param + "\n";
         }
      }
      void addKernelArgs(DataType* dt, bool sharedData=false){
	 std::string arg = PREFIX_MEMBERNAME + dt->getName() + ".getDataRef()";
	 if (sharedData)
	       arg ="getSharedMembers()." + dt->getName();
	 std::string param = dt->getName();
	 std::string typeStr = dt->getDescriptor();
	 std::ostringstream os;
	 std::ostringstream os_gpu;
	 if (dt->isArray())
	 {
	    os << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	       << TAB << dt->getName() << ".getDataRef(),\n"
	       << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
	       << TAB << dt->getName() << ".getDataRef(),\n"
	       << TAB << dt->getName() << "_start_offset.getDataRef(),\n"
	       << TAB << dt->getName() << "_num_elements.getDataRef(),\n"
	       << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
	       << TAB << dt->getName() << ".getDataRef(),\n"
	       << TAB << dt->getName() << "_max_elements,\n"
	       << TAB << dt->getName() << "_num_elements.getDataRef(),\n"
	       << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	       << TAB << dt->getName() << ".getDataRef(),\n"
	       << TAB << "//need more info here\n"
	       << "#endif\n";

	    ArrayType* arr_dt = dynamic_cast<ArrayType*>(dt);
	    os_gpu << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	       << TAB << "ShallowArray_Flat<" << arr_dt->getType()->getTypeString() << ", Array_Flat<int>::MemLocation::UNIFIED_MEM>* " << dt->getName() << ",\n"
	       << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
	       << TAB << arr_dt->getType()->getTypeString() << "* " << dt->getName() << ",\n"
	       << TAB << "int* " << dt->getName() << "_start_offset,\n"
	       << TAB << "int* " << dt->getName() << "_num_elements,\n"
	       << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
	       << TAB << arr_dt->getType()->getTypeString() << "* " << dt->getName() << ",\n"
	       << TAB << "int " << dt->getName() << "_max_elements,\n"
	       << TAB << "int* " << dt->getName() << "_num_elements,\n"
	       << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	       << TAB << dt->getDescriptor() << "* " << dt->getName() << "\n"
	       << TAB << "//need more info here\n"
	       << "#endif\n";
	 }
	 if (_gpuKernelArgs.empty())
	 {
	    if (dt->isArray())
	    {
	       _gpuKernelArgsAsCalledFromCPU = TAB + TAB + os.str() + "\n";
	       _gpuKernelArgs = TAB + os_gpu.str() + "\n";

	    }else
	    {
	       _gpuKernelArgsAsCalledFromCPU = TAB + TAB + arg + "\n";
	       _gpuKernelArgs = TAB + typeStr + " " + param + "\n";
	    }
	 }
	 else{
	    if (dt->isArray())
	    {
	       _gpuKernelArgsAsCalledFromCPU += TAB + TAB + os.str() + "\n";
	       _gpuKernelArgs += TAB + os_gpu.str() + "\n";

	    }else
	    {
	       _gpuKernelArgsAsCalledFromCPU += TAB + TAB + ", " + arg + "\n";
	       _gpuKernelArgs += TAB + ", " + typeStr + " " + param + "\n";
	    }
	 }
      }
      void printGPUSource(std::string method, std::ostringstream& os);

      void addSharedDataToKernelArgs(const MemberContainer<DataType>& sharedMembers)
      {
	 if (sharedMembers.size() > 0) {
	    MemberContainer<DataType>::const_iterator it, end = sharedMembers.end();
	    for (it = sharedMembers.begin(); it != end; ++it) {
	       //std::string name="getSharedMembers()." + it->first;
	       //addKernelArgs(name, name, it->second->getDescriptor());
	       addKernelArgs(it->second, true);
	    }
	 }
      };
      std::string getKernelArgsAsCalledFromCPU(){ return _gpuKernelArgsAsCalledFromCPU; };

   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const Class& rv);
      void printBeginning(std::ostringstream& os);
      void printCopyright(std::ostringstream& os);
      void printHeaders(const std::set<IncludeHeader>& headers, 
			std::ostringstream& os);
      void printClasses(std::ostringstream& os);
      void printClassHeaders(std::ostringstream& os);
      void printTypeDefs(AccessType type, std::ostringstream& os);
      void printMethodDefinitions(AccessType type, std::ostringstream& os);
      void printExternCDefinitions(std::ostringstream& os);
      void printExternCPPDefinitions(std::ostringstream& os);
      void printExtraSourceStrings(std::ostringstream& os);
      void printAttributeStaticInstances(std::ostringstream& os);
      void printPartnerClasses(std::ostringstream& os);
      void printMemberClassesMethods(std::ostringstream& os);
      void printMethods(std::ostringstream& os);
      void printAttributes(AccessType type, std::ostringstream& os, MachineType mach_type=MachineType::CPU);
      void printAccess(AccessType type, const std::string& name, 
		       std::ostringstream& os);
      void printAccessMemberClasses(AccessType type, const std::string& name, 
					   std::ostringstream& os);
      bool isAccessRequired(AccessType type);
      void generateOutput(const std::string& modifier, 
			  const std::string& directory,
			  std::ostringstream& os);     
      void generateOutputCustom(const std::string& filename, 
	    const std::string& directory,
	    std::ostringstream& os);
      void generateHeader(const std::string& moduleName);     
      void generateClassDefinition(std::ostringstream& os); 
      void generateSource(const std::string& moduleName);
      void prepareBaseString(std::string& bases, 
			       const std::string& arg = "");
      void addConstructor();
      void addCopyConstructor();
      void addEqualOperator();
      void addCopyOwnedHeap();
      void addDestructOwnedHeap();
      void addDestructor();
      void addDuplicate();
      bool hasOwnedHeapData();
      std::string _name;
      std::string _nameParentClass;
      // Duplicate types that are not direct superClasses.
      std::vector<std::string> _duplicateTypes;
      std::set<IncludeHeader> _headers;
      std::set<IncludeHeader> _extraSourceHeaders;
      std::set<IncludeClass> _classes;
      std::map<AccessType, std::vector<Class*> > _memberClasses;
      std::vector<Class*> _partnerClasses;

      std::vector<std::string> _extraSourceStrings;
      std::vector<BaseClass*> _baseClasses;
      std::vector<Attribute*> _attributes;
      std::vector<Attribute*> _attributes_gpu; //copied from InterfaceImplementorBase
      std::vector<Method*> _methods;
      std::vector<std::string> _templateClassParameters;
      std::vector<std::string> _templateClassSpecializations;
      bool _fileOutput;
      bool _userCode;
      //bool _has_gpu_attributes; //turn this on if we need to create 'index', '_container'
      std::string _sourceFileBeginning;
      bool _copyingDisabled;
      bool _copyingRemoved;
      bool _templateClass;
      bool _generateSourceFile;
      bool _memberClass;
      std::string _alternateFileName;
      std::vector<FriendDeclaration> _friendDeclarations;
      MacroConditional _macroConditional;
      std::vector<TypeDefinition> _typeDefinitions;
      std::pair<PrimeType, SubType> _classInfo;

      /* add here any thing that you want to add to class definition section, 
       * e.g. a function declaration [but the body won't be here or in the source file]
       * This serve the purpose of having the funciton body in a different file, 
       * e.g. LifeNodeCompCategory.incl
       */
      //std::map<AccessType, std::vector<std::string> > _extraClassHeaderString;
      //std::map<std::string, std::vector<std::string> > _extraClassHeaderString;
      
      //map from file name, e.g. "LifeNodeCompCategory.incl" to the body of class
      std::map<std::string, std::vector<Method*>> _methodsInDifferentFile;
      std::string _gpuKernelArgsAsCalledFromCPU; //CUDA kernel argument
      std::string _gpuKernelArgs; //CUDA kernel argument
};

#endif
