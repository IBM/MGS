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
#include "AccessType.h"
#include "Attribute.h"
#include "Method.h"
#include <string>
#include <vector>
#include <set>
#include <map>

class DataType;
class BaseClass;

class Class
{
   public:
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
			 , int accessType = AccessType::PUBLIC, bool suppressPointers=false);

      void addClass(const std::string& cl, 
		     const std::string& conditional = "") {
	 _classes.insert(IncludeClass(cl, conditional));
      }

      void addMemberClass(std::auto_ptr<Class>& cl, 
		     int accessType, const std::string& conditional = "") {
	 cl->setMemberClass();
	 _memberClasses[accessType].push_back(cl.release());
      }

      void addPartnerClass(std::auto_ptr<Class>& cl, 
		     const std::string& conditional = "") {
	 _partnerClasses.push_back(cl.release());
      }

      void addBaseClass(std::auto_ptr<BaseClass>& bc) {
	 _baseClasses.push_back(bc.release());
      }

      void addAttribute(std::auto_ptr<Attribute>& att) {
	 if (att->getStatic() ) _generateSourceFile=true;
	 _attributes.push_back(att.release());
      }

      void addMethod(std::auto_ptr<Method>& mt) {
	if (!(mt->isInline())) _generateSourceFile=true;
	 _methods.push_back(mt.release());
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
      void setMemberClass(bool memberClass=true) {_memberClass=memberClass;}
      bool isMemberClass() {return _memberClass;}
      void setAlternateFileName(std::string s) {_alternateFileName=s;}
      std::string getFileName();

   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const Class& rv);
      void printBeginning(std::ostringstream& os);
      void printCopyright(std::ostringstream& os);
      void printHeaders(const std::set<IncludeHeader>& headers, 
			std::ostringstream& os);
      void printClasses(std::ostringstream& os);
      void printClassHeaders(std::ostringstream& os);
      void printTypeDefs(int type, std::ostringstream& os);
      void printMethodDefinitions(int type, std::ostringstream& os);
      void printExternCDefinitions(std::ostringstream& os);
      void printExternCPPDefinitions(std::ostringstream& os);
      void printExtraSourceStrings(std::ostringstream& os);
      void printAttributeStaticInstances(std::ostringstream& os);
      void printPartnerClasses(std::ostringstream& os);
      void printMethods(std::ostringstream& os);
      void printAttributes(int type, std::ostringstream& os);
      void printAccess(int type, const std::string& name, 
		       std::ostringstream& os);
      void printAccessMemberClasses(int type, const std::string& name, 
					   std::ostringstream& os);
      bool isAccessRequired(int type);
      void generateOutput(const std::string& modifier, 
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
      // Duplicate types that are not direct superClasses.
      std::vector<std::string> _duplicateTypes;
      std::set<IncludeHeader> _headers;
      std::set<IncludeHeader> _extraSourceHeaders;
      std::set<IncludeClass> _classes;
      std::map<int, std::vector<Class*> > _memberClasses;
      std::vector<Class*> _partnerClasses;

      std::vector<std::string> _extraSourceStrings;
      std::vector<BaseClass*> _baseClasses;
      std::vector<Attribute*> _attributes;
      std::vector<Method*> _methods;
      std::vector<std::string> _templateClassParameters;
      std::vector<std::string> _templateClassSpecializations;
      bool _fileOutput;
      bool _userCode;
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
};

#endif
