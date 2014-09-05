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

#ifndef Method_H
#define Method_H
#include "Mdl.h"

#include "AccessType.h"
#include "MacroConditional.h"
#include <string>
#include <vector>
#include <sstream>
#include <memory>

class DataType;
class Attribute;

class Method
{
   public:
      Method();
      Method(const std::string& name, const std::string& returnStr = ""
	     , const std::string& functionBody = "");
      virtual void duplicate(std::auto_ptr<Method>& dup) const;
      virtual ~Method();

      const std::string& getName() const {
	 return _name;
      }

      void setName(const std::string& name) {
	 _name = name;
      }

      const std::string& getReturnStr() const {
	 return _returnStr;
      }

      void setReturnStr(const std::string& returnStr) {
	 _returnStr = returnStr;
      }

      const std::string& getFunctionBody() const {
	 return _functionBody;
      }

      void setFunctionBody(const std::string& functionBody) {
	 _functionBody = functionBody;
      }

      void appendToFunctionBody(const std::string& functionBody) {
	 _functionBody += functionBody;
      }

      void addParameter(const std::string& parameter) {
	 _parameters.push_back(parameter);
      }

      const std::vector<std::string>& getParameters() const {
	 return _parameters;
      }

      int getAccessType() const {
	 return _accessType;
      }

      void setAccessType(int acc=AccessType::PUBLIC) {
	 _accessType = acc;
      }

      bool isVirtual() const {
	 return _virtual;
      }

      void setVirtual(bool vir=true) {
	 _virtual = vir;
      }

      bool isPureVirtual() const {
	 return _pureVirtual;
      }

      void setPureVirtual(bool pure=true) {
	 _pureVirtual = pure;
      }

      bool isConst() const {
	 return _const;
      }

      void setConst(bool c=true) {
	 _const = c;
      }

      bool isExternC() const {
	 return _externC;
      }

      void setExternC(bool ext=true) {
	 _externC = ext;
      }

      bool isExternCPP() const {
	 return _externCPP;
      }

      void setExternCPP(bool ext=true) {
	 _externCPP = ext;
      }

      bool isInline() const {
	 return _inline;
      }

      void setInline(bool inl=true) {
	 _inline = inl;
      }

      bool isTemplate() const {
	 return _template;
      }

      void setTemplate(bool templ=true) {
	 _template = templ;
      }

      bool isStatic() const {
	 return _static;
      }

      void setStatic(bool stat=true) {
	 _static = stat;
      }

      void addTemplateParameter(const std::string& str) {
	 _templateParameters.push_back(str);
      }

      void getTemplateParametersString(std::string&);

      const MacroConditional& getMacroConditional() const {
	 return _macroConditional;
      }

      void setMacroConditional(const MacroConditional& macroConditional) {
	 _macroConditional = macroConditional;
      }      

      void printSource(const std::string& className, std::ostringstream& os);
      void printDefinition(int type, std::ostringstream& os);
      void printExternCDefinition(std::ostringstream& os);
      void printExternCPPDefinition(std::ostringstream& os);
      void printSourceBody(std::ostringstream& os);

   protected:
      virtual std::string printConstructorExtra();
   private:
      void internalPrintDefinition(const std::string& tab, 
				   std::ostringstream& os);
      std::string _name;
      std::string _returnStr;
      std::string _functionBody;
      std::vector<std::string> _parameters;
      std::vector<std::string> _templateParameters;
      int _accessType;
      bool _virtual;
      bool _pureVirtual;
      bool _const;
      bool _externC;
      bool _externCPP;
      bool _inline;
      bool _template;
      bool _static;
      MacroConditional _macroConditional;
};

#endif
