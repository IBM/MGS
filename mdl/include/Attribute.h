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

#ifndef Attribute_H
#define Attribute_H
#include "Mdl.h"

#include "AccessType.h"
#include "DataType.h"
#include "MacroConditional.h"
#include <string>
#include <vector>

class Attribute
{
   public:
      Attribute(int accessType = AccessType::PUBLIC);
      virtual void duplicate(std::auto_ptr<Attribute>& dup) const = 0;
      virtual ~Attribute();
      virtual std::string getName() const = 0;
      virtual std::string getType() const = 0;
      virtual bool isBasic() const = 0;
      virtual bool isPointer() const = 0;
      virtual bool isOwned() const = 0;
      
      int getAccessType() const {
	 return _accessType;
      }

      void setAccessType(int acc) {
	 _accessType = acc;
      }

//       const std::string& getContainerType() const {
// 	 return _containerType;
//       }

//       void setContainerType(const std::string& containerType) {
// 	 _containerType = containerType;
//       }

      bool getStatic() const {
	 return _static;
      }

      void setStatic(bool sta = true) {
	 _static = sta;
      }

      // Returns code to embed into source (.C) file, if the attribute is
      // static, one instance is needed.
      std::string getStaticInstanceCode(const std::string& className) const;

      // Returns if the attribute is a reference.
      virtual bool isReference() const {
	 return false;
      }

      // Returns is the attribute is a C array;
      virtual bool isCArray() const {
	 return false;
      }

      // References have to be initialized at construction, therefore every
      // constructor should initialize the reference, the Class therefore 
      // needs the name of the reference when it is going to be passed in
      // as a parameter to the constructor. This method returns the parameter
      // that will be added to the constructor. Different base classes
      // might have attributes with the same name, className is used for
      // resolving this issue. The default case for  className 
      // (className == "") means, the reference is for the current class, 
      // not for a base class.
      virtual std::string getConstructorParameter(
	 const std::string& className = "") const {
	 return "";
      }

      // References have to be initialized at construction, therefore every
      // constructor should initialize the reference, the Class therefore 
      // needs the name of the reference when it is going to be passed in
      // as a parameter to the constructor. This method returns the parameters
      // name. Different base classes might have attributes with the same name,
      // className is used for resolving this issue. The default case for
      // className (className == "") means, the reference is for the
      // current class, not for a base class.
      virtual std::string getConstructorParameterName(
	 const std::string& className = "") const {
	 return "";
      }

      // Prints itself to be put in .h file if the type matches
      std::string getDefinition(int type) const;
      
      // This method fills in the initializer to be used in Constructors,
      // if the attribute is basic or it is a pointer the it initializes 
      // itself to 0.
      // Custom attribute overrides if a certain string needs to be used
      // in initializing the attribute.
      virtual void fillInitializer(std::string& init) const;

      // This method fills in the initializer to be used in copy Constructors,
      // If the attribute is not an owned pointer it is assigned to the value
      // obtained using copyFrom
      void fillCopyInitializer(std::string& init, 
			       const std::string& copyFrom) const;

      // This method returns the string that is required for copying this 
      // attribute. Parameter tab is used for indentation.
      std::string getCopyString(const std::string& tab);

      // This method returns the string that is required for deleting this 
      // attribute.
      virtual std::string getDeleteString();

      // This method returns true if this Attribute should not be copied.
      virtual bool isDontCopy() const {
	 return false;
      }

      std::string getConstructorParameterNameExtra() const {
	 return _constructorParameterNameExtra;
      }
      
      void setConstructorParameterNameExtra(const std::string& c) {
	 _constructorParameterNameExtra = c;
      }

      const MacroConditional& getMacroConditional() const {
	 return _macroConditional;
      }

      void setMacroConditional(const MacroConditional& macroConditional) {
	 _macroConditional = macroConditional;
      }      

   private:
      // Shows if the attribute is public, protected, or private
      int _accessType;

      // if != "", this attribute is in a container so it should be
      // copied specially.
//      std::string _containerType;

      // Static if true. 
      bool _static;
      
      // This is used if an attribute has to be in the default constructor's
      // parameter list even though it is not a reference, or it is a reference
      // but the user wants a special name, not the CG name.
      std::string _constructorParameterNameExtra;

      MacroConditional _macroConditional;
};

#endif
