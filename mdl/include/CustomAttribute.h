#include <memory>
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

#ifndef CustomAttribute_H
#define CustomAttribute_H
#include "Mdl.h"

#include "AccessType.h"
#include "DataType.h"
#include "Attribute.h"
#include <string>
#include <vector>

class CustomAttribute : public Attribute
{
   public:
      CustomAttribute();
      CustomAttribute(const std::string& name, const std::string& type,
		      AccessType accessType = AccessType::PUBLIC);
      void duplicate(std::unique_ptr<Attribute>&& dup) const;
      virtual ~CustomAttribute();

      virtual std::string getName() const;
      void setName(const std::string& name);
      virtual std::string getType() const;
      void setType(const std::string& type);
      virtual bool isBasic() const;
      void setBasic(bool basic=true);
      virtual bool isPointer() const;
      void setPointer(bool pointer=true);
      virtual bool isOwned() const;
      void setOwned(bool owned=true);
      virtual bool isCArray() const;
      void setCArray(bool cArray=true);
      std::string getCArraySize() const;
      void setCArraySize(const std::string& cArraySize);
      virtual bool isReference() const;
      void setReference(bool reference = true);
      const std::string& getInitializeString() const;
      void setInitializeString(const std::string& init);
      virtual std::string getDeleteString();
      void setCustomDeleteString(std::string);
      void setCompleteCustomDeleteString(std::string);
      std::string getCustomDeleteString();

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
	 const std::string& className = "") const;

      // References have to be initialized at construction, therefore every
      // constructor should initialize the reference, the Class therefore 
      // needs the name of the reference when it is going to be passed in
      // as a parameter to the constructor. This method returns the parameters
      // name. Different base classes might have attributes with the same name,
      // className is used for resolving this issue. The default case for
      // className (className == "") means, the reference is for the
      // current class, not for a base class.
      virtual std::string getConstructorParameterName(
	 const std::string& className = "") const;

      // This method returns true if this Attribute should not be copied.
      virtual bool isDontCopy() const;

      // This method fills in the initializer to be used in Constructors,
      // if the attribute is basic or it is a pointer the it initializes 
      // itself to 0.
      // Custom attribute overrides if a certain string needs to be used
      // in initializing the attribute.
      virtual void fillInitializer(std::string& init, const Class* classObj=0) const;


   private:
      std::string _name;
      std::string _type;
      bool _basic;
      bool _pointer;
      bool _owned;
      bool _cArray;
      bool _customDelete;
      /* this flag is added to provide a way to customize the deletion
       * of _nodeInstanceAccessors */
      bool _completeCustomeDelete=false; 
      std::string _cArraySize;
      bool _reference;
      std::string _parameterName;
      std::string _initializeString;
      std::string _customDeleteString;
};

#endif
