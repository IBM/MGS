// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef Constant_H
#define Constant_H
#include "Mdl.h"

#include "InterfaceImplementorBase.h"
#include <memory>
#include <string>

class Generatable;

class Constant : public InterfaceImplementorBase {

   public:
      Constant(const std::string& fileName);
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const;
      virtual ~Constant();
      virtual std::string getType() const;
   protected:
      virtual std::string getModuleTypeName() const;

      // used by generateType, dictates which insrance type will be created.
      virtual std::string getInstanceNameForType() const {
	 return getInstanceBaseName();
      }

      virtual void internalGenerateFiles();

      virtual void addExtraInstanceBaseMethods(Class& instance) const;
//      virtual void addExtraInstanceProxyMethods(Class& instance) const;   // commented out by Jizhu Lu on 02/10/2006

      // This function is called by the getType method, it returns the
      // arguments that are necessary in creating a new instance.
      virtual std::string getInstanceNameForTypeArguments() const;

      // This function can be overridden to add an attribute to the class of generate type.
      virtual void addGenerateTypeClassAttribute(Class& c) const;

      // This method is used by generateFactory. It returns the arguments
      // that will be used  while instantiating the loaded class.
      virtual std::string getLoadedInstanceTypeArguments();
};


#endif // Constant_H
