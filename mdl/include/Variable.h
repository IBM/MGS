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

#ifndef Variable_H
#define Variable_H
#include "Mdl.h"

#include "ConnectionCCBase.h"
#include <memory>

class Generatable;

class Variable : public ConnectionCCBase {

   public:
      Variable(const std::string& fileName);
      Variable(const Variable& rv);
      Variable operator=(const Variable& rv);
      virtual void duplicate(std::auto_ptr<Generatable>& rv) const;
      virtual ~Variable();
      virtual std::string getType() const;
   protected:
      // This method is used by generateFactory. It returns the name of the
      // class that will be loaded by the factory.
      virtual std::string getLoadedInstanceTypeName();

      // This method is used by generateFactory. It returns the arguments
      // that will be used  while instantiating the loaded class.
      virtual std::string getLoadedInstanceTypeArguments();

      virtual std::string getModuleTypeName() const;
      virtual void internalGenerateFiles();

      virtual void addExtraInstanceBaseMethods(Class& instance) const;
      virtual void addExtraInstanceMethods(Class& instance) const;
      virtual void addExtraInstanceProxyMethods(Class& instance) const;
      virtual void addCompCategoryBaseConstructorMethod(Class& instance) const;
      virtual void addExtraCompCategoryBaseMethods(Class& instance) const;

   private:
      void copyContents(const Variable& rv);
      void destructContents();
};


#endif // Variable_H
