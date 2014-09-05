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

#ifndef BaseClass_H
#define BaseClass_H
#include "Mdl.h"

#include <string>
#include <vector>
#include <memory>

class Attribute;

// Base class is used by Class, some of the base classes have references
// that need to be initialized at the constructor. The derived classes 
// have to know anout these references so that they can pass the 
// input parameters to them.
class BaseClass
{
   public:
      BaseClass(const std::string& name, const std::string& conditional);
      BaseClass(const std::string& name);
      BaseClass(const BaseClass& rv);
      BaseClass& operator=(const BaseClass& rv);
      void duplicate(std::auto_ptr<BaseClass>& dup) const;
      ~BaseClass();
      const std::string& getName() const ;
      const std::string& getConditional() const ;
      void setName(const std::string name);
      const std::vector<Attribute*>& getAttributes() const;
      void addAttribute(std::auto_ptr<Attribute>& att);     
      std::string getInitString() const;

   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const BaseClass& rv);
      std::string _name;
      std::string _conditional;
      std::vector<Attribute*> _attributes;

};


#endif // BaseClass_H
