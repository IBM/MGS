// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      void duplicate(std::unique_ptr<BaseClass>&& dup) const;
      ~BaseClass();
      const std::string& getName() const ;
      const std::string& getConditional() const ;
      void setName(const std::string name);
      const std::vector<Attribute*>& getAttributes() const;
      void addAttribute(std::unique_ptr<Attribute>&& att);     
      std::string getInitString() const;

   private:
      void destructOwnedHeap();
      void copyOwnedHeap(const BaseClass& rv);
      std::string _name;
      std::string _conditional;
      std::vector<Attribute*> _attributes;

};


#endif // BaseClass_H
