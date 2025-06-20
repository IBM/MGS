// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_edgeset_extension_H
#define C_edgeset_extension_H
#include "Copyright.h"

#include <vector>
#include <list>
#include <string>
#include "C_production.h"

class C_edge_type_set;
class C_declarator;
class C_index_set_specifier;
class GslContext;
class SyntaxError;

class C_edgeset_extension : public C_production
{
   public:
      C_edgeset_extension(const C_edgeset_extension&);
      C_edgeset_extension(C_declarator *, C_index_set_specifier *, 
			  SyntaxError *);
      C_edgeset_extension(C_declarator*, SyntaxError *);
      C_edgeset_extension(C_index_set_specifier *, SyntaxError *);
      virtual ~C_edgeset_extension();
      virtual C_edgeset_extension* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string& getEdgeType() { 
	 return _type; 
      }
      const std::vector<int>& getIndices() { 
	 return _indices; 
      }

   private:
      C_declarator* _declarator;
      C_index_set_specifier* _indexSetSpecifier;
      std::string _type;
      std::vector<int> _indices;
};
#endif
