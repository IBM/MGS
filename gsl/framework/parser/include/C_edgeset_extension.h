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
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
