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

#ifndef C_edgeset_H
#define C_edgeset_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
class C_nodeset;
class C_edgeset_extension;
class C_edgeset_list;
class C_declarator;
class EdgeSet;
class SyntaxError;

class C_edgeset : public C_production
{
   public:
      C_edgeset(const C_edgeset&);
      C_edgeset(C_declarator*, C_edgeset_extension *, SyntaxError *);
      C_edgeset(C_nodeset* pre, C_nodeset* post, SyntaxError *);
      C_edgeset(C_declarator*, C_declarator*, SyntaxError *);
      C_edgeset(C_declarator*, C_declarator*, C_edgeset_extension*, 
		SyntaxError *);
      C_edgeset(C_nodeset* pre, C_nodeset* post, C_edgeset_extension*, 
		SyntaxError *);
      C_edgeset(C_edgeset*, C_edgeset*, SyntaxError *);
      C_edgeset(C_edgeset*, C_edgeset*, C_edgeset_extension*, SyntaxError *);
      virtual ~C_edgeset();
      virtual C_edgeset* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      EdgeSet* getEdgeSet() {
	 return _edgeset;
      }

   private:
      C_edgeset_extension* _edgesetExtension;
      C_nodeset* _pre;
      C_nodeset* _post;
      C_declarator* _edgesetDeclarator;
      EdgeSet* _edgeset;
      C_declarator* _dec1;
      C_declarator* _dec2;
      C_edgeset* _edgeset1;
      C_edgeset* _edgeset2;
};
#endif
