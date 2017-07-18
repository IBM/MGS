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

#ifndef _C_PARAMETER_TYPE_PAIR_H_
#define _C_PARAMETER_TYPE_PAIR_H_
#include "Copyright.h"

#include <string>

#include "C_production.h"

class C_declarator;
class C_init_attr_type_node;
class C_init_attr_type_edge;
class LensContext;
class SyntaxError;

class C_parameter_type_pair : public C_production
{

   public:
      enum ModelType {_EDGE,_NODE};
      enum ParameterType {_INIT, _IN, _OUT, _UNINITIALIZED};
      C_parameter_type_pair(const C_parameter_type_pair&);
      C_parameter_type_pair(C_declarator *, C_init_attr_type_node *, 
			    SyntaxError * error);
      C_parameter_type_pair(C_declarator *, C_init_attr_type_edge *, 
			    SyntaxError * error);
      C_parameter_type_pair(SyntaxError * error);
      virtual ~C_parameter_type_pair();
      virtual C_parameter_type_pair* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      ModelType getModelType() {
	 return _modelType;
      }
      ParameterType getParameterType() {
	 return _parameterType;
      }
      C_init_attr_type_node *getInitAttrTypeNode() const { 
	 return _iat_node; 
      }
      C_init_attr_type_edge *getInitAttrTypeEdge() const { 
	 return _iat_edge; 
      }
      std::string const& getModelName() const;

   private:
      C_declarator *_declarator;
      C_init_attr_type_node *_iat_node;
      C_init_attr_type_edge *_iat_edge;
      ModelType _modelType;
      ParameterType _parameterType;

};
#endif
