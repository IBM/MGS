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

#ifndef C_type_specifier_H
#define C_type_specifier_H
#include "Copyright.h"

#include <string>
#include <list>
#include <vector>
#include <memory>
#include "C_production.h"

class LensContext;
class C_initializable_type_specifier;
class C_non_initializable_type_specifier;
class DataItem;
class SyntaxError;

class C_type_specifier : public C_production
{
   public:
      enum Type
      {
         _PSET,
         _REPNAME,
         _LIST,
         _MATRIX,
         _GRIDCOORD,
         _NDPAIR,
         _INT,
         _FLOAT,
         _STRING,
         _RELNODESET,
         _NODESET,
         _NODETYPE,
         _EDGETYPE,
         _FUNCTOR,
         _GRID,
         _COMPOSITE,
         _SERVICE,
         _EDGESET,
         _PORT,
         _TRIGGER,
	 _GRANULEMAPPER,
	 _UNSPECIFIED
      };

      C_type_specifier(const C_type_specifier&);
      C_type_specifier(C_initializable_type_specifier *, SyntaxError *);
      C_type_specifier(C_non_initializable_type_specifier *, SyntaxError *);
      virtual ~C_type_specifier();
      virtual C_type_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      DataItem* getValidArgument(DataItem *);
      Type getType() const;
      const char* getDataItemType();

      C_type_specifier* getNextTypeSpecifier() {
	 return _nextTypeSpec;
      }
      C_initializable_type_specifier *getInitTypeSpec() const { 
	 return _initTypeSpec; 
      }
      C_non_initializable_type_specifier *getNonInitTypeSpec() const { 
	 return _nonInitTypeSpec; 
      }

   private:
      C_initializable_type_specifier* _initTypeSpec;
      C_non_initializable_type_specifier* _nonInitTypeSpec;
      Type _type;
      C_type_specifier* _nextTypeSpec;

};
#endif
