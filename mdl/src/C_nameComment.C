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

#include "C_nameComment.h"
#include "C_production.h"
#include "MdlContext.h"
#include <memory>
#include <string>

void C_nameComment::execute(MdlContext* context) 
{
}

C_nameComment::C_nameComment(const std::string& name,
			     const std::string& comment, 
			     int blockSize, int incrementSize) 
   : C_production(), _name(name), _comment(comment), _blockSize(blockSize),
     _incrementSize(incrementSize)
{
} 

C_nameComment::C_nameComment(const std::string& name,
			     int blockSize, int incrementSize) 
   : C_production(), _name(name), _comment(""), _blockSize(blockSize),
     _incrementSize(incrementSize)
{
} 

C_nameComment::C_nameComment(const C_nameComment& rv) 
   : C_production(rv), _name(rv._name), _comment(rv._comment), 
     _blockSize(rv._blockSize), _incrementSize(rv._incrementSize)
{
}

void C_nameComment::duplicate(std::unique_ptr<C_nameComment>&& rv) const
{
   rv.reset(new C_nameComment(*this));
}

void C_nameComment::duplicate(std::unique_ptr<C_production>&& rv) const
{
   rv.reset(new C_nameComment(*this));
}

C_nameComment::~C_nameComment() 
{
}
