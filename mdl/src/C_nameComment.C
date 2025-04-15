// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
