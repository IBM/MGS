// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NameComment.h"

NameComment::NameComment(const std::string& name, const std::string& comment, 
			 int blockSize, int incrementSize)
   : _name(name), _comment(comment), _blockSize(blockSize), 
     _incrementSize(incrementSize) 
{   
}

NameComment::~NameComment() 
{
}

