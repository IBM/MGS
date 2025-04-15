// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Rotation.h"

Rotation::Rotation()
{
}

Rotation::Rotation(const Rotation& r) :
	_rotation(r._rotation), _index(r._index)
{
}

bool Rotation::operator ==(const Rotation& r)
{
	return (_index==r._index);
}

bool Rotation::operator <(const Rotation& r)
{
	return (_index<r._index);
}

void Rotation::operator +=(const Rotation& r)
{
	_rotation += r._rotation;
}

Rotation::~Rotation()
{
}
