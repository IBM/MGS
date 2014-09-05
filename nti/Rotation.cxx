// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
