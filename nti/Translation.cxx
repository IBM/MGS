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

#include "Translation.h"
#include "string.h"

Translation::Translation()
{
}

Translation::Translation(const Translation& t) :
	_index(t._index)
{
	memcpy(_translation, t._translation, sizeof(double)*3);
}

void Translation::setTranslation(double* translation)
{
	memcpy(_translation, _translation, sizeof(double)*3);
}

bool Translation::operator ==(const Translation& t)
{
	return (_index==t._index);
}

bool Translation::operator <(const Translation& t)
{
	return (_index<t._index);
}

void Translation::operator +=(const Translation& t)
{
	_translation[0] += t._translation[0];
	_translation[1] += t._translation[1];
	_translation[2] += t._translation[2];	
}

Translation::~Translation()
{
}
