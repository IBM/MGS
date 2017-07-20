// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Option.h"

char Option::SHORT_NAME_NONE = 0;
std::string Option::LONG_NAME_NONE = "";
Option Option::OPTION_NONE = Option(Option::SHORT_NAME_NONE, Option::LONG_NAME_NONE, Option::TYPE_NONE);

Option::Option(char shortName, std::string longName, Type type) {
	fieldShortName = shortName;
	fieldLongName = longName;
	fieldType = type;
}

Option::Option(const Option &option) {
	fieldShortName = option.getShortName();
	fieldLongName = option.getLongName();
	fieldType = option.getType();
}

Option::~Option() {
}

Option &Option::operator=(const Option &option) {
	setShortName(option.getShortName());
	setLongName(option.getLongName());
	setType(option.getType());
	return(*this);
}
