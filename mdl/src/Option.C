// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
