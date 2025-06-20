// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef OPTION_H_
#define OPTION_H_
#include "Mdl.h"

#include <string>

class Option {
	public:
		static char SHORT_NAME_NONE;
		static std::string LONG_NAME_NONE;
		static Option OPTION_NONE;
		enum Type {
			TYPE_NONE, TYPE_OPTIONAL, TYPE_REQUIRED
		};
	private:
		char fieldShortName;
		std::string fieldLongName;
		Type fieldType;
	public:
		Option(char shortName, std::string longName, Type type);
		Option(const Option &option); 
                virtual ~Option();
		//
		char getShortName() const;
		void setShortName(char shortName);
		std::string getLongName() const;
		void setLongName(const std::string &longName);
		Type getType() const;
		void setType(Type type);
		//
		Option &operator=(const Option &option);
};

inline char Option::getShortName() const {
	return(fieldShortName);
}
inline void Option::setShortName(char shortName) {
	fieldShortName = shortName;
}
inline std::string Option::getLongName() const {
	return(fieldLongName);
}
inline void Option::setLongName(const std::string &longName) {
	fieldLongName = longName;
}
inline Option::Type Option::getType() const {
	return(fieldType);
}
inline void Option::setType(Type type) {
	fieldType = type;
}
inline bool operator==(const Option &optionA, const Option &optionB) {
	return(optionA.getShortName() == optionB.getShortName() &&
		optionA.getLongName() == optionB.getLongName() &&
		optionA.getType() == optionB.getType());
}

#endif /*OPTION_H_*/
