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
