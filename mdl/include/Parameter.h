// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PARAMETER_H_
#define PARAMETER_H_
#include "Mdl.h"

#include <string>
#include "Option.h"

class Parameter {
	private:
		Option fieldOption;
		std::string fieldValue;
	public:
		Parameter(const Option &option, const std::string &value);
		Parameter(const Parameter &parameter);
		Parameter(const Option &option);
		virtual ~Parameter();
		//
		Option getOption() const;
		void setOption(const Option &option);
		std::string getValue() const;
		void setValue(const std::string &value);
		//
		Parameter &operator=(const Parameter &parameter);
};

inline Parameter::Parameter(const Option &option, const std::string &value) :
	fieldOption(option), fieldValue(value) {
}
inline Parameter::Parameter(const Parameter &parameter) :
	fieldOption(parameter.getOption()), fieldValue(parameter.getValue()) {
}
inline Parameter::Parameter(const Option &option) :
	fieldOption(option), fieldValue("") {
}
inline Parameter::~Parameter() {}
inline Option Parameter::getOption() const {
	return(fieldOption);
}
inline void Parameter::setOption(const Option &option) {
	fieldOption = option;
}
inline std::string Parameter::getValue() const {
	return(fieldValue);
}
inline void Parameter::setValue(const std::string &value) {
	fieldValue = value;
}
inline Parameter &Parameter::operator=(const Parameter &parameter) {
	setOption(parameter.getOption());
	setValue(parameter.getValue());
	return(*this);
}

#endif /*PARAMETER_H_*/
