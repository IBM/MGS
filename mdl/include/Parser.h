// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef PARSER_H_
#define PARSER_H_
#include "Mdl.h"

#include "Option.h"
#include "Parameter.h"
#include <iostream>
#include <vector>
#include <algorithm>

class Parser {
	public:
		typedef std::string String;
		typedef std::vector<String> StringVector;
		typedef std::vector<Option> OptionVector;
		typedef std::vector<Parameter> ParameterVector;
		class Exception {
			private:
				String fieldMessage;
			public:
				Exception(String message) : fieldMessage(message) {}
				~Exception() {}
				String getMessage() {
					return(fieldMessage);
				}
		}; 
        private:
		OptionVector fieldOptions;
	public:
		Parser();
		virtual ~Parser();
		//
		int getOptionCount() const;
		Option getOption(int i) const;
		void addOption(Option option);
		void removeOption(Option option);
                void help();
		//
		ParameterVector parse(int argc, char *argv[]);
	private:
		OptionVector &getOptions();
		int countOptions(String arg);
		int findOption(String arg);
		int countOptions(char c);
		int findOption(char c);
		int countType(String arg, Option::Type type);
                int countArgs(StringVector &args, StringVector::size_type start);
};

inline Parser::Parser() {}
inline Parser::~Parser() {}
inline int Parser::getOptionCount() const {
	return(fieldOptions.size());
}
inline Option Parser::getOption(int i) const {
	return(fieldOptions.at(i));
}
inline void Parser::addOption(Option option) {
	// Doesn't throw an exception if same option is added twice...
	fieldOptions.push_back(option);
}
inline void Parser::removeOption(Option option) {
	// Doesn't throw an exception if non-existent option is removed...
        fieldOptions.erase(find(fieldOptions.begin(), fieldOptions.end(), option));
}

inline Parser::OptionVector &Parser::getOptions() {
	return(fieldOptions);
}

#endif /*PARSER_H_*/
