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

#include "Utility.h"
#include <string>
#include <vector>
#include <memory>
#include <sstream>

#include "Class.h"
#include "DataType.h"
#include "Method.h"
#include "Attribute.h"
#include "CustomAttribute.h"
#include "Constants.h"


namespace mdl {

   void stripNameForCG(std::string& name)
   {
      if (name[0] == '(' && name[name.size()-1] == ')') {
	 std::string tmp = name.substr(1, name.size()-2);
	 name = tmp;
      }
      if (name[0] == '*') {
	 std::string tmp = name.substr(1, name.size()-1);
	 name = tmp;
      }
   }

   bool findInPhases(const std::string& name, 
		     const std::vector<Phase*>& vec) {
      std::vector<Phase*>::const_iterator it, end = vec.end();
      for (it = vec.begin(); it != end; ++it) {
	 if (name == (*it)->getName()) {
	    return true;
	 }      
      }
      return false;      
   };   

   bool findInTriggeredFunctions(const std::string& name, 
				 const std::vector<TriggeredFunction*>& vec) {
      std::vector<TriggeredFunction*>::const_iterator it, end = vec.end();
      for (it = vec.begin(); it != end; ++it) {
	 if (name == (*it)->getName()) {
	    return true;
	 }      
      }
      return false;      
   };   

   void tokenize(const std::string& data, std::vector<std::string>& tokens, 
		 char separator) {
      tokens.clear();
   
      int it = 0;
      int tokIndex = 0;
      while (tokIndex != -1) { 
	 tokIndex = data.find(separator, it);
	 if (tokIndex == it) {
	    ++it;
	 } else if (tokIndex == -1) {
	    tokens.push_back(data.substr(it));
	 } else {
	    tokens.push_back(data.substr(it, tokIndex - it));
	    it = tokIndex+1;
	 }      
      }    
   }

   void addOptionalServicesToClass(Class& instance, 
				   MemberContainer<DataType>& services)
   {
      MemberContainer<DataType>::iterator it, end = services.end();
      DataType* cur;
      std::auto_ptr<Attribute> attCup;
      CustomAttribute* cusAtt;
      std::auto_ptr<Method> methodCup;
      std::string methodName;
      std::string returnType;

      for(it = services.begin(); it != end; ++it) {
	 cur = it->second;
	 std::ostringstream functionBody;

	 methodName = PREFIX + GETSERVICE + cur->getName();
	 returnType = cur->getDescriptor() + "*";
	 functionBody 
	    << TAB << "if (" << cur->getName() << " == 0) {\n"
	    << TAB << TAB << cur->getName() << " = new " 
	    << cur->getDescriptor() << ";\n"
	    << TAB << "}\n"
	    << TAB << "return " << cur->getName() << ";\n";  
	 methodCup.reset(new Method(methodName, returnType, 
				    functionBody.str()));
      
	 instance.addMethod(methodCup);

	 cusAtt = new CustomAttribute(cur->getName(), cur->getDescriptor());
	 cusAtt->setPointer();
	 cusAtt->setOwned();
	 cusAtt->setBasic(cur->isBasic());
	 attCup.reset(cusAtt);

	 instance.addAttribute(attCup);
      }
   }

}

// Test for tokenize
// void printall(const std::string& str, const std::vector<std::string>& vec) {

//    std::cout << "For " << str << " --> ";
//    std::vector<std::string>::const_iterator it, end = vec.end();
//    for (it = vec.begin(); it != end; ++it) {
//       std::cout << *it << " ";
//    }
//    std::cout << "\n";
// }

// int main(int argc, char** argv) {

//    for(int i = 1; i < argc; ++i) {
//       std::string str(argv[i]);
//       std::vector<std::string> vec;
//       tokenize(str, vec, ':');
//       printall(str, vec);      
//    }
   
//    return 0;
// }
