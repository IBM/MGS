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

#ifndef SBROWSER_H
#define SBROWSER_H
#include "Copyright.h"

#include "PauseActionable.h"
#include "QueryResult.h"

#include "Simulation.h"
#include "SimulationQueriable.h"
#include "Publisher.h"
#include "EnumEntry.h"
#include "QueryField.h"
#include "TriggerDataItem.h"
#include "Service.h"
#include "ServiceDescriptor.h"
#include "ServiceDataItem.h"
#include "GraphicalUserInterface.h"

#include <string>
#include <vector>
#include <list>

class Repertoire;
class GraphicalUserInterface;
class Publisher;
class Queriable;
class QueriableDescriptor;
class QueryDescriptor;
class QueryField;
class TriggerType;

class SBrowser : public PauseActionable {
  public:
  SBrowser(Simulation& sim, GraphicalUserInterface* UI);
  void browse();
  void action();
  ~SBrowser();

  private:
  Simulation& _sim;
  std::string addColon(std::string);
  std::string bracket(std::string, int);
  std::string bracket(int, int);
  std::string getCommand();
  int getSelection();
  int getSelection(const std::string&);
  bool isMatch(std::string input, std::string comp);
  Repertoire* _root;
  Queriable* _currentQ;
  SimulationQueriable* _simQ;

  // history of queriables visited & locations
  std::vector<Queriable*> _historyQ;
  std::vector<const char*> _location;

  bool _fieldSet;
  QueryResult* _result;
  GraphicalUserInterface* _UI;

  int _i;

  void cmdHandler(const char* cmd);
  int parseQuery(const char* q);

  // sends the name and location of the current queriable
  void sendNameLocation();

  // sends all the subscriptions of the given service
  void sendService(Service* sv);

  // sends the query fields of the current queriable
  void sendDescriptor(QueryDescriptor& qd);

  // sends the results of the query
  void sendResults(QueryResult* r);

  // used to send the list of sub-queriables
  void sendQbls(std::list<Queriable*> const& v);

  // sends all the triggers attached to a publisher
  void sendTriggers(Publisher* p);

  // sends all the services attached to a publisher
  void sendServices(Publisher* p);

  // sends the bookmarks
  void sendBookmarks();
  void sendQbookmarks();  // auxillary to sendBookmarks()
  void sendTbookmarks();  // auxillary to sendBookmarks()
  void sendSbookmarks();  // auxillary to sendBookmarks()

  // refresh everything related to the current queriable
  // (sub queriables, services/triggers, etc)
  // bookmarks are also sent
  void refresh();

  // bookmarks: q = queriable, s = service, t = trigger
  // l = location. for example, _qlbookmarks stores the
  // locations (psuedo url thats displayed in browser)
  // associated with the elements in _qbookmarks
  std::vector<std::vector<const char*> > _qlbookmarks;
  std::vector<Queriable*> _qbookmarks;
  std::vector<std::vector<const char*> > _slbookmarks;
  std::vector<std::vector<const char*> > _tlbookmarks;
  std::vector<TriggerType*> _tbookmarks;
  std::vector<Service*> _sbookmarks;
};

inline SBrowser::SBrowser(Simulation& sim, GraphicalUserInterface* UI)
    : _sim(sim) {
  _currentQ = _simQ = new SimulationQueriable(&sim);
  // only this queriable need be deleted in destructor. JK.
  _historyQ.push_back(_currentQ);
  _UI = UI;
  _result = NULL;
  _location.push_back(_currentQ->getQueriableDescriptor().getName().c_str());
}

inline SBrowser::~SBrowser() { delete _historyQ[0]; }

inline void SBrowser::sendQbls(std::list<Queriable*> const& qbls) {
  std::list<Queriable*>::const_iterator itr = qbls.begin();
  std::list<Queriable*>::const_iterator end = qbls.end();

  _UI->getServer()->sendToClient("display browser clear queriables");

  while (itr != end) {
    // TUAN: temporary disable the code
    //     until C++ version is replaced
    /*
     char buff2[MAXBUFFSIZE];
     strcpy(buff2, "display browser queriable <>");

     if ((*itr)->isPublisherQueriable()) {
       strcat(buff2, "is publisher<>");
     } else {
       strcat(buff2, "not publisher<>");
     }

     std::string name = (*itr)->getQueriableDescriptor().getName();
     std::string desc = (*itr)->getQueriableDescriptor().getDescription();
     char qname[MAXBUFFSIZE];
     char qdesc[MAXBUFFSIZE];
     strcpy(qname, name.c_str());
     strcpy(qdesc, desc.c_str());

     _UI->getServer()->muter(qname);
     _UI->getServer()->muter(qdesc);

     strcat(buff2, qname);
     strcat(buff2, "<>");
     strcat(buff2, qdesc);
     strcat(buff2, "<>");
     _UI->getServer()->sendToClient(buff2);
         */
    itr++;
  }
}

inline void SBrowser::sendService(Service* sv) {

  std::string dataval = sv->getStringValue();
  // TUAN: temporary disable the code
  /*
  char subbuff[MAXBUFFSIZE];
  strcpy(subbuff, "display browser subscription <>");

  if ((dataval == "") || (dataval == " ")) {
    dataval = "No Value";
  }
  strcat(subbuff, dataval.c_str());
  strcat(subbuff, "<>");
  _UI->getServer()->sendToClient(subbuff);
  */
}

inline void SBrowser::sendResults(QueryResult* r) {
  // : temporary disable the code
  /*
_UI->getServer()->sendToClient("display browser clear results");
for (unsigned i = 0; i < r->size(); i++) {
char buff[MAXBUFFSIZE];
strcpy(buff, "display browser result <>");
Queriable* q = (*r)[i];

if (q->isPublisherQueriable()) {
strcat(buff, "is publisher<>");
} else {
strcat(buff, "not publisher<>");
}

char qname[MAXBUFFSIZE];
strcpy(qname, q->getQueriableDescriptor().getName().c_str());
_UI->getServer()->muter(qname);
strcat(buff, qname);
strcat(buff, "<>");

char qdesc[MAXBUFFSIZE];
strcpy(qdesc, q->getQueriableDescriptor().getDescription().c_str());
_UI->getServer()->muter(qdesc);
strcat(buff, qdesc);
strcat(buff, "<>");

_UI->getServer()->sendToClient(buff);
}
*/
}

inline void SBrowser::sendDescriptor(QueryDescriptor& qd) {
  // TUAN: temporary disable the code
  //       until C++ version replace
  /*
std::vector<QueryField*> v = qd.getQueryFields();

_UI->getServer()->sendToClient("display browser queryfields begin");

for (unsigned i = 0; i < v.size(); i++) {
char buff[MAXBUFFSIZE];
strcpy(buff, "display browser queryfield name <>");
strcat(buff, v[i]->getName().c_str());
strcat(buff, "<>");
_UI->getServer()->sendToClient(buff);

strcpy(buff, "display browser queryfield desc <>");
strcat(buff, v[i]->getDescription().c_str());
strcat(buff, "<>");
_UI->getServer()->sendToClient(buff);

if (v[i]->getType() == QueryField::ENUM) {
std::vector<EnumEntry*> enums = v[i]->getEnumEntries();
_UI->getServer()->sendToClient("display browser queryfield enums begin");
for (unsigned j = 0; j < enums.size(); j++) {
  char buff2[MAXBUFFSIZE];
  strcpy(buff2, "display browser queryfield enum <>");
  strcat(buff2, enums[j]->getValue().c_str());
  strcat(buff2, "<>");
  strcat(buff2, enums[j]->getDescription().c_str());
  strcat(buff2, "<>");
  _UI->getServer()->sendToClient(buff2);
}
_UI->getServer()->sendToClient("display browser queryfield enums end");
} else {
char buff2[MAXBUFFSIZE];
strcpy(buff2, "display browser queryfield format <>");
strcat(buff2, v[i]->getFormat().c_str());
_UI->getServer()->sendToClient(buff2);
}
}

_UI->getServer()->sendToClient("display browser queryfields end");
*/
}

inline void SBrowser::sendNameLocation() {
  // TUAN: temporary disable the code
  /*
char outcmd[MAXBUFFSIZE];

strcpy(outcmd, "display browser setname <>");

char description[MAXBUFFSIZE];
strcpy(description, _currentQ->getQueriableDescriptor().getName().c_str());
_UI->getServer()->muter(description);
strcat(outcmd, description);
strcat(outcmd, "<>");
_UI->getServer()->sendToClient(outcmd);

char msg[MAXBUFFSIZE];
strcpy(msg, "display browser location ");

unsigned length = MAXBUFFSIZE - (strlen(msg) + 1);
char* location = new char[length];

strcpy(location, "/");

for (unsigned i = 0; (i < _location.size()) && (i < length); i++) {
strcat(location, _location[i]);
strcat(location, "/");
}
strcat(msg, location);
_UI->getServer()->sendToClient(msg);

strcpy(msg, "display browser description <>");
strcat(msg, _currentQ->getQueriableDescriptor().getDescription().c_str());
strcat(msg, "<>");
_UI->getServer()->sendToClient(msg);
*/
}

inline void SBrowser::sendTriggers(Publisher* p) {
  // TUAN: temporary disable the code
  /*
const std::vector<TriggerType*>& ttypes = p->getTriggerDescriptors();

int nbrTypes = ttypes.size();
for (int i = 0; i < nbrTypes; ++i) {
char buff[MAXBUFFSIZE];
strcpy(buff, "display browser trigger <>");
strcat(buff, ttypes[i]->getName().c_str());
strcat(buff, "<>");
strcat(buff, ttypes[i]->getDescription().c_str());
strcat(buff, "<>");

_UI->getServer()->sendToClient(buff);
}*/
}

inline void SBrowser::sendServices(Publisher* p) {
  // TUAN: temporary disable the code
  /*
const std::vector<ServiceDescriptor>& serviceDescriptors =
p->getServiceDescriptors();

std::vector<ServiceDescriptor>::const_iterator it,
end = serviceDescriptors.end();
for (it = serviceDescriptors.begin(); it != end; ++it) {
char buff[MAXBUFFSIZE];
strcpy(buff, "display browser service <>");
strcat(buff, it->getName());
strcat(buff, "<>");
strcat(buff, it->getDescription());
strcat(buff, "<>");
strcat(buff, it->getDataItemDescription());
strcat(buff, "<>");
_UI->getServer()->sendToClient(buff);
} */
}

inline void SBrowser::browse() {
  std::cout << "SBrowser: browsing, now sending wakeup!" << std::endl;
  _UI->getServer()->sendToClient("display browser wakeup");

  std::list<Queriable*> const& qbls = _currentQ->getQueriableList();

  std::string cmd;

  int go = 1;

  _i = 1;
  sendQbls(qbls);
  sendDescriptor(_currentQ->getQueryDescriptor());
  sendBookmarks();
  if (_currentQ->isPublisherQueriable()) {
    Publisher* p = _currentQ->getQPublisher();
    _UI->getServer()->sendToClient("display browser clear publisher");
    sendServices(p);
    sendTriggers(p);
  } else {
    _UI->getServer()->sendToClient("display browser clear publisher");
  }
  while (_i) {
    _simQ->refresh();
    sendNameLocation();
    go = 0;
    cmd = getCommand();
    cmdHandler(cmd.c_str());
    if (cmd == "control stop") {
      break;
    }
    if (_currentQ->isPublisherQueriable()) {
      Publisher* p = _currentQ->getQPublisher();
      _UI->getServer()->sendToClient("display browser clear publisher");
      sendServices(p);
      sendTriggers(p);
    } else {
      _UI->getServer()->sendToClient("display browser clear publisher");
    }
  }
}

inline void SBrowser::refresh() {
  std::cout << "SBrowser: Refreshing.." << std::endl;
  _simQ->refresh();

  sendNameLocation();
  sendQbls(_currentQ->getQueriableList());
  if (_currentQ->isPublisherQueriable()) {
    Publisher* p = _currentQ->getQPublisher();
    _UI->getServer()->sendToClient("display browser clear publisher");
    sendServices(p);
    sendTriggers(p);
  } else {
    _UI->getServer()->sendToClient("display browser clear publisher");
  }

  sendDescriptor(_currentQ->getQueryDescriptor());
  sendBookmarks();
}

inline void SBrowser::sendBookmarks() {
  _UI->getServer()->sendToClient("display browser clear bookmarks");
  sendQbookmarks();
  sendTbookmarks();
  sendSbookmarks();
}

inline void SBrowser::sendQbookmarks() {
  // TUAN: temporary disable the code
  /*
for (unsigned int u = 0; u < _qbookmarks.size(); u++) {
char buff[MAXBUFFSIZE];

strcpy(buff, "display browser bookmark <>queriable<>");

if (_qbookmarks[u]->isPublisherQueriable()) {
strcat(buff, "is publisher<>");
} else {
strcat(buff, "not publisher<>");
}

std::string name = _qbookmarks[u]->getQueriableDescriptor().getName();
std::string desc =
  _qbookmarks[u]->getQueriableDescriptor().getDescription();
char qname[MAXBUFFSIZE];
char qdesc[MAXBUFFSIZE];
strcpy(qname, name.c_str());
strcpy(qdesc, desc.c_str());

_UI->getServer()->muter(qname);
_UI->getServer()->muter(qdesc);

strcat(buff, qname);
strcat(buff, "<>");
strcat(buff, qdesc);
strcat(buff, "<>");
char location[MAXBUFFSIZE];
strcpy(location, "/");

for (unsigned int i = 0; i < _qlbookmarks[u].size(); i++) {
strcat(location, _qlbookmarks[u][i]);
strcat(location, "/");
}

_UI->getServer()->muter(location);
strcat(buff, location);
strcat(buff, "<>");
_UI->getServer()->sendToClient(buff);
}
*/
}

inline void SBrowser::sendTbookmarks() {
  // TUAN: temporary disable the code
  /*
unsigned int nbrTypes = _tbookmarks.size();

for (unsigned int i = 0; i < nbrTypes; ++i) {
char buff[MAXBUFFSIZE];
strcpy(buff, "display browser bookmark <>trigger<>");
strcat(buff, _tbookmarks[i]->getName().c_str());
strcat(buff, "<>");
strcat(buff, _tbookmarks[i]->getDescription().c_str());
strcat(buff, "<>");

char location[MAXBUFFSIZE];
strcpy(location, "/");

for (unsigned int j = 0; j < _tlbookmarks[i].size(); j++) {
strcat(location, _tlbookmarks[i][j]);
strcat(location, "/");
}

_UI->getServer()->muter(location);
strcat(buff, location);
strcat(buff, "<>");
_UI->getServer()->sendToClient(buff);
}
*/
}

inline void SBrowser::sendSbookmarks() {
  // TUAN: temporary disable the code
  /*
for (unsigned i = 0; i < _sbookmarks.size(); i++) {
char buff[MAXBUFFSIZE];
strcpy(buff, "display browser bookmark <>service<>");
strcat(buff, _sbookmarks[i]->getName().c_str());
strcat(buff, "<>");
strcat(buff, _sbookmarks[i]->getDescription().c_str());
strcat(buff, "<>");
strcat(buff, _sbookmarks[i]->getDataItemDescription().c_str());
strcat(buff, "<>");

char location[MAXBUFFSIZE];
strcpy(location, "/");

for (unsigned int j = 0; j < _slbookmarks[i].size(); j++) {
strcat(location, _slbookmarks[i][j]);
strcat(location, "/");
}

_UI->getServer()->muter(location);
strcat(buff, location);
strcat(buff, "<>");
_UI->getServer()->sendToClient(buff);
}
*/
}

inline void SBrowser::cmdHandler(const char* cmd) {

  std::string cmdLine(cmd);

  std::string cmd_browse_query_queryfields("browse query queryfields");
  std::string cmd_browse_refresh("browse refresh");
  std::string cmd_browse_bookmark_queriable("browse bookmark queriable");
  std::string cmd_browse_bookmark_trigger("browse bookmark trigger");
  std::string cmd_browse_bookmark_service("browse bookmark service");
  std::string cmd_browse_get_service_dataitem("browse get service dataitem");
  std::string cmd_view_service("browse view service");
  std::string cmd_browse_queriable_bookmark_select(
      "browse Queriables bookmark select");
  std::string cmd_browse_service_bookmark_select(
      "browse Services bookmark select");
  std::string cmd_browse_back("browse back");

  std::string cmd_browse_select_queriable("browse select queriable");
  std::string cmd_browse_select_publisher("browse select publisher");
  std::string cmd_browse_select_queriable_publisher(
      "browse select queriable publisher");
  std::string cmd_browse_select_result("browse select result");
  std::string cmd_browse_get_queriable_dataitem(
      "browse get queriable dataitem");
  std::string cmd_browse_get_result_dataitem("browse get result dataitem");

  if (cmdLine == cmd_browse_query_queryfields) {

    // TUAN: temporary disable the code until we convert all strcpy... to C++
    /*    char* buff = new char[strlen(cmd) + 1];
        strcpy(buff, cmd);
        strtok(buff, "<>");  // get rid of header;
                             // 100
        int max = atoi(strtok(NULL, "<>"));
        // 0
        int min = atoi(strtok(NULL, "<>"));
        // 100
        int size = atoi(strtok(NULL, "<>"));
    if (parseQuery(cmd)) {
      _result = (_currentQ->query(max, min, size)).release();
      sendResults(_result);
    }
    delete[] buff;
    */
  } else if (cmdLine == cmd_browse_refresh) {
    //  if (strcmp(cmd, "browse refresh") == 0) {
    refresh();
    return;
  } else if (cmdLine == cmd_browse_query_queryfields) {
    //  if (strcmp(cmd, "browse bookmark queriable") == 0) {
    _qbookmarks.push_back(_currentQ);
    _qlbookmarks.push_back(_location);
  } else if (cmdLine == cmd_browse_bookmark_trigger) {
    //  if (strncmp(cmd, "browse bookmark trigger", 23) == 0) {

    if (_currentQ->isPublisherQueriable()) {
      Publisher* p = _currentQ->getQPublisher();
      const std::vector<TriggerType*>& v = p->getTriggerDescriptors();
      int selection = atoi(cmd + 23);
      _tbookmarks.push_back(v[selection]);
      _tlbookmarks.push_back(_location);
    }
  } else if (cmdLine == cmd_browse_bookmark_service) {
    //  if (strncmp(cmd, "browse bookmark service", 23) == 0) {
    if (_currentQ->isPublisherQueriable()) {
      Publisher* p = _currentQ->getQPublisher();
      const std::vector<ServiceDescriptor>& v = p->getServiceDescriptors();
      int selection = atoi(cmd + 23);
      std::string serviceName(v[selection].getName());
      Service* service = p->getService(serviceName);
      _sbookmarks.push_back(service);
      _slbookmarks.push_back(_location);
    }
  } else if (cmdLine == cmd_browse_get_service_dataitem) {
    //  if (strncmp(cmd, "browse get service dataitem", 27) == 0) {
    if (_currentQ->isPublisherQueriable()) {
      // TUAN: temporary disable the code
      /*
Publisher* p = _currentQ->getQPublisher();
const std::vector<ServiceDescriptor>& v = p->getServiceDescriptors();
int selection = atoi(cmd + 27);
std::string serviceName(v[selection].getName());
Service* s = p->getService(serviceName);
ServiceDataItem* sd = new ServiceDataItem();
sd->setService(s);
char dataitembuff[MAXBUFFSIZE];
strcpy(dataitembuff, "display dataitem ");
strcat(dataitembuff, sd->getType());

_UI->getServer()->sendToClient(dataitembuff);
std::unique_ptr<DataItem> apdi(sd);
_UI->getServer()->keepDataItem(apdi);
*/
    } else {
      std::ostringstream os;
      os << "display extended_error <>The server returned the following: <> "
            "Could"
         << " not complete requested action. The item you have selected does "
            "not "
         << "have a Publisher, so no service can be retrieved from it. There "
            "may "
         << "be a mismatch in what you have selected, and what the server "
            "thinks "
         << "you have selected.\n Item selected: \""
         << _currentQ->getQueriableDescriptor().getName()
         << "\".<>";
      // TUAN: temporary disable the code
      /*
  char errBuff[MAXBUFFSIZE];
  strcpy(errBuff, os.str().c_str());
  _UI->getServer()->sendToClient(errBuff);
      */
    }
  } else if (cmdLine == cmd_view_service) {
    // if (strncmp(cmd, "browse view service", 19) == 0) {
    if (_currentQ->isPublisherQueriable()) {
      Publisher* p = _currentQ->getQPublisher();
      const std::vector<ServiceDescriptor>& v = p->getServiceDescriptors();
      unsigned selection = atoi(cmd + 19);

      if ((selection >= 0) && (selection < v.size())) {
        std::string serviceName(v[selection].getName());
        Service* sv = p->getService(serviceName);
        sendService(sv);
      }
    } else {
      std::cout
          << "SBrowser: Selected Queriable is not publisher queriable..oops"
          << std::endl << "\t"
          << _currentQ->getQueriableDescriptor().getName().c_str() << std::endl
          << "\t"
          << _currentQ->getQueriableDescriptor().getDescription().c_str()
          << std::endl;
    }
  } else if (cmdLine == cmd_browse_queriable_bookmark_select) {
    //  if (strncmp(cmd, "browse Queriables bookmark select", 33) == 0) {
    unsigned selection = atoi(cmd + 33);

    if ((selection >= 0) && (selection < _qbookmarks.size())) {
      _currentQ = _qbookmarks[selection];
      _location = _qlbookmarks[selection];
      _historyQ.push_back(_currentQ);
      std::list<Queriable*> const& qbls = _currentQ->getQueriableList();
      sendQbls(qbls);
      sendDescriptor(_currentQ->getQueryDescriptor());
    }
  } else if (cmdLine == cmd_browse_service_bookmark_select) {
    //  if (strncmp(cmd, "browse Services bookmark select", 31) == 0) {
    int selection = atoi(cmd + 31);
    sendService(_sbookmarks[selection]);
  } else if (cmdLine == cmd_browse_back) {

    // if (strcmp(cmd, "browse back") == 0) {
    std::cout << "SBrowser: browser history size =" << _historyQ.size()
              << std::endl;

    if (_historyQ.size() > 1) {
      _historyQ.pop_back();
      _location.pop_back();
      // delete _currentQ;
      _currentQ = _historyQ[_historyQ.size() - 1];
      std::list<Queriable*> const& qbls = _currentQ->getQueriableList();
      sendQbls(qbls);
      sendDescriptor(_currentQ->getQueryDescriptor());
    }
  } else if (cmdLine == cmd_browse_select_queriable) {

    //  if (strncmp(cmd, "browse select queriable", 23) == 0) {
    unsigned selection = atoi(cmd + 23);
    std::list<Queriable*> qlist = _currentQ->getQueriableList();
    std::list<Queriable*>::iterator itr;

    if ((selection >= 0) && (selection < qlist.size())) {
      if (qlist.size() == 0) {
      }
      itr = qlist.begin();
      for (unsigned ct = 0; ct <= selection; ct++) {
        if (selection == ct) {
          _currentQ = (*itr);
        }
        itr++;
      }

      std::unique_ptr<DataItem> d;
      _currentQ->getDataItem(d);
      TriggerDataItem* tdi = dynamic_cast<TriggerDataItem*>(d.get());

      if (tdi) {
        std::cout
            << "SBrowser: got a TriggerDataItem, sending to client to display."
            << std::endl;
        std::ostringstream os;
        os << "display browser trigger_value <>Description: ";
        os << tdi->getTrigger()->getDescription() << std::endl << "Value: ";
        os << tdi->getTrigger()->status() << "\n";
        os << "Delay: " << tdi->getTrigger()->getDelay() << "<>";
        // TUAN: temporary disable the code
        /*
char outbuff[MAXBUFFSIZE];
strcpy(outbuff, os.str().c_str());
_UI->getServer()->sendToClient(outbuff);
        */
      }
      std::list<Queriable*> const& qbls = _currentQ->getQueriableList();
      _historyQ.push_back(_currentQ);
      const char* n = strdup(_currentQ->getQueriableDescriptor().getName().c_str());
      _location.push_back(n);

      sendQbls(qbls);
      sendDescriptor(_currentQ->getQueryDescriptor());

      if (_currentQ->isPublisherQueriable()) {
        Publisher* p = _currentQ->getQPublisher();
        _UI->getServer()->sendToClient("display browser clear publisher");
        sendServices(p);
        sendTriggers(p);
      } else {
        _UI->getServer()->sendToClient("display browser clear publisher");
      }
    }
  } else if (cmdLine == cmd_browse_select_publisher) {

    //  if (strncmp(cmd, "browse select publisher", 23) == 0) {
    unsigned selection = atoi(cmd + 23);

    if ((_result != NULL) && (selection < _result->size())) {
      std::unique_ptr<Queriable> dup;
      (*_result)[selection]->duplicate(dup);
      Queriable* q = dup.release();
      Publisher* p = q->getQPublisher();

      _UI->getServer()->sendToClient("display browser clear publisher");
      sendTriggers(p);
      sendServices(p);
    }
  } else if (cmdLine == cmd_browse_select_queriable_publisher) {

    //  if (strncmp(cmd, "browse select queriable publisher", 33) == 0) {
    unsigned selection = atoi(cmd + 33);
    std::list<Queriable*> qlist = _currentQ->getQueriableList();

    std::list<Queriable*>::iterator itr;

    if ((selection >= 0) && (selection < qlist.size())) {
      itr = qlist.begin();
      for (unsigned ct = 0; ct <= selection; ct++) {
        if (selection == ct) {
          Publisher* p = (*itr)->getQPublisher();
          sendServices(p);
          sendTriggers(p);
        }
        itr++;
      }
    }
  } else if (cmdLine == cmd_browse_select_result) {

    //  if (strncmp(cmd, "browse select result", 20) == 0) {
    unsigned selection = atoi(cmd + 20);

    if ((_result != NULL) && (selection < _result->size())) {
      std::unique_ptr<Queriable> dup;
      (*_result)[selection]->duplicate(dup);
      _currentQ = dup.release();
      _historyQ.push_back(_currentQ);
      const char* n = strdup(_currentQ->getQueriableDescriptor().getName().c_str());
      _location.push_back(n);

      std::list<Queriable*> const& qbls = _currentQ->getQueriableList();
      sendNameLocation();
      sendQbls(qbls);
      sendDescriptor(_currentQ->getQueryDescriptor());

      if (_currentQ->isPublisherQueriable()) {
        Publisher* p = _currentQ->getQPublisher();
        _UI->getServer()->sendToClient("display browser clear publisher");
        sendServices(p);
        sendTriggers(p);
      } else {
        _UI->getServer()->sendToClient("display browser clear publisher");
      }
    }
  } else if (cmdLine == cmd_browse_get_queriable_dataitem) {

    //  if (strncmp(cmd, "browse get queriable dataitem", 29) == 0) {
    unsigned selection = atoi(cmd + 29);
    std::list<Queriable*> qlist = _currentQ->getQueriableList();

    std::list<Queriable*>::iterator itr;

    // TUAN: temporary disable the code
    /*
if ((selection >= 0) && (selection < qlist.size())) {
  itr = qlist.begin();
  for (unsigned ct = 0; ct <= selection; ct++) {
    if (selection == ct) {
      std::unique_ptr<DataItem> d;
      (*itr)->getDataItem(d);
      char dataitembuff[MAXBUFFSIZE];
      strcpy(dataitembuff, "display dataitem ");
      strcat(dataitembuff, d->getType());

      _UI->getServer()->sendToClient(dataitembuff);
      _UI->getServer()->keepDataItem(d);
    }
    itr++;
  }
}
    */
  } else if (cmdLine == cmd_browse_get_result_dataitem) {

    //  if (strncmp(cmd, "browse get result dataitem", 26) == 0) {
    unsigned selection = atoi(cmd + 20);

    if ((_result != NULL) && (selection < _result->size())) {
      // TUAN: temporary disable the code
      /*
std::unique_ptr<Queriable> dup;
(*_result)[selection]->duplicate(dup);
Queriable* q = dup.release();
std::unique_ptr<DataItem> d;
q->getDataItem(d);
char dataitembuff[MAXBUFFSIZE];
strcpy(dataitembuff, "display dataitem ");
strcat(dataitembuff, d->getType());

_UI->getServer()->sendToClient(dataitembuff);
_UI->getServer()->keepDataItem(d);
*/
    }
  }
}

inline int SBrowser::parseQuery(const char* q) {
  // TUAN: temporary disable the code
  //      until all strtok, strcpy convert to C++
  /*
char* buff = new char[strlen(q) + 1];
strcpy(buff, q);

strtok(buff, "<>");  // header
strtok(NULL, "<>");  // max
strtok(NULL, "<>");  // min
strtok(NULL, "<>");  // size

char* label = strtok(NULL, "<>");

QueryDescriptor& qd = _currentQ->getQueryDescriptor();
std::vector<QueryField*> qfs = qd.getQueryFields();

unsigned i = 0;

while ((label != NULL) && (i <= qfs.size())) {
std::ostringstream os;
os << "display extended_error <>There was a problem with your query:<>";
os << "The sytem could not perform query. There was a mismatch when trying "
    "to fill";
os << " out the queryfields";

char* data = strtok(NULL, "<>");

if (strcmp(label, "enum") == 0) {
int enumidx = atoi(data);

std::vector<EnumEntry*> enums = qfs[i]->getEnumEntries();
if (qfs[i]->getType() != QueryField::ENUM) {
  std::cout << "SBrowser: parseQuery: Trying to select enum from "
               "something that isn't an ENUM queryfield..., it is of "
               "type ";
  std::cout << qfs[i]->getType() << std::endl;
  os << " Tried to select an enumeration from a VALUE queryfield.\n ";

  os << " The Queriable in question was: ";
  os << _currentQ->getQueriableDescriptor().getName() << "<>"
     << std::endl;
  os << "It is possible that you did not fill all the queryfileds "
        "properly. \n";
  char errbuff[MAXBUFFSIZE];
  strcpy(errbuff, os.str().c_str());
  _UI->getServer()->sendToClient(errbuff);
} else {
  qfs[i]->setField(enums[enumidx]->getValue());
}
} else {
if (qfs[i]->getType() != QueryField::VALUE) {
  std::cout
      << "SBrowser: parseQuery: This is not a VALUE queryfield, its a ";
  std::cout << qfs[i]->getType() << std::endl;
  std::cout << "SBrowser: parseQuery: queriable info: ";
  std::cout << _currentQ->getQueriableDescriptor().getName() << std::endl;

  os << " Tried to set a value of an ENUM queryfield. \n";
  os << " The Queriable in question was: ";
  os << _currentQ->getQueriableDescriptor().getName() << "<>"
     << std::endl;
  os << "It is possible that you did not fill all the queryfields "
        "properly.\n";
  char errbuff[MAXBUFFSIZE];
  strcpy(errbuff, os.str().c_str());
  _UI->getServer()->sendToClient(errbuff);
} else {
  qfs[i]->setField(data);
}
}
i++;
label = strtok(NULL, "<>");
}
  */
  return (1);
}

inline std::string SBrowser::addColon(std::string str) {
  if (str != "") str = " : " + str;
  return str;
}

inline std::string SBrowser::bracket(std::string str, int i) {
  if (str != "") {
    if (i == 1) str = "[" + str + "] ";
    if (i == 2) str = " <" + str + "> ";
    if (i == 3) str = "  (" + str + ") ";
    if (i == 4) str = "   {" + str + "} ";
  }
  return str;
}

inline std::string SBrowser::bracket(int j, int i) {
  std::ostringstream ostr;
  ostr << j;
  std::string str = ostr.str();
  if (i == 1) str = "[" + str + "] ";
  if (i == 2) str = " <" + str + "> ";
  if (i == 3) str = "  (" + str + ") ";
  if (i == 4) str = "   {" + str + "} ";
  return str;
}

inline std::string SBrowser::getCommand() {
  std::string cmd = _UI->getCommand();

  if (cmd == "control pause") {
    cmd = _UI->getCommand();
  }

  if ((cmd == "control start") || (cmd == "control resume") ||
      (cmd == "control stop")) {
    _i = 0;
  }
  return cmd;
}

inline int SBrowser::getSelection() {
  int rval = 0;
  std::string cmd = getCommand();
  const char* s = cmd.c_str();

  if (cmd.size() > 13) {
    std::string num(s + 13);  // skipping over the "browse select" part

    std::istringstream istr(num);
    istr >> rval;
    if (!rval) {
      rval = 0;
    }
    return rval;
  } else {
    return 0;
  }
}

inline int SBrowser::getSelection(const std::string& cmd) {
  const char* s = cmd.c_str();

  int selection = atoi(s);

  return (selection);

  std::string cmd_control_stop("control stop");
  std::string cmd_control_resume("control resume");
  if ((cmd == cmd_control_stop) || (cmd == cmd_control_resume)) {
    //  if ((strcmp(s, "control stop") == 0) || (strcmp(s, "control resume") ==
    // 0)) {
    return 0;
  }

  return (0);
}

inline bool SBrowser::isMatch(std::string input, std::string comp) {
  bool rval = false;
  std::string::iterator inp_iter = input.begin();
  std::string::iterator cmp_iter = comp.begin();
  std::string::iterator cmp_end;

  if (input == comp)
    rval = true;
  else if (((*inp_iter) == (*cmp_iter)) && (input.length() == 1))
    rval = true;
  else if (((*inp_iter) + 32 == (*cmp_iter)) && (input.length() == 1))
    rval = true;
  else {
    rval = true;
    if (input.length() == comp.length()) {
      inp_iter = input.begin();
      cmp_end = comp.end();
      for (cmp_iter = comp.begin(); cmp_iter != cmp_end; cmp_iter++) {
        if (((*inp_iter) != (*cmp_iter)) && ((*inp_iter) + 32 != (*cmp_iter)))
          rval = false;
        inp_iter++;
      }
    } else
      rval = false;
  }
  return rval;
}

inline void SBrowser::action() {
  std::cout << "SBrowser: Action()!.. " << std::endl;
  ;
  if (_UI->getServer()->isConnected()) {
    std::cout << "SBrowser: Browsing now..." << std::endl;
    browse();
  } else {
    std::cout << "SBrowser: not connected, so not browsing!" << std::endl;
  }
}

#endif
