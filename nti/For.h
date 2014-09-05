// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

/*
 * For.h
 *  Created on: Jun 6, 2010
 *      Author: wagnerjo
 */

#ifndef FOR_H_
#define FOR_H_

#include <pthread.h>
#include <iostream>

class Mutex {
	public:
		Mutex() {
			//fieldLock = PTHREAD_MUTEX_INITIALIZER;
			if(pthread_mutex_init(getLock(), NULL) != 0) {
				// Stall one cleanup...
				//std::cerr << "Mutex::Mutex() : COULD NOT INITIALIZE MUTEX!" << std::endl;
			  assert(0);
			}
		}
		~Mutex() {
			pthread_mutex_destroy(getLock());
		}
		bool lock() {
			return(pthread_mutex_lock(getLock()) == 0);
		}
		bool unlock() {
			return(pthread_mutex_unlock(getLock()) == 0);
		}
		int trylock() {
			return(pthread_mutex_trylock(getLock()));
		}
	private:
		Mutex(const Mutex &);
		void operator=(const Mutex&);
		pthread_mutex_t fieldLock;
		pthread_mutex_t *getLock() { return(&fieldLock); };
};

template <class WORKER, class USERDATA>
class For {
	private:
		int fieldThread, fieldLo, fieldBy, fieldHi;
		WORKER *fieldWorker;
		USERDATA *fieldUserData;
		Mutex *fieldMutex;
		//
		For(int thread, int lo, int by, int hi, WORKER *worker, USERDATA *userData, Mutex *mutex) {
			fieldThread = thread;
			fieldLo = lo;
			fieldBy = by;
			fieldHi = hi;
			fieldWorker = worker;
			fieldUserData = userData;
			fieldMutex = mutex;
		}
		virtual ~For() {}
		int getThread() {
			return(fieldThread);
		}
		int getLo() {
			return(fieldLo);
		}
		int getBy() {
			return(fieldBy);
		}
		int getHi() {
			return(fieldHi);
		}
		WORKER *getWorker() {
			return(fieldWorker);
		}
		USERDATA *getUserData() {
			return(fieldUserData);
		}
		Mutex *getMutex() {
			return(fieldMutex);
		}
		//
		static void *Work(void *forLoop) {
			return(Work((For<WORKER,USERDATA> *) forLoop));
		}
		static void *Work(For<WORKER,USERDATA> *forLoop) {
			int lo = forLoop->getLo(), by = forLoop->getBy(), hi = forLoop->getHi();
			//std::cout << "Hello World!!! Lo = " << lo << " By = " << by << " Hi = " << hi << std::endl;
			for (int i = lo; i < hi; i += by) {
				forLoop->getWorker()->doWork(forLoop->getThread(), i, forLoop->getUserData(), forLoop->getMutex());
			}
			return(NULL);
		}
		// Recursive...
		static int execute(int lo, int by, int hi, WORKER *worker, USERDATA *userData, int THREADS, int t, Mutex *mutex) {
			For<WORKER,USERDATA> forLoop(t, lo + t*by, THREADS*by, hi, worker, userData, mutex);
			if (t == THREADS - 1) {
				Work(&forLoop);
			} else {
				pthread_t thread;
				int rc = 0;
				//std::cout << "Creating thread " << t << std::endl;
				if ((rc = pthread_create(&thread, NULL, Work, &forLoop)) != 0) {
					std::cout << "ERROR; return code from pthread_create() is " << rc << std::endl;
					return(rc);
				}
				if ((rc = execute(lo, by, hi, worker, userData, THREADS, t + 1, mutex)) != 0) {
					std::cout << "ERROR; return code from execute() is " << rc << std::endl;
					return(rc);
				}
				if ((rc = pthread_join(thread, NULL)) != 0) {
					std::cout << "ERROR; return code from pthread_join() is " << rc << std::endl;
					return(rc);
				}
				//pthread_detach(thread);
			}
			return(0);
		}
	public:
		// Recursive...
		static int execute(int lo, int by, int hi, WORKER *worker, USERDATA *userData = NULL, int THREADS = 1) {
			Mutex mutex;
			return(execute(lo, by, hi, worker, userData, THREADS, 0, &mutex));
		}
		// Non-recursive...
		/*static int execute(int lo, int hi, WORKER *worker, int THREADS = 1) {
			pthread_t *threads = new pthread_t[THREADS - 1];
			For<WORKER> **forLoops = new For<WORKER> *[THREADS - 1];
			int rc = 0, Delta = (int) ceil(double(hi - lo)/double(THREADS));
			for(int t = 0; t < THREADS - 1; t++) {
				std::cout << "Creating thread " << t << std::endl;
				forLoops[t] = new For<WORKER>(lo + t*Delta, lo + (t + 1)*Delta, worker);
				if ((rc = pthread_create(&threads[t], NULL, Work, forLoops[t])) != 0) return(rc);
			}
			for (int i = lo + (THREADS - 1)*Delta; i < hi; i++) worker->Work(i);
			for(int t = 0; t < THREADS - 1; t++) {
				if ((rc = pthread_join(threads[t], NULL)) != 0) return(rc);
				delete forLoops[t];
			}
			delete[] (threads);
			delete[] (forLoops);
			return(rc);
		}*/
};

#endif /* FOR_H_ */
