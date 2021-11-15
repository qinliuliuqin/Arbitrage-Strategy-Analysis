#include <stdlib.h>
#include <iostream> 
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <queue>

#include "utils/Pair.cpp"
#include "dict/DictConfig.cpp"
#include "dict/JointDictionary.cpp"
#include "dict/MonoDictionary.cpp"
#include "CLLDAConfig.cpp"

int *LoadIndices(string sFilePath, int *iNoTokens=NULL, int nTokens=0){
	int *aIndices = NULL;
	vector<int> vIndices;
	bool bVector;
	if(nTokens <= 0){
		nTokens = 0;
		bVector = true;
	}else{
		aIndices = new int[nTokens];
		bVector = false;
	}
	
	int tokenIdx = 0;
	ifstream fin(sFilePath.c_str());
	if(fin.fail())
            cerr << "Unable to open file " << sFilePath << " to load indices" << endl;    
	else{
		while(! fin.eof()){
			string s;
			getline(fin,s);

			if(s.empty())
				continue;

			int iIdx;
			sscanf(s.c_str(),"%d",&iIdx);
			if(bVector){
				while(vIndices.size() <= tokenIdx)
					vIndices.push_back(-1);
				vIndices[tokenIdx] = iIdx;
			}else{
				aIndices[tokenIdx] = iIdx;
			}
			tokenIdx += 1;
		}
	}
	
	if(nTokens != 0 && nTokens != tokenIdx){
		printf( "No tokens scanned (%d) didn't match with existing count (%d)\n",tokenIdx,nTokens);
		exit(-1);
	}
	nTokens = tokenIdx;
	if(iNoTokens != NULL)
		*iNoTokens = nTokens; 

	// Copy the contents from vector to array
	if(bVector && nTokens > 0){
		aIndices = new int[nTokens];
		for(int i=0;i<nTokens;i++)
			aIndices[i] = vIndices[i];
	}
	return aIndices;
}

Tripple<int *, int *, int *> WordDocFreqIndices(int *wIndices, int *dIndices, int iNoTokens, int &iDocWords){
	int iCurDoc = -1;
	map<int,int> wordFreqInThisDoc;
	vector<int> vwIndices, vdIndices, vFreq;
	for(int i=0;i<iNoTokens;i++){
		if(dIndices[i] != iCurDoc){
			if(iCurDoc != -1){
				for(map<int,int>::iterator iter=wordFreqInThisDoc.begin();iter!=wordFreqInThisDoc.end();++iter){
					vwIndices.push_back((*iter).first);
					vdIndices.push_back(iCurDoc);
					vFreq.push_back((*iter).second);
				}
				wordFreqInThisDoc.clear();
			}
			iCurDoc = dIndices[i];
		}
		if(wordFreqInThisDoc.find(wIndices[i]) == wordFreqInThisDoc.end())
			wordFreqInThisDoc[wIndices[i]] = 0;
		wordFreqInThisDoc[wIndices[i]] += 1;
	}
	if(iCurDoc != -1){
		for(map<int,int>::iterator iter=wordFreqInThisDoc.begin();iter!=wordFreqInThisDoc.end();++iter){
			vwIndices.push_back((*iter).first);
			vdIndices.push_back(iCurDoc);
			vFreq.push_back((*iter).second);
		}
		wordFreqInThisDoc.clear();
	}
	iDocWords = vwIndices.size();
	int *w = (int *)calloc(iDocWords, sizeof(int));
	int *d = (int *)calloc(iDocWords, sizeof(int));
	int *f = (int *)calloc(iDocWords, sizeof(int));
	int iTotal = 0;
	for(int i=0;i<iDocWords;i++){
		w[i] = vwIndices[i];
		d[i] = vdIndices[i];
		f[i] = vFreq[i];
		iTotal += vFreq[i];
	}
	if(iTotal != iNoTokens){
		fprintf(stderr, "Total tokens didn't match (%d != %d)\n", iTotal, iNoTokens);
		exit(1234);
	}
	return Tripple<int *, int *, int *>(w, d, f);
}

int *ExtendIndices(int *aFirst, int iNoFirstTokens, int *aSecond, int iNoSecondTokens,int *iTotalNoTokens=NULL){
	int *indices = new int[iNoFirstTokens+iNoSecondTokens];

	int idx = 0, iMaxIdx=-1;
	for(int i=0;i<iNoFirstTokens;i++){
		indices[idx++] = aFirst[i];
		if(iMaxIdx == -1 || iMaxIdx < aFirst[i])
			iMaxIdx = aFirst[i];
	}
	if(iMaxIdx == -1)
		iMaxIdx = 0;
	for(int i=0;i<iNoSecondTokens;i++)
		indices[idx++] = (aSecond[i] + iMaxIdx);
	if(iTotalNoTokens != NULL)
		*iTotalNoTokens = idx;
	return indices;
}

int MaxIdx(int *aFirst, int iNoFirstTokens, int *aSecond, int iNoSecondTokens){
	int iMaxIdx=0;
	for(int i=0;i<iNoFirstTokens;i++)
		if(iMaxIdx < aFirst[i])
			iMaxIdx = aFirst[i];
	for(int i=0;i<iNoSecondTokens;i++)
		if(iMaxIdx < aSecond[i])
			iMaxIdx = aSecond[i];
	return iMaxIdx;
}

// WS = [WS;load(t_WS)+max(WS)];
int *ExtendIndices(int *aFirst, int iNoFirstTokens, int *aSecond, int iNoSecondTokens,int iTgtOffset,int *iTotalNoTokens=NULL){
	int *indices = new int[iNoFirstTokens+iNoSecondTokens];

	int idx = 0, iMaxIdx=-1;
	for(int i=0;i<iNoFirstTokens;i++){
		indices[idx++] = aFirst[i];
		if(iMaxIdx == -1 || iMaxIdx < aFirst[i])
			iMaxIdx = aFirst[i];
	}
	if(iMaxIdx == -1)
		iMaxIdx = 0;
	if(iTgtOffset < iMaxIdx){
		fprintf(stderr, "Source indices (%d) cross the tgt offset (%d)\n", iMaxIdx, iTgtOffset);
		fprintf(stderr, "May lead to data inconsistencies\n");
	}
	for(int i=0;i<iNoSecondTokens;i++)
		indices[idx++] = (aSecond[i] + iTgtOffset);
	if(iTotalNoTokens != NULL)
		*iTotalNoTokens = idx;
	return indices;
}

int *DocLengths(const int *d,int D,int n){
    int *docLength = new int[D];
    for(int i=0;i<D;i++) docLength[i]=0;
    for(int i=0;i<n;i++){
        docLength[d[i]]++;
    }
    return docLength;
}

Pair<int *, int *> AddNewWordForEachDoc(int *wordIndices, int *docIndices, int &iNoTokens, int iNewWordId){
	int iMaxDocIdx = -1*(int)INFINITY, iMinDocIdx = (int)INFINITY;
	for(int i=0;i<iNoTokens;i++){
		if(iMaxDocIdx < docIndices[i])
			iMaxDocIdx = docIndices[i];
		if(iMinDocIdx > docIndices[i])
			iMinDocIdx = docIndices[i];
	}
	if(iMaxDocIdx == -1*(int)INFINITY || iMinDocIdx == (int)INFINITY){
		fprintf(stderr, "There are no documents to add\n");
		exit(1234);
	}
	int iNoDocs = iMaxDocIdx-iMinDocIdx+1;
	int iNoNewTokens = iNoTokens+iNoDocs;
	int *newWordIndices = (int *)calloc(iNoNewTokens, sizeof(int));
	int *newDocIndices = (int *)calloc(iNoNewTokens, sizeof(int));
	memcpy(newWordIndices, wordIndices, iNoTokens*sizeof(int));
	memcpy(newDocIndices, docIndices, iNoTokens*sizeof(int));
	int i, idocIdx;
	for(i=iNoTokens, idocIdx=iMinDocIdx;i<iNoNewTokens;i++,idocIdx++){
		newWordIndices[i] = iNewWordId;
		newDocIndices[i] = idocIdx; 
	}
	if(idocIdx < iMaxDocIdx){
		fprintf(stderr, "Should have added a NULL word for each doc -- Added only till %d but there are %d docs\n", idocIdx, iMaxDocIdx);
		exit(1234);
	}
	iNoTokens = iNoNewTokens;
	return Pair<int *, int *>(newWordIndices, newDocIndices);
}

int *UnigramFreqs(int *WordIndices, int iNoTokens, int iNoWords){
	if(iNoTokens <= 0)
		return NULL;
	int *UniFreqs = (int *)calloc(iNoWords, sizeof(int));
	for(int idx=0;idx<iNoTokens;idx++)
		UniFreqs[WordIndices[idx]]+=1;
	return UniFreqs;
}

int *BigramFreqs(int *WordIndices, int iNoTokens, int iNoWords){
	if(iNoTokens <= 0)
		return NULL;
	int *BigramFreqs = (int *)calloc(iNoWords*iNoWords, sizeof(int));
	int wprev = WordIndices[0], w;
	for(int idx=1;idx<iNoTokens;idx++){
		w = WordIndices[idx];
		BigramFreqs[wprev*iNoWords+w]++;
		wprev = w;
	}
	return BigramFreqs;
}

double *UnigramProbs(int *WordIndices, int iNoTokens, int iNoWords, double dBeta = 0){
	if(iNoTokens <= 0)
		return NULL;
	double *UniProbs = (double *)calloc(iNoWords, sizeof(double));
	for(int idx=0;idx<iNoTokens;idx++)
		UniProbs[WordIndices[idx]]+=1;
	double dTotalTemp = 0, dWBeta = iNoWords*dBeta;
	for(int idx=0;idx<iNoWords;idx++){
		double dProb = (((double)UniProbs[idx] + dBeta)/((double)iNoTokens + dWBeta));
		UniProbs[idx] = dProb;
		dTotalTemp += UniProbs[idx];
	}
	if(fabs(dTotalTemp-1) > 1e-8){
		fprintf(stderr, "Sum of unigram probability of words :%e != 1\n", dTotalTemp);
		exit(1234);
	}
	return UniProbs;
}

vector<vector<Pair<int,float> > > ContextFrequencyCounts(int *WordIndices, int *DocIndices, int iNoTokens, int iNoWords, int iNoTopWords=5, bool bIncludeItsOWN=false){
	vector<vector<Pair<int, float> > > vPMI;
	if(iNoTokens <= 0)
		return vPMI;

	int iCurrentDoc = DocIndices[0], iWindowSize = 2, iWord, iExistWord, iTotalPairs = 0;
	vector<int> windowWords;
	vector<map<int,float> > CoOccurrenceCounts;
	bool windowFull = false;
	int *wordFreq = (int *)calloc(iNoWords, sizeof(int));
	for(int i=0;i<iNoWords;i++)
		CoOccurrenceCounts.push_back(map<int,float>());
	for(int idx=0;idx<iNoTokens;idx++){
		iWord = WordIndices[idx];
		wordFreq[iWord]++;
		if(iCurrentDoc != DocIndices[idx]){	// New doc started 
			windowWords.clear();
			windowFull = false;
			iCurrentDoc = DocIndices[idx];
		}
		if(! windowFull && iWindowSize == windowWords.size())
			windowFull = true;
		if(windowFull)
			windowWords.erase(windowWords.begin());
		for(int i=0;i<windowWords.size();i++){
			iExistWord = windowWords[i];
			CoOccurrenceCounts[iExistWord][iWord]++;
			CoOccurrenceCounts[iWord][iExistWord]++;
			iTotalPairs += 2;
		}
		if(bIncludeItsOWN){
			CoOccurrenceCounts[iWord][iWord]++;
			iTotalPairs +=1;
		}
		windowWords.push_back(iWord);
	}

	Pair<int,float>::Compare pDecreasingOrder(true);
	Pair<int,float> pTop;
	double fConst = 2*log((double)iNoTokens) - log((double)iTotalPairs);
	if(! isfinite(fConst)){
		fprintf(stderr, "The constant became %e : log(%d * %d / %d)\n", fConst, iNoTokens, iNoTokens, iTotalPairs);
		exit(1234);
	}
	for(int i=0;i<iNoWords;i++){
		priority_queue<Pair<int,float>, vector<Pair<int,float> >, Pair<int,float>::Compare> pqTopPMI(pDecreasingOrder);
		for(map<int, float>::iterator iter = CoOccurrenceCounts[i].begin(); iter != CoOccurrenceCounts[i].end(); ++iter){
			int j = (*iter).first;
			float freq = (*iter).second;
			if(freq <= 0)
				continue;
			float fMI = log((float)(freq)/(float)(wordFreq[i] * wordFreq[j])); 
			fMI += fConst;
			pqTopPMI.push(Pair<int,float>(j,fMI));
		}
		vector<Pair<int, float> > vWords;
		for(int k=0;k < iNoTopWords && ! pqTopPMI.empty();k++){
			pTop = pqTopPMI.top();
			vWords.push_back(pTop);
			pqTopPMI.pop();
		}
		while(! pqTopPMI.empty())
			pqTopPMI.pop();
		vPMI.push_back(vWords);
	}
	return vPMI;
}

vector<Pair<int,int> *> WordDocFreq(int *WordIndices, int *DocIndices, int iNoTokens){
	vector<map<int,int> > vmWordDocFreq;
	for(int idx=0;idx<iNoTokens;idx++){
		int wi = WordIndices[idx], di = DocIndices[idx];
		while(vmWordDocFreq.size() <= wi)
			vmWordDocFreq.push_back(map<int,int>());
		if(vmWordDocFreq[wi].find(di) == vmWordDocFreq[wi].end())
			vmWordDocFreq[wi][di] = 1;
		else
			vmWordDocFreq[wi][di]++;
	}

	// Change the data structure
	int iTokensInserted = 0, iNoWords = vmWordDocFreq.size();
	vector<Pair<int,int> *> vpWordDocFreq;
	for(int iWordId=0;iWordId<iNoWords;iWordId++){
		while(vpWordDocFreq.size() <= iWordId)
			vpWordDocFreq.push_back(NULL);

		for(map<int,int>::iterator iter=vmWordDocFreq[iWordId].begin(); iter != vmWordDocFreq[iWordId].end(); ++iter){
			Pair<int,int> *temp = new Pair<int,int>((*iter).first,(*iter).second);
			temp->Next = vpWordDocFreq[iWordId];
			vpWordDocFreq[iWordId] = temp;
		}
	}

	for(int i=0;i<iNoWords;i++)
		for(Pair<int,int> *p = vpWordDocFreq[i];p!=NULL;p=p->Next)
			iTokensInserted += p->Second;
	if(iNoTokens != iTokensInserted)
		cerr << "Mismatch in the number of tokens inserted "<<iTokensInserted << " and to be inserted " << iNoTokens << endl;
	return vpWordDocFreq;
}

vector<string> LoadWords(string sFilePath){
	vector<string> vWords;
	
	ifstream fin(sFilePath.c_str());
	if(fin.fail())
            cerr << "Unable to open file " << sFilePath << " to load indices" << endl;    
        else{
            while(! fin.eof()){
                string s;
                getline(fin,s);

                if(s.empty())
                    continue;

				char *buf = new char[s.length()+1];
				sscanf(s.c_str(),"%s",buf);
				string sWord = string(buf);
				vWords.push_back(sWord);
            }
        }
	
	return vWords;
}

// WO = [WO;load(t_WO)+max(WO)];
vector<string> ExtendWords(vector<string> aFirst, vector<string> aSecond ){
	vector<string> Words;

	int iFirstVectorSize = aFirst.size(), iSecondVectorSize = aSecond.size();
	for(int i=0;i<iFirstVectorSize;i++)
		Words.push_back(aFirst[i]);
	
	for(int i=0;i<iSecondVectorSize;i++)
		Words.push_back(aSecond[i]);

	return Words;
}

Pair<int *,int *> LoadTestData(CLLDAConfig ldaConfig, int &iTestTokens, int &W, int &D){

    int *wTest, *dTest;
    if(ldaConfig.m_bFoundTestData == false){
        wTest = NULL;
        dTest = NULL;
        return Pair<int *,int *>(wTest,dTest);
    }

    int *SrcWordTestIndices, *TgtWordTestIndices, *SrcDocTestIndices, *TgtDocTestIndices;
    int iNoSrcWordTestTokens=0,iNoTgtWordTestTokens=0,iNoSrcDocTestTokens=0,iNoTgtDocTestTokens=0,iNoWordTokens=0,iNoDocTokens=0;
    if(! ldaConfig.m_sSrcWordTestIndicesPath.empty()){
        SrcWordTestIndices = LoadIndices(ldaConfig.m_sSrcWordTestIndicesPath,&iNoSrcWordTestTokens);
        SrcDocTestIndices = LoadIndices(ldaConfig.m_sSrcDocTestIndicesPath,&iNoSrcDocTestTokens);
        if(iNoSrcWordTestTokens != iNoSrcDocTestTokens){
            fprintf(stderr, "No src word test tokens (%d) != No src doc test tokens (%d)\n", iNoSrcWordTestTokens, iNoSrcDocTestTokens);
            exit(1234);
        }
    }

    if(! ldaConfig.m_sTgtWordTestIndicesPath.empty()){
        TgtWordTestIndices = LoadIndices(ldaConfig.m_sTgtWordTestIndicesPath,&iNoTgtWordTestTokens);
        TgtDocTestIndices = LoadIndices(ldaConfig.m_sTgtDocTestIndicesPath,&iNoTgtDocTestTokens);
        if(iNoTgtWordTestTokens != iNoTgtDocTestTokens){
            fprintf(stderr, "No tgt word test tokens (%d) != No tgt doc test tokens (%d)\n", iNoTgtWordTestTokens, iNoTgtDocTestTokens);
            exit(1234);
        }
    }

    wTest = ExtendIndices(SrcWordTestIndices,iNoSrcWordTestTokens,TgtWordTestIndices,iNoTgtWordTestTokens,&iNoWordTokens);
    dTest = ExtendIndices(SrcDocTestIndices,iNoSrcDocTestTokens,TgtDocTestIndices,iNoTgtDocTestTokens,&iNoDocTokens);
    if(iNoWordTokens != iNoDocTokens){
        fprintf(stderr, "No word test tokens (%d) != No doc test tokens (%d)\n",iNoWordTokens, iNoDocTokens);
        exit(1234);
    }
    iTestTokens = iNoWordTokens;

    // copy over the word and document indices into internal format
    for (int i=0; i<iTestTokens; i++) {
        wTest[i] = wTest[i] - 1;
        dTest[i] = dTest[i] - 1;
    }

    int wPrev = W, dPrev = D;
    for (int i=0; i<iTestTokens; i++) {
        if (wTest[ i ] > W) W = wTest[ i ];
        if (dTest[ i ] > D) D = dTest[ i ];
    }
    if(wPrev != W)
        W = W + 1;
    if(dPrev != D)
        D = D + 1;

    Pair<int *,int *> testData(wTest,dTest);
    return testData;
}

Vocab VocabFrmVector(vector<string> &vWords, bool *bIncludeFlag = NULL){
    Vocab v = Vocab();
	if(bIncludeFlag == NULL){
		for(int i=0;i<vWords.size();i++)
			v.AddWord(vWords[i]);
	}else{
		for(int i=0;i<vWords.size();i++)
			if(bIncludeFlag[i])
				v.AddWord(vWords[i]);
	}
    return v;
}

MonoDictionary PrepareDictionary(MonoDictConfig dc, vector<string> &Words){
	fprintf(stderr, "Initializing the monodictionary with empty string arguments\n");
	MonoDictionary md;
	StopWordList swl(dc.m_sSwlPath);
	Vocab v = VocabFrmVector(Words);
	md.RestrictVocab(v);

	md.LoadDictionary(swl, dc.m_dThreshold);
	return md;
}

JointDictionary PrepareDictionary(DictConfig dc, vector<string> &SrcWords, vector<string> &TgtWords, bool *bSrcTrainFlag = NULL, bool *bTgtTrainFlag = NULL){

    JointDictionary jd(dc.m_sSrcVcbPath,dc.m_sTgtVcbPath,dc.m_sSrc2TgtDictPath,dc.m_sTgt2SrcDictPath);
    StopWordList swlSrc(dc.m_sSrcSwlPath), swlTgt(dc.m_sTgtSwlPath);
    Vocab vSrc = VocabFrmVector(SrcWords, bSrcTrainFlag), vTgt = VocabFrmVector(TgtWords, bTgtTrainFlag);
    jd.RestrictVocab(&vSrc,&vTgt);

    jd.LoadDictionary(swlSrc,swlTgt,dc.m_dThreshold);
    //jd.LoadDictionary(dc.m_dThreshold);

    return jd;
}

bool *WordsOccurredInTrainingData(int iNoWords, int *Indices, int iNoTokens, int IdxOffset=0){
	bool *bTrainFlag = (bool *)calloc(iNoWords, sizeof(bool));
	for(int i=0;i<iNoTokens;i++)
		bTrainFlag[Indices[i]+IdxOffset] = true;
	return bTrainFlag;
}

void PrintFilteredWords(int *iUnigramFreqs, int W, vector<string> words, int iThreshold=-1){
	for(int i=0;i<W;i++)
		if(iUnigramFreqs[i] > iThreshold)
			cout << words[i] << endl;
}

void PrintTopicWords(const vector<string> &Words, bool *bSrcLang, const double *wp, int W, int T,int iTopic=-1,int iNoWords=10){
    double dAvgMixedWordPos = 0;
    Pair<int,double>::Compare pDecreasingOrder(true);
    priority_queue<Pair<int,double>, vector<Pair<int,double> >, Pair<int,double>::Compare> pqTopicWords(pDecreasingOrder);
    for(int k=0;k<T;k++){
        while(pqTopicWords.empty() == false)
            pqTopicWords.pop();
        for(int w=0;w<W;w++)
            pqTopicWords.push(Pair<int,double>(w,wp[w*T+k]));
        if(pqTopicWords.size() != W)
            cerr << "Strangley, number of words pushed " << pqTopicWords.size() << " != " << W << endl;
        if(iNoWords < 0 || iNoWords > W)
            iNoWords = W;
        printf("\n\nTopic %d\n",k+1);
        bool bFirstWordLang = bSrcLang[pqTopicWords.top().First];
        int iPosOfFirstMixedLangWord = -1;
		double dFirstMixedLangWordProb = 0;
        //for(int i=0;i<iNoWords;i++){
        for(iPosOfFirstMixedLangWord = 1;bSrcLang[pqTopicWords.top().First] == bFirstWordLang;iPosOfFirstMixedLangWord++){
            if(pqTopicWords.empty())
                break;
            Pair<int,double> pTop = pqTopicWords.top();
            if(iPosOfFirstMixedLangWord <= iNoWords)
                printf("\t\t[%d]=%s (%d),%lf\n",pTop.First,Words[pTop.First].c_str(),bSrcLang[pTop.First], pTop.Second);
            pqTopicWords.pop();
        }
		dFirstMixedLangWordProb = pqTopicWords.top().Second;
        for(int iTemp = iPosOfFirstMixedLangWord;iTemp<=iNoWords;iTemp++){
            if(pqTopicWords.empty())
                break;
            Pair<int,double> pTop = pqTopicWords.top();
            printf("\t\t[%d]=%s (%d),%lf\n",pTop.First,Words[pTop.First].c_str(),bSrcLang[pTop.First], pTop.Second);
            pqTopicWords.pop();
        }
		printf("Mixed word at %d with prob %e\n", iPosOfFirstMixedLangWord, dFirstMixedLangWordProb);
    }
    cout << "Average Mixed Word Position on " << T << " topics is " << (dAvgMixedWordPos/T) << endl;
}


double AddLog2(double a, double b){
	int iThreshold = sizeof(double) * 8;	// Number of bits used for double
	double dSum;
	if(a - b > iThreshold)
		dSum = a;
	else if (a > b)
		dSum = a + log(1 + exp(b-a));
	else if (b - a > iThreshold)
		dSum = b;
	else
		dSum = b + log(1 + exp(a-b));
	//printf("%e + %e = %e\n", a, b, dSum);
	return dSum;
}

double PlainAddLog2(double a, double b, bool bDebug=false){	// returns loc(C) such that C = exp(a) + exp(b)
	if(bDebug == true){
		printf("add %e + %e\n", a, b);
	}
	double dSum;
	if(a > b)
		dSum = a + log(1 + exp(b-a));
	else
		dSum = b + log(1 + exp(a-b));
	if(bDebug)
		printf("return:%e\n", dSum);
	if(! isfinite(dSum) || ! isfinite(a) || ! isfinite(b)){
		fprintf(stderr, "Sum of log[exp(%e) + exp(%e)] = %e\n", a, b, dSum);
		//exit(1234);
	}
	return dSum;
}

double EmpiricalLikelihood(const double *dWordTopicProbs, int iNoTokens, const int *w, const int *d, int iNoDocsToSample, double *dDocTopicDistributions, double dAlpha, int T, int D){

	//int *iTestDocLengths = DocLengths(d,D,iNoTokens);

	double *dTestSampleDocProb = (double *)calloc(D*iNoDocsToSample, sizeof(double));
	double dWordSampleDocProb, dLikelihood= 0, dTestDocProb = 0;
	int wi,di,wioffset,diSampleoffset, dioffset;
	for(int iSample=0;iSample<iNoDocsToSample;iSample++){
		diSampleoffset = iSample*D; // offset for p(d_t|d_s)
		dioffset = iSample*T; // offset for p(k|d_s)

		for(int i=0;i<iNoTokens;i++){
			wi = w[i];
			di = d[i];
			dWordSampleDocProb = 0;

			wioffset = wi*T;
			for(int k=0;k<T;k++)
				dWordSampleDocProb += (dWordTopicProbs[wioffset+k] * dDocTopicDistributions[dioffset+k] );
			dWordSampleDocProb = log(dWordSampleDocProb);
			/*
			if(! isfinite(dWordSampleDocProb))
				fprintf(stderr, "Becoming %e for word:%d\n",dWordSampleDocProb, wi);
			else
				fprintf(stderr, ".");
			*/
			dTestSampleDocProb[diSampleoffset+di] += dWordSampleDocProb;
		}

	}
	//fprintf(stderr, "\n Accumulating");
	dLikelihood = 0;
	for(int iDocId=0;iDocId<D;iDocId++){
		//fprintf(stderr, "for %d", iDocId);
		//int iTestDocLength = iTestDocLengths[iDocId];
		//if(iTestDocLength == 0)
		//	continue;
		dTestDocProb = 0;
		dTestDocProb = dTestSampleDocProb[iDocId];	// This is p(d_t|d_s=0), initializing to this value is crucial
		//fprintf(stderr, "p(test:%d|sample:0)=%e ... total = %e\n", iDocId, dTestSampleDocProb[iDocId], dTestDocProb);
		/*
		double dMinTestSampleDocLValue = dTestSampleDocProb[iDocId];
		for(int iSample=0;iSample<iNoDocsToSample;iSample++)
			if(dMinTestSampleDocLValue > dTestSampleDocProb[iSample*D+iDocId])
				dMinTestSampleDocLValue = dTestSampleDocProb[iSample*D+iDocId];
		dLikelihood += dMinTestSampleDocLValue;
		*/

		for(int iSample=1;iSample<iNoDocsToSample;iSample++){	// Starting with sample 1 as dTestDocProb is initialized to sample 0 value
			diSampleoffset = iSample*D;
			dTestDocProb = PlainAddLog2(dTestDocProb, dTestSampleDocProb[diSampleoffset+iDocId]);
			//fprintf(stderr, "p(test:%d|sample:%d)=%e ... total = %e\n", iDocId, iSample, dTestSampleDocProb[diSampleoffset+iDocId], dTestDocProb);

			/*
			dTestDocProb += exp(dTestSampleDocProb[diSampleoffset+iDocId] - dMinTestSampleDocLValue);
			double dContrib = exp(dTestSampleDocProb[diSampleoffset+iDocId] - dMinTestSampleDocLValue);
			if(dContrib == 0)
				fprintf(stderr, "contrib of sample: %d for doc: %d of length:%d is exp(%e)=%e .. total = %e\n", iSample, iDocId, iTestDocLength, dTestSampleDocProb[diSampleoffset+iDocId]-dMinTestSampleDocLValue, dContrib, dTestDocProb);
			else if(! isfinite(dContrib))
				fprintf(stderr, "contrib of sample: %d for doc: %d of length:%d is exp(%e)=%e .. total = %e\n", iSample, iDocId, iTestDocLength, dTestSampleDocProb[diSampleoffset+iDocId]-dMinTestSampleDocLValue, dContrib, dTestDocProb);
			else
				fprintf(stderr, "contrib of sample: %d for doc: %d of length:%d is exp(%e)=%e .. total = %e\n", iSample, iDocId, iTestDocLength, dTestSampleDocProb[diSampleoffset+iDocId]-dMinTestSampleDocLValue, dContrib, dTestDocProb);
			*/

			/*
			dTestDocProb += exp(dTestSampleDocProb[diSampleoffset+iDocId]/iTestDocLength);
			double dContrib = exp(dTestSampleDocProb[diSampleoffset+iDocId]/iTestDocLength);
			
			if(dContrib == 0)
				fprintf(stderr, "contrib of sample: %d for doc: %d of length:%d is exp(%e)=%e .. total = %e\n", iSample, iDocId, iTestDocLength, dTestSampleDocProb[diSampleoffset+iDocId], dContrib, dTestDocProb);
			*/
		}
		dLikelihood += dTestDocProb;
		if(! isfinite(dLikelihood) || dLikelihood == 0){
			fprintf(stderr, "Likelihood became non-finite:%e after doc %d, but added only %e\n", dLikelihood, iDocId, dTestDocProb); 
			exit(1234);
		}
		/*
		dLikelihood += log(dTestDocProb);
		if(! isfinite(dLikelihood)){
			fprintf(stderr, "Likelihood became non-finite:%e after doc %d, but added only log(%e/%d) = %e\n", dLikelihood, iDocId, dTestDocProb, iNoDocsToSample, log(dTestDocProb)); 
			exit(1234);
		}
		*/
		/*
		dLikelihood += log(dTestDocProb/(double)iNoDocsToSample);
		if(! isfinite(dLikelihood)){
			fprintf(stderr, "Likelihood became non-finite:%e after doc %d, but added only log(%e/%d) = %e\n", dLikelihood, iDocId, dTestDocProb, iNoDocsToSample, log(dTestDocProb/(double)iNoDocsToSample)); 
			exit(1234);
		}
		*/
	}
	if(dTestSampleDocProb != NULL)
		free(dTestSampleDocProb);

	/*
	for(int iDocId=0;iDocId<D;iDocId++){
		dLikelihood += log(dTestDocProb[iDocId]/(double)iNoDocsToSample);
		if(! isfinite(dLikelihood)){
			fprintf(stderr, "Likelihood became non-finite:%e after doc %d, but added only log(%e/%d) = %e\n", dLikelihood, iDocId, dTestDocProb[iDocId], iNoDocsToSample, log(dTestDocProb[iDocId]/(double)iNoDocsToSample)); 
			exit(1234);
		}
	}
	*/
	//if(iTestDocLengths != NULL)
	//	free(iTestDocLengths);
	return dLikelihood;
}

bool ArtificialSrcWord(string sWord){
	return sWord.find("_Src") != string::npos ? true : false;
}

bool ArtificialTgtWord(string sWord){
	return sWord.find("_Tgt") != string::npos ? true : false;
}

bool ArtificialWord(string sWord){
	if(ArtificialSrcWord(sWord) || ArtificialTgtWord(sWord))
		return true;
	else
		return false;
}

void AddArtificialTranslations(MonoDictionary &md, vector<vector<int> > &wordTrans, vector<string> &Words, bool bForEveryWord=false){
    cout << "Number of dict entries before adding artificial translations " << md.NoEntries() << endl;
	if(bForEveryWord == false){
		for(int i=0;i<wordTrans.size();i++){
			if(wordTrans[i].size() != 0)
				continue;
			string word = Words[i];
			md.ForceAddEntry(word,word+"_Tgt",1.0);
		}   
	}else{
		for(int i=0;i<wordTrans.size();i++){
			string word = Words[i];
			md.ForceAddEntry(word,word+"_Tgt",1.0);
		}
	}
	for(int i=0;i<wordTrans.size();i++)
		wordTrans[i] = md.WordDictEntries(Words[i]);
    cout << "Final number of dict entries " << md.NoEntries() << endl;
    //return wordTrans;
}

void AddArtificialTranslations(JointDictionary &jd, vector<vector<int> > &wordTrans, vector<string> &Words, bool *bSrcLang, bool bForEveryWord=false, string sSrcNull=string(), string sTgtNull=string()){
    cout << "Number of dict entries before adding artificial translations " << jd.NoEntries() << endl;
	if(bForEveryWord == false){
		for(int i=0;i<wordTrans.size();i++){
			if(wordTrans[i].size() != 0)
				continue;
			string word = Words[i];
			if(bSrcLang[i]){
				if(word == sSrcNull)
					continue;
				if(sTgtNull.empty())
					jd.ForceAddEntry(word,word+"_Tgt",1.0);
				else
					jd.ForceAddEntry(word,sTgtNull,1.0);
			}else{
				if(word == sTgtNull)
					continue;
				if(sSrcNull.empty())
					jd.ForceAddEntry(word+"_Src",word,1.0);
				else
					jd.ForceAddEntry(sSrcNull, word, 1.0);
			}
		}   
	}else{
		for(int i=0;i<wordTrans.size();i++){
			string word = Words[i];
			if(bSrcLang[i]){
				if(word == sSrcNull)
					continue;
				if(sTgtNull.empty())
					jd.ForceAddEntry(word,word+"_Tgt",1.0);
				else
					jd.ForceAddEntry(word,sTgtNull,1.0);
			}else{
				if(word == sTgtNull)
					continue;
				if(sSrcNull.empty())
					jd.ForceAddEntry(word+"_Src",word,1.0);
				else
					jd.ForceAddEntry(sSrcNull, word, 1.0);
			}
		}
	}
	for(int i=0;i<wordTrans.size();i++)
		if(bSrcLang[i])
			wordTrans[i] = jd.SrcWordDictEntries(Words[i]);
		else
			wordTrans[i] = jd.TgtWordDictEntries(Words[i]);
    cout << "Final number of dict entries " << jd.NoEntries() << endl;
    //return wordTrans;
}

vector<vector<int> > WordTranslations(MonoDictionary &md, vector<string> &Words){
    vector<vector<int> > vTrans;
    for(int i=0;i<Words.size();i++){
        while(vTrans.size() <= i){ 
            vTrans.push_back(vector<int>());
        }
		vector<int> thisWordTrans = md.WordDictEntries(Words[i]);
        if(thisWordTrans.size() != 0)
            vTrans[i] = thisWordTrans;
    }   
    return vTrans;
}

vector<vector<int> > WordTranslations(JointDictionary &jd, vector<string> &Words, bool *bSrcLang){
    vector<vector<int> > vTrans;
    for(int i=0;i<Words.size();i++){
        while(vTrans.size() <= i){ 
            vTrans.push_back(vector<int>());
        }
        vector<int> thisWordTrans;
        if(bSrcLang[i])
            thisWordTrans = jd.SrcWordDictEntries(Words[i]);
        else
            thisWordTrans = jd.TgtWordDictEntries(Words[i]);
        /*
           cout << Words[i] << endl;
           for(int j=0;j<thisWordTrans.size();j++){
           int iEntryId = thisWordTrans[j];
           cout << "\t" << iEntryId << "\t" << jd.getString(iEntryId) << endl;
           }
         */
        if(thisWordTrans.size() != 0)
            vTrans[i] = thisWordTrans;
        //  wordTrans.push_back(& (jd.TgtWordDictEntries(Words[i])));
    }   
    return vTrans;
}

bool CheckTranslations(MonoDictionary &md, vector<vector<int> > &vTrans, vector<string> &Words){
    for(int i=0;i<vTrans.size();i++){
        vector<int> thisWordTrans = vTrans[i];
        if(thisWordTrans.size() == 0)
            continue;
        //cout << Words[i] << " has " << thisWordTrans.size() << " translations " << endl;
        for(int j=0;j<thisWordTrans.size();j++){
            int iEntryId = thisWordTrans[j];
            Tripple<string,string,double> tpTemp = md.getEntry(iEntryId);
			if(tpTemp.First != Words[i] && tpTemp.Second != Words[i]){
				fprintf(stderr, "The word %s is not a part of %s\n", Words[i].c_str(), md.getString(iEntryId).c_str());
				return false;
			}
            //  cout << "\t" << iEntryId << "\t" << jd.getString(iEntryId) << endl;
        }
    }
    return true;
}

bool CheckTranslations(JointDictionary &jd, vector<vector<int> > &vTrans, vector<string> &Words, bool *bSrcLang){
    for(int i=0;i<vTrans.size();i++){
        vector<int> thisWordTrans = vTrans[i];
        if(thisWordTrans.size() == 0)
            continue;
        bool bSrcWord = bSrcLang[i];
        //cout << Words[i] << " has " << thisWordTrans.size() << " translations " << endl;
        for(int j=0;j<thisWordTrans.size();j++){
            int iEntryId = thisWordTrans[j];
            Tripple<string,string,double> tpTemp = jd.getEntry(iEntryId);
            if(bSrcWord && tpTemp.First != Words[i]){
                cerr << " Src word of " << jd.getString(iEntryId) <<" didn't match with " << Words[i] << endl;
                return false;
            }
            else if (bSrcWord == false && tpTemp.Second != Words[i]){
                cerr << " Tgt word of " << jd.getString(iEntryId) <<" didn't match with " << Words[i] << endl;
                return false;
            }
            //  cout << "\t" << iEntryId << "\t" << jd.getString(iEntryId) << endl;
        }
    }
    return true;
}

void AddCognateTranslations(JointDictionary &jd, vector<string> &srcWords, vector<string> &tgtWords){
	for(int i=0;i<srcWords.size();i++){
		string sSrcWord = srcWords[i];
		for(int j=0;j<tgtWords.size();j++)
			if(sSrcWord == tgtWords[j]){
				jd.ForceAddEntry(sSrcWord, tgtWords[j], 1.0);
				break;
			}
	}
}

void AddRandomTranslations(JointDictionary &jd, vector<string> &srcWords, vector<string> &tgtWords, int iNoRandTrans=2){
    cout << "Number of dict entries before adding random translations " << jd.NoEntries() << endl;
	int iTotalNoSrcWords = srcWords.size(), iTotalNoTgtWords = tgtWords.size(), iExistingTrans;
	double dVal;
	for(int i=0;i<srcWords.size();i++){
		string sSrcWord = srcWords[i];
		iExistingTrans = jd.SrcWordDictEntries(sSrcWord).size();
		for(int j=0;j<iNoRandTrans;){
			int randWordIdx = (int) ( (double) randomMT() * (double) iTotalNoTgtWords / (double) (4294967296.0 + 1.0) );
			if(! jd.TryGetValue(sSrcWord, tgtWords[randWordIdx], dVal)){
				jd.ForceAddEntry(sSrcWord, tgtWords[randWordIdx], 0.1);
				j++;
			}
		}
	}
	/*
	for(int i=0;i<tgtWords.size();i++){
		string sTgtWord = tgtWords[i];
		iExistingTrans = jd.TgtWordDictEntries(sTgtWord).size();
		for(int j=0;j<iNoRandTrans;){
			int randWordIdx = (int) ( (double) randomMT() * (double) iTotalNoSrcWords / (double) (4294967296.0 + 1.0) );
			if(! jd.TryGetValue(srcWords[randWordIdx], sTgtWord, dVal)){
				jd.ForceAddEntry(srcWords[randWordIdx], sTgtWord, 0.1);
				j++;
			}
		}
	}
	*/
}

Pair<int,int> *SynonymPairs(MonoDictionary &jd, vector<string> &Words){
    // For fast access of wordId given word, transform Words into a map
    map<string,int> mWordId;
    for(int i=0;i<Words.size();i++)
		mWordId[Words[i]]=i;
    if(mWordId.size() != Words.size()){
        cerr << "The words vector size " << Words.size() << " didn't match with map size " << mWordId.size() << endl;
        exit(1234);
    }

    bool bPrintWarning = true;
    Pair<int,int> *transPairs = (Pair<int,int> *)calloc(jd.NoEntries(), sizeof(Pair<int,int>));
    for(int i=0;i<jd.NoEntries();i++){
        transPairs[i].First = -1;
        transPairs[i].Second = -1;
    }

    string sOtherWord;
    int iOtherWordId, iDictIdx, iDictEntry;
    Tripple<string,string,double> transEntry;

	for(int i=0;i<jd.NoEntries();i++){
		transEntry = jd.getEntry(i);
		if(mWordId.find(transEntry.First) == mWordId.end()){
			if(! ArtificialWord(transEntry.First) && bPrintWarning){
				cerr << "Dictionary has a new tgt word "<< transEntry.First <<", probably didn't u load with restricted vocab" << endl;
				bPrintWarning = false; // Don't print it again
			}
		}else
			transPairs[i].First = mWordId[transEntry.First];
		if(mWordId.find(transEntry.Second) == mWordId.end()){
			if(! ArtificialWord(transEntry.Second) && bPrintWarning){
				cerr << "Dictionary has a new tgt word "<< transEntry.Second <<", probably didn't u load with restricted vocab" << endl;
				bPrintWarning = false; // Don't print it again
			}
		}else
			transPairs[i].Second = mWordId[transEntry.Second];
	}
    return transPairs;
}

Pair<int,int> *TranslationPairs(JointDictionary &jd, vector<string> &Words,bool *bSrcLang){
    // For fast access of wordId given word, transform Words into a map
    map<string,int> mWordId;
    string sTemp;
    for(int i=0;i<Words.size();i++)
        if(bSrcLang[i]){
            sTemp = Words[i]+"_SRC";
            mWordId[sTemp]=i;
        }else{
            sTemp = Words[i]+"_TGT";
            mWordId[sTemp]=i;
        }
    if(mWordId.size() != Words.size()){
        cerr << "The words vector size " << Words.size() << " didn't match with map size " << mWordId.size() << endl;
        exit(1234);
    }

    bool bPrintWarning = true;
    Pair<int,int> *transPairs = (Pair<int,int> *)calloc(jd.NoEntries(), sizeof(Pair<int,int>));
    for(int i=0;i<jd.NoEntries();i++){
        transPairs[i].First = -1;
        transPairs[i].Second = -1;
    }

    string sOtherWord;
    int iOtherWordId, iDictIdx, iDictEntry;
    Tripple<string,string,double> transEntry;
    vector<int> wordDictEntries;

    for(int i=0;i<Words.size();i++){
        if(bSrcLang[i]){
            wordDictEntries = jd.SrcWordDictEntries(Words[i]);
            for(iDictIdx=0;iDictIdx<wordDictEntries.size();iDictIdx++){
                iDictEntry = wordDictEntries[iDictIdx];
                transEntry = jd.getEntry(iDictEntry);
                sOtherWord = transEntry.Second;
                if(mWordId.find(sOtherWord+"_TGT") == mWordId.end()){
						if(! ArtificialTgtWord(sOtherWord) && bPrintWarning){
                            cerr << "Dictionary has a new tgt word "<< sOtherWord <<", probably didn't u load with restricted vocab" << endl;
                            bPrintWarning = false; // Don't print it again
                        }
						transPairs[iDictEntry].First = i;
                        continue;
                }
                sOtherWord += "_TGT";
                iOtherWordId = mWordId[sOtherWord];
                if(bSrcLang[iOtherWordId] == true)
                    cerr << "The word " << sOtherWord << " should be a tgt word translation of word["<<i<<"]="<<Words[i]<<endl;
                if(transPairs[iDictEntry].Second != -1 && transPairs[iDictEntry].Second != iOtherWordId)
                    cerr << "Conflict of tgt words "<< Words[transPairs[iDictEntry].Second] << " & " << sOtherWord << " for the same dictionary entry " << iDictEntry << endl;
                transPairs[iDictEntry].First = i;
                transPairs[iDictEntry].Second = iOtherWordId;
            }
        }else{
            wordDictEntries = jd.TgtWordDictEntries(Words[i]);
            for(iDictIdx=0;iDictIdx<wordDictEntries.size();iDictIdx++){
                iDictEntry = wordDictEntries[iDictIdx];
                transEntry = jd.getEntry(iDictEntry);
                sOtherWord = transEntry.First;
                if(mWordId.find(sOtherWord+"_SRC") == mWordId.end()){
                    if(! ArtificialSrcWord(sOtherWord) && bPrintWarning){
                        cerr << "Dictionary has a new src word" << sOtherWord << ", probably didn't u load with restricted vocab" << endl;
                        bPrintWarning = false; // Don't print it again
                    }
					transPairs[iDictEntry].Second = i;
                    continue;
                }
                sOtherWord += "_SRC";
                iOtherWordId = mWordId[sOtherWord];
                if(bSrcLang[iOtherWordId] == false)
                    cerr << "The word " << sOtherWord << " should be a src word translation of word["<<i<<"]="<<Words[i]<<endl;
                if(transPairs[iDictEntry].First != -1 && transPairs[iDictEntry].First != iOtherWordId)
                    cerr << "Conflict of src words "<< Words[transPairs[iDictEntry].First] << " & " << sOtherWord << " for the same dictionary entry " << iDictEntry << endl;
                transPairs[iDictEntry].First = iOtherWordId;
                transPairs[iDictEntry].Second = i;
            }
        }
    }
    return transPairs;
}

void LoadMatchings(string sFilePath, MonoDictionary &md){
	
	int iAdded = 0;
	ifstream fin(sFilePath.c_str());
	istringstream istream;
	if(fin.fail())
            cerr << "Unable to open file " << sFilePath << " to load matchings" << endl;    
        else{
            while(! fin.eof()){
                string s;
                getline(fin,s);

                if(s.empty())
                    continue;

				char *buf1 = new char[s.length()], *buf2 = new char[s.length()];
				sscanf(s.c_str() ,"%s %s",buf1, buf2);

				if(md.AddEntry(string(buf1), string(buf2), 0.5))
					iAdded++;
            }
        }
	printf("Added %d matching entries\n", iAdded);
	fin.close();
}

void LoadMatchings(string sFilePath, JointDictionary &jd){
	
	int iAdded = 0;
	ifstream fin(sFilePath.c_str());
	istringstream istream;
	if(fin.fail())
            cerr << "Unable to open file " << sFilePath << " to load indices" << endl;    
        else{
            while(! fin.eof()){
                string s;
                getline(fin,s);

                if(s.empty())
                    continue;

				string sFir, sSec;
				istream.str(s);
				istream >> sFir >> sSec ;

				jd.ForceAddEntry(sFir, sSec, 0.1);
				iAdded++;
            }
        }
	printf("Added %d matching entries\n", iAdded);
	fin.close();
}

void SaveWordTranslations(ofstream &out, vector<vector<int> > vTrans){
	int iNoWords = vTrans.size();
	out.write((const char *)&iNoWords, sizeof(int));
	for(int iWordId=0;iWordId<iNoWords;iWordId++){
		int iNoTransForThisWord = vTrans[iWordId].size();
		out.write((const char *)&iNoTransForThisWord, sizeof(int));
		for(int iTrans=0;iTrans<iNoTransForThisWord;iTrans++)
			out.write((const char *)&vTrans[iWordId][iTrans], sizeof(int));
	}
}

vector<vector<int> > LoadWordTranslations(ifstream &in){
	vector<vector<int> > vTrans;
	int iNoWords = 0, iNoTransForThisWord = 0, iTemp = 0;
	in.read((char *)&iNoWords, sizeof(int));
	for(int iWordId=0;iWordId<iNoWords;iWordId++){
		vTrans.push_back(vector<int>());
		iNoTransForThisWord = 0;
		in.read((char *)&iNoTransForThisWord, sizeof(int));
		for(int iTrans=0;iTrans<iNoTransForThisWord;iTrans++){
			iTemp = 0;
			in.read((char *)&iTemp, sizeof(int));
			vTrans[iWordId].push_back(iTemp);
		}
		if(vTrans[iWordId].size() != iNoTransForThisWord){
			fprintf(stderr, "Number of read translations (%d) is not same as the number of translations saved (%d)\n", vTrans[iWordId].size(), iNoTransForThisWord); 
			exit(1234);
		}
	}
	if(vTrans.size() != iNoWords){
		fprintf(stderr, "Number of read words (%d) is not same as the number of words for which the translations are saved (%d)\n", vTrans.size(), iNoWords);
		exit(1234);
	}
	return vTrans;
}

void SavePairInt(ofstream &out, Pair<int, int> *vpArr, int iSize){

	for(int i=0;i<iSize;i++){
		out.write((const char *)&vpArr[i].First, sizeof(int));
		out.write((const char *)&vpArr[i].Second, sizeof(int));
	}
}

Pair<int, int>* LoadPairInt(ifstream &in, int iSize){
	Pair<int,int> *pArr = (Pair<int,int> *)calloc(iSize, sizeof(Pair<int,int>));
	for(int i=0;i<iSize;i++){
		in.read((char *)&pArr[i].First, sizeof(int));
		in.read((char *)&pArr[i].Second, sizeof(int));
	}
	return pArr;
}

void SavePairInt(ofstream &out, vector<Pair<int, int> > vpArr){
	int iNoEntries = vpArr.size(), iSrc, iTgt;
	out.write((const char *)&iNoEntries, sizeof(int));
	for(int i=0;i<iNoEntries;i++){
		iSrc = vpArr[i].First;
		iTgt = vpArr[i].Second;
		out.write((const char *)&iSrc, sizeof(int));
		out.write((const char *)&iTgt, sizeof(int));
	}
}

vector<Pair<int, int> > LoadPairInt(ifstream &in){
	int iNoEntries = 0, iSrc, iTgt;
	bool bSrc, bTgt;
	in.read((char *)&iNoEntries, sizeof(int));
	vector<Pair<int, int> > vpArr;
	for(int i=0;i<iNoEntries;i++){
		in.read((char *)&iSrc, sizeof(int));
		in.read((char *)&iTgt, sizeof(int));
		vpArr.push_back(Pair<int,int>(iSrc, iTgt));
	}
	if(iNoEntries != vpArr.size()){
		fprintf(stderr, "Number of entries saved and scanned (for entryHasWordsInfo) didn't match\n");
		exit(1234);
	}
	return vpArr;
}

void SaveEntryHasWordsInfo(ofstream &out, vector<Pair<bool, bool> > entryHasWords){
	int iNoEntries = entryHasWords.size(), iSrc, iTgt;
	out.write((const char *)&iNoEntries, sizeof(int));
	for(int i=0;i<iNoEntries;i++){
		iSrc = (entryHasWords[i].First ? 1 : 0);
		iTgt = (entryHasWords[i].Second ? 1 : 0);
		out.write((const char *)&iSrc, sizeof(int));
		out.write((const char *)&iTgt, sizeof(int));
	}
}

vector<Pair<bool, bool> > LoadEntryHasWordsInfo(ifstream &in){
	int iNoEntries = 0, iSrc, iTgt;
	bool bSrc, bTgt;
	in.read((char *)&iNoEntries, sizeof(int));
	vector<Pair<bool, bool> > entryHasWords;
	for(int i=0;i<iNoEntries;i++){
		in.read((char *)&iSrc, sizeof(int));
		if(iSrc != 0 && iSrc != 1){
			fprintf(stderr, "Boolean value must have been saved either as 1 or 0 and not %d\n", iSrc);
			exit(1234);
		}
		in.read((char *)&iTgt, sizeof(int));
		if(iTgt != 0 && iTgt != 1){
			fprintf(stderr, "Boolean value must have been saved either as 1 or 0 and not %d\n", iTgt);
			exit(1234);
		}
		bSrc = (iSrc == 1 ? true : false);
		bTgt = (iTgt == 1 ? true : false);
		entryHasWords.push_back(Pair<bool, bool>(bSrc, bTgt));
	}
	if(iNoEntries != entryHasWords.size()){
		fprintf(stderr, "Number of entries saved and scanned (for entryHasWordsInfo) didn't match\n");
		exit(1234);
	}
	return entryHasWords;
}

void SaveIntArr(ofstream &out, int *Arr, int iSize){
	for(int i=0;i<iSize;i++)
		out.write((const char *)&Arr[i], sizeof(int));
}

int *LoadIntArr(ifstream &in, int iSize){
	int *Arr = (int *)calloc(iSize, sizeof(int));
	for(int i=0;i<iSize;i++)
		in.read((char *)&Arr[i], sizeof(int));
	return Arr;
}

void SaveBoolArr(ofstream &out, bool *bArr, int iSize){
	int iTemp;
	for(int i=0;i<iSize;i++){
		iTemp = (bArr[i] ? 1 : 0);
		out.write((const char *)&iTemp, sizeof(int));
	}
}

bool *LoadBoolArr(ifstream &in, int iSize){
	bool *bArr = (bool *)calloc(iSize, sizeof(bool));
	int iTemp;
	for(int i=0;i<iSize;i++){
		in.read((char *)&iTemp, sizeof(int));
		if(iTemp != 0 && iTemp != 1){
			fprintf(stderr, "Boolean value must have been saved either as 1 or 0 and not %d\n", iTemp);
			exit(1234);
		}
		bArr[i] = (iTemp == 1 ? true : false);
	}
	return bArr;
}

map<string, vector<int> > LoadSeedTopicalWords(string sFilePath, int T=-1){
	// if T != -1, then load only the first T topics and discard the rest
	string sword;
	ifstream fin(sFilePath.c_str());

	map<string, vector<int> > seedWordTopics;
	if(fin.fail())
		cerr << "Unable to open file " << sFilePath << " to load indices" << endl;
	else{
		int t=0;
		while(! fin.eof()){
			string s;
			getline(fin,s);

			if(s.empty())
				continue;

			int from=0;
			for(int i=0;i<s.length();i++){
				if(s[i] == ','){
					sword = s.substr(from, i-from);
					if(seedWordTopics.find(sword) == seedWordTopics.end()){
						seedWordTopics[sword] = vector<int>();
						seedWordTopics[sword].push_back(t);
					}else if(seedWordTopics[sword][seedWordTopics[sword].size()-1] != t)	// Make sure that you are not pusing the same topic again -- redundancy check
						seedWordTopics[sword].push_back(t);
					from = i+1;
				}
			}
			if(from < s.length()){
				sword = s.substr(from, s.length()-from);
				if(seedWordTopics.find(sword) == seedWordTopics.end()){
					seedWordTopics[sword] = vector<int>();
					seedWordTopics[sword].push_back(t);
				}else if(seedWordTopics[sword][seedWordTopics[sword].size()-1] != t)
					seedWordTopics[sword].push_back(t);
			}
			t+=1;
			if(T > 0 && t >= T)
				break;
		}
	}
	/*
	for(map<string, vector<int> >::iterator iter=seedWordTopics.begin();iter!=seedWordTopics.end();iter++){
		printf("%s ", iter->first.c_str());
		for(int i=0;i<iter->second.size();i++)
			printf("%d ", iter->second[i]);
		printf("\n");
	}
	*/
	return seedWordTopics;
}

vector<int> LoadReferenceClusterLabels(string sFilePath){
	vector<int> labels;
	ifstream fin(sFilePath.c_str());
	int label;
	while(! fin.eof()){
		string s;
		getline(fin,s);
		if(s.empty())
			continue;
		label = atoi(s.c_str());
		labels.push_back(label);
	}
	printf("\n");
	fin.close();

	return labels;
}
