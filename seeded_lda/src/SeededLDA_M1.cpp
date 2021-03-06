#include <stdio.h>
#include <math.h>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <map>

#include "ext/cokus.cpp"

#include "CLLDAConfig.cpp"
#include "IO_CLLDA.cpp"
#include "utils/utils.cpp"
#include "stat/GammaFunc.cpp"

using namespace std;

int iVerbose = 1;

class GuidedLDA{

	bool *bSrcLang;
	vector<string> Words;

	int *z , *order, *ztot, *wp, *dp, *docLength;
	int *w, *d, *wTest, *dTest;
	int W, D, T, iNoTokens, iNoTestTokens;
	double dAlpha , dBeta, dKAlpha, dWBeta;
	double *dUnigramProbs;
	int *iUnigramFreq;

	SeededLDAConfig ldaConfig;

	vector<vector<int> > seedTopicWords;
	int *wpSeed, *ztotSeed, *bSeedTot;
	bool *bSeed;
	double dMu, dTau, dWMu, d2Tau;

	void Init(){
		bSrcLang = NULL;
		z = NULL; order=NULL; ztot=NULL; wp=NULL; dp=NULL; docLength = NULL;
		w = NULL; d = NULL; wTest = NULL; dTest = NULL;
		W = 0; D = 0; T = 0; iNoTokens = 0; iNoTestTokens = 0;
		SEED = 3; OUTPUT = 2; iNoIter = 0;
		dAlpha = 0; dBeta = 0; dKAlpha = 0; dWBeta = 0; 
		iUnigramFreq = NULL;


		dTemp = 0;
	}

	public: 
	double dTemp;
	int SEED, OUTPUT, iNoIter;

	GuidedLDA(){ Init(); }
	GuidedLDA(string sConfigPath) {
		Init();
		LoadData(sConfigPath);
	}

	void PreInitialize(int , char *argv[]);
	void LoadData(string sConfigPath);
	void RandomInitialize();
	void RandomOrder();
	void Iterate(bool bOnTestSet=true,int iNoSamples=0, const int *iOrder=NULL, int iNoExtraIter=0);

	double GetTopicDocProb(int iTopic, int iDocId);
	double GetWordTopicProb(int iWordId, int iTopic);
	double GetWordDocProb(int iWordId, int iDocId);

	bool CheckTopicDistribution(int iTopic);
	bool CheckDocDistribution(int iDocId);
	bool CheckConsistency();
	
	double Perplexity(int iNoTokens,const int *w,const int *d);
	void PauseToEvaluate(int iter, bool bOnTestSet=true, int iNoSamples=0, const int *iOrder=NULL);
	void UpdateHyperParams();

	void PrintTopicalWords(int aNoWords=10);
	void PrintTextTopics();
	void TopicStats();
	void PrintDocTopicDist(string sFilePath);

	void LoadFoldInData();
	void FoldIn();
	bool ConsistencyFoldInDocs();
	void PauseWhileFolding(int iter);
	void PrintDocTopicDistFoldIn();
	void EntropyOfDistributions();

	void SaveObject(string sFilePath);
	void LoadObject(string sFilePath);
	void SaveTopicAssignments(string sFilePath,int idx=-1);
	void LoadTopicAssignments(string sFilePath);
	void InitializeFrmPrevCounts();
	void PrintBestTopic(string sFilePath);
	void EvaluateWordAssociation(string sWordAssocPath);

	int LeftOutTokens();
};

void GuidedLDA::SaveTopicAssignments(string sFilePath, int idx){
	fprintf(stderr, "Saving to %s\n", sFilePath.c_str());
	if(sFilePath.empty()){
		fprintf(stderr, "Input non-empty filepath to save object\n");
		exit(1234);
	}

	if(idx != -1){
		std::stringstream ss;
		ss << sFilePath << "." << idx;
		sFilePath = ss.str();
	}

	ofstream meta(sFilePath.c_str(), ios::binary);
	meta.write((const char *)&W, sizeof(int));
	meta.write((const char *)&D, sizeof(int));
	meta.write((const char *)&T, sizeof(int));
	meta.write((const char *)&iNoTokens, sizeof(int));

	SaveIntArr(meta, z, iNoTokens);
	SaveBoolArr(meta, bSeed, iNoTokens);
	meta.close();
	fprintf(stderr, "Saving to %s done\n", sFilePath.c_str());
}

void GuidedLDA::LoadTopicAssignments(string sFilePath){
	fprintf(stderr, "Loading from %s\n", sFilePath.c_str());
	if(sFilePath.empty()){
		fprintf(stderr, "The filepath is empty\n");
		exit(1234);
	}

	ifstream meta(sFilePath.c_str(), ios::binary);
	if(meta.fail()){
		fprintf(stderr, "Failed to open file %s\n", sFilePath.c_str());
		exit(1234);
	}
	meta.read((char *)&W, sizeof(int));
	meta.read((char *)&D, sizeof(int));
	meta.read((char *)&T, sizeof(int));
	fprintf(stderr, "Retrieving W=%d D=%d T=%d\n", W, D, T);
	int iTemp;
	meta.read((char *)&iTemp, sizeof(int));
	if(iTemp != iNoTokens){
		fprintf(stderr, "No tokens from the config file (%d) didn't match saved tokens (%d)\n", iNoTokens, iTemp);
		exit(1234);
	}

	z = LoadIntArr(meta, iNoTokens);
	bSeed = LoadBoolArr(meta, iNoTokens);
	meta.close();
	fprintf(stderr, "Loading from %s done\n", sFilePath.c_str());
}

void GuidedLDA::InitializeFrmPrevCounts(){
	int i, topic, wi, di, widx, iAllowTill, wioffset, idxoffset;
	bool bFrmSeedTopic;
	for (i=0; i<iNoTokens; i++)
	{
		wi = w[ i ];
		di = d[ i ];

		topic = z[i];
		bFrmSeedTopic = bSeed[i];
		if(bFrmSeedTopic){
			ztotSeed[topic]++;
			wpSeed[wi*T+topic]++;
			bSeedTot[topic*2+1]++;
		}else{
			ztot[topic]++;
			wp[wi*T+topic]++;
			if(seedTopicWords[wi].size() > 0)
				bSeedTot[topic*2]++;
		}
		dp[di*T+topic]++;
	}
	if( CheckConsistency() == false){
		fprintf(stderr, "Initial topic assignments are not consistent\n");
		exit(1234);
	}
}

/*
void GuidedLDA::SaveObject(string sFilePath){

	fprintf(stderr, "THIS FUNCTION PROBABLY NEEDS TO BE UPDATED\n"); exit(1234);
	fprintf(stderr, "Saving to %s\n", sFilePath.c_str());
	if(sFilePath.empty()){
		fprintf(stderr, "Input non-empty filepath to save object\n");
		exit(1234);
	}

	ofstream meta(sFilePath.c_str(), ios::binary);
	meta.write((const char *)&W, sizeof(int));
	meta.write((const char *)&D, sizeof(int));
	meta.write((const char *)&T, sizeof(int));
	meta.write((const char *)&dAlpha, sizeof(double));
	meta.write((const char *)&dBeta, sizeof(double));

	SaveIntArr(meta, wp, T*W);
	SaveIntArr(meta, ztot, T);
	SaveIntArr(meta, dp, T*D);
	SaveIntArr(meta, docLength, D);
	meta.close();
	fprintf(stderr, "Saving to %s done\n", sFilePath.c_str());
}

void GuidedLDA::LoadObject(string sFilePath){
	fprintf(stderr, "THIS FUNCTION PROBABLY NEEDS TO BE UPDATED\n"); exit(1234);
	fprintf(stderr, "Loading from %s\n", sFilePath.c_str());
	if(sFilePath.empty()){
		fprintf(stderr, "The filepath is empty\n");
		exit(1234);
	}

	ifstream meta(sFilePath.c_str(), ios::binary);
	if(meta.fail()){
		fprintf(stderr, "Failed to open file %s\n", sFilePath.c_str());
		exit(1234);
	}
	meta.read((char *)&W, sizeof(int));
	meta.read((char *)&D, sizeof(int));
	meta.read((char *)&T, sizeof(int));
	meta.read((char *)&dAlpha, sizeof(double));
	meta.read((char *)&dBeta, sizeof(double));

	dKAlpha = (double) T*dAlpha;
	dWBeta = (double) W*dBeta;
	printf("W:%d D:%d T:%d\n", W, D, T);
	printf("Alpha:%e Beta:%e dAlpha:%e dWBeta:%e\n", dAlpha, dBeta, dKAlpha, dWBeta);

	wp = LoadIntArr(meta, T*W);
	ztot = LoadIntArr(meta, T);
	dp = LoadIntArr(meta, T*D);
	docLength = LoadIntArr(meta, D);
	meta.close();
	fprintf(stderr, "Loading from %s done\n", sFilePath.c_str());

}
*/
void GuidedLDA::PreInitialize(int argc, char *argv[]){
	// Preinitilize certain parameters and doesn't allow them to be overwritten
	// May be useful for repeated experiments
	if(argc > 3){
		//T = atoi(argv[2]);
		SEED = atoi(argv[3]);
		printf("Choose %d as seed\n", SEED);
	}
}

void GuidedLDA::LoadData(string sConfigPath){
	cout << "Reading from " << sConfigPath << endl;
	ldaConfig.LoadConfig(sConfigPath);
	ldaConfig.PrintConfig();

	T = ldaConfig.m_iNoTopics;
	iNoIter = ldaConfig.m_iNoIterations;
	dAlpha = ldaConfig.m_dAlpha;
	dBeta = ldaConfig.m_dBeta;

	int iNoSrcWordIndices=0,iNoTgtWordIndices=0,iNoSrcDocIndices=0,iNoTgtDocIndices=0,iNoSrcWords=0,iNoTgtWords=0;
	int *SrcWordIndices = NULL, *SrcDocIndices = NULL, *TgtWordIndices = NULL, *TgtDocIndices = NULL;
	vector<string> SrcWords, TgtWords;
	if(dTemp != 0){
		fprintf(stderr, "****** Using the preinitialized alpha %lf instead of %lf\n", dTemp, dBeta);
		dBeta = dTemp;
	}

	SrcWordIndices = LoadIndices(ldaConfig.m_sSrcWordIndicesPath,&iNoSrcWordIndices);
	SrcDocIndices = LoadIndices(ldaConfig.m_sSrcDocIndicesPath,&iNoSrcDocIndices);
	SrcWords = LoadWords(ldaConfig.m_sSrcWordsPath);
	if(iNoSrcWordIndices != iNoSrcDocIndices)
		cerr << "Mismatch of Src indices " << endl;
	iNoSrcWords = SrcWords.size();

	if(! ldaConfig.m_sTgtWordIndicesPath.empty())
		TgtWordIndices = LoadIndices(ldaConfig.m_sTgtWordIndicesPath,&iNoTgtWordIndices);
	if(! ldaConfig.m_sTgtDocIndicesPath.empty())
		TgtDocIndices = LoadIndices(ldaConfig.m_sTgtDocIndicesPath,&iNoTgtDocIndices);
	if(! ldaConfig.m_sTgtWordsPath.empty())
		TgtWords = LoadWords(ldaConfig.m_sTgtWordsPath);
	if(iNoTgtWordIndices != iNoTgtDocIndices)
		cerr << "Mismatch of Tgt indices" << endl;
	iNoTgtWords = TgtWords.size();

	int *SrcWordTestIndices = NULL, *TgtWordTestIndices = NULL, *SrcDocTestIndices = NULL, *TgtDocTestIndices = NULL;
    int iNoSrcWordTestTokens=0,iNoTgtWordTestTokens=0,iNoSrcDocTestTokens=0,iNoTgtDocTestTokens=0,iNoTestWordTokens=0,iNoTestDocTokens=0;
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

    int iTgtWordOffset = MaxIdx(SrcWordIndices, iNoSrcWordIndices, SrcWordTestIndices, iNoSrcWordTestTokens);
    int iTgtDocOffset = MaxIdx(SrcDocIndices, iNoSrcDocIndices, SrcDocTestIndices, iNoSrcDocTestTokens);
	fprintf(stderr, "iTgtWordOffset:%d & iTgtDocOffset:%d\n", iTgtWordOffset, iTgtDocOffset);

    int iNoWordTokens = 0, iNoDocTokens = 0 , iNoWords = 0;
    w = ExtendIndices(SrcWordIndices,iNoSrcWordIndices,TgtWordIndices,iNoTgtWordIndices,iTgtWordOffset, &iNoWordTokens);
    d = ExtendIndices(SrcDocIndices,iNoSrcDocIndices,TgtDocIndices,iNoTgtDocIndices,iTgtDocOffset, &iNoDocTokens);
    wTest = ExtendIndices(SrcWordTestIndices, iNoSrcWordTestTokens, TgtWordTestIndices, iNoTgtWordTestTokens, iTgtWordOffset, &iNoTestWordTokens);
    dTest = ExtendIndices(SrcDocTestIndices, iNoSrcDocTestTokens, TgtDocTestIndices, iNoTgtDocTestTokens, iTgtDocOffset, &iNoTestDocTokens);
    if(iNoWordTokens != iNoDocTokens)
        cerr << "No. of Word tokens didn't match with No. of Doc tokens" << endl;
	iNoTokens = iNoWordTokens;
    if(iNoTestWordTokens != iNoTestDocTokens){
        fprintf(stderr, "No test word tokens didn't match with no test doc tokens\n");
        exit(1234);
    }
	iNoTestTokens = iNoTestWordTokens;

	bool *bSrcLang = new bool[SrcWords.size() + TgtWords.size()];
	int idx=0;
	for(int i=0;i<SrcWords.size();i++)
		bSrcLang[idx++] = true;
	for(int i=0;i<TgtWords.size();i++)
		bSrcLang[idx++] = false;

	Words = ExtendWords(SrcWords,TgtWords);

	if(SrcWordIndices != NULL)
		free(SrcWordIndices);
	if(SrcDocIndices != NULL)
		free(SrcDocIndices);
	if(TgtWordIndices != NULL)
		free(TgtWordIndices);
	if(TgtDocIndices != NULL)
		free(TgtDocIndices);
	if(SrcWordTestIndices != NULL)
		free(SrcWordTestIndices);
	if(SrcDocTestIndices != NULL)
		free(SrcDocTestIndices);
	if(TgtWordTestIndices != NULL)
		free(TgtWordTestIndices);
	if(TgtDocTestIndices != NULL)
		free(TgtDocTestIndices);

	// seeding
	seedMT( 1 + SEED * 2 ); // seeding only works on uneven numbers

	/* allocate memory */
	z  = (int *) calloc( iNoTokens , sizeof( int ));
	order  = (int *) calloc( iNoTokens , sizeof( int ));  
	ztot  = (int *) calloc( T, sizeof( int ));

	// copy over the word and document indices into internal format
	for (int i=0; i<iNoWordTokens; i++) {
		w[ i ] = (int) w[ i ] - 1;
		d[ i ] = (int) d[ i ] - 1;
	}
	for(int i=0;i<iNoTestWordTokens;i++){
        wTest[i] = wTest[i]-1;
        dTest[i] = dTest[i]-1;
    }

	W = 0;
	D = 0;
	for (int i=0; i<iNoWordTokens; i++) {
		if (w[ i ] > W) W = w[ i ];
		if (d[ i ] > D) D = d[ i ];
	}
	for(int i=0;i<iNoTestWordTokens;i++){
        if(wTest[i] > W) W = wTest[i];
        if(dTest[i] > D) D = dTest[i];
    }
	W = W + 1;
	D = D + 1;

	if(W != Words.size()){
		fprintf(stderr, "Mismatch in the number of words from indices (%d) and actual words (%d)\n", W, Words.size());
		fprintf(stderr, "Be aware of this error\n");
		W = Words.size();
	}
	iUnigramFreq = UnigramFreqs(w, iNoTokens, W);
	dUnigramProbs = UnigramProbs(w, iNoTokens, W);
	docLength = DocLengths(d,D,iNoTokens);
	dWBeta = (double) (W*dBeta);
	dKAlpha = (double) (T*dAlpha);

	wp  = (int *) calloc( T*W , sizeof( int ));
	dp  = (int *) calloc( T*D , sizeof( int ));

	map<string,vector<int> > wordPriorTopics = LoadSeedTopicalWords(ldaConfig.sSeedTopicalWordsPath);
	for(int i=0;i<Words.size();i++){
		if(wordPriorTopics.find(Words[i]) != wordPriorTopics.end()){
			seedTopicWords.push_back(wordPriorTopics[Words[i]]);
			if(seedTopicWords[1].size() > 1){
				printf("Not unique seed %s ", Words[i].c_str());
				for(int j=0;j<seedTopicWords[i].size();j++)
					printf("%d ", seedTopicWords[i][j]);
				printf("\n");
			}
		}else{
			seedTopicWords.push_back(vector<int>());
		}
	}
	wpSeed = (int *) calloc(T*W, sizeof(int));
	ztotSeed = (int *)calloc(T, sizeof(int));
	bSeed = (bool *)calloc(iNoTokens, sizeof(bool));
	bSeedTot = (int *) calloc(2*T, sizeof(int));
	dMu = ldaConfig.dMu;
	dWMu = W*dMu;
	dTau = ldaConfig.dTau;
	d2Tau = 2*dTau;
	if (OUTPUT==2) {
		printf( "Running LDA Gibbs Sampler Version 1.0\n" );
		printf( "Arguments:\n" );
		printf( "\tNumber of words      W = %d\n"    , W );
		printf( "\tNumber of docs       D = %d\n"    , D );
		printf( "\tNumber of topics     T = %d\n"    , T);
		printf( "\tNumber of iterations N = %d\n"    , iNoIter );
		printf( "\tHyperparameter   ALPHA = %4.4f\n" , dAlpha );
		printf( "\tHyperparameter    BETA = %4.4f\n" , dBeta );
		printf( "\tSeed number            = %d\n"    , SEED );
		printf( "\tNumber of tokens       = %d\n"    , iNoTokens);
		//printf( "Checking: sizeof(int)=%d sizeof(long)=%d sizeof(double)=%d\n" , sizeof(int) , sizeof(long) , sizeof(double));
	}
}

void GuidedLDA::RandomInitialize(){
	int i, topic, wi, di, widx, itidx;
	double dRand;
	for (i=0; i<iNoTokens; i++)
	{
		wi = w[ i ];
		di = d[ i ];
		bool bFrmSeedTopic = false;
		if(seedTopicWords[wi].size() > 0){
			dRand = ( (double) randomMT() / (double) (4294967296.0 + 1.0) );
			if( dRand  < dTau)
				bFrmSeedTopic = true;
		}

		if(bFrmSeedTopic == false){
			// pick a random topic 0..T-1
			topic = (int) ( (double) randomMT() * (double) T / (double) (4294967296.0 + 1.0) );
			ztot[ topic ]++; // increment ztot matrix
			wp[ wi*T + topic ]++; // increment wp count matrix
			bSeed[i] = bFrmSeedTopic;
			if(seedTopicWords[wi].size() > 0)
				bSeedTot[topic*2]++;
		}else{
			if(seedTopicWords[wi].size() == 1)
				topic = seedTopicWords[wi][0];
			else{
				itidx = (int) ( (double) randomMT() * (double) seedTopicWords[wi].size() / (double) (4294967296.0 + 1.0) );
				topic = seedTopicWords[wi][itidx];
			}
			ztotSeed[topic]++;
			wpSeed[wi*T+topic]++;
			bSeed[i] = bFrmSeedTopic;
			bSeedTot[topic*2+1]++;
		}
		if(topic < 0 || topic >= T){
			fprintf(stderr, "sampled incorrect topic %d\n", topic);
			exit(1234);
		}
		z[ i ] = topic; // assign this word token to this topic
		dp[ di*T + topic ]++; // increment dp count matrix
	}

	if( CheckConsistency() == false){
		fprintf(stderr, "Initial topic assignments are not consistent\n");
		exit(1234);
	}
}

void GuidedLDA::RandomOrder(){
    printf( "Determining random order update sequence\n" );
    int rp, temp;
    for (int i=0; i<iNoTokens; i++) order[i]=i; // fill with increasing series
    for (int i=0; i<(iNoTokens-1); i++) {
        // pick a random integer between i and nw
        rp = i + (int) ((double) (iNoTokens-i) * (double) randomMT() / (double) (4294967296.0 + 1.0));

        // switch contents on position i and position rp
        temp = order[rp];
        order[rp]=order[i];
        order[i]=temp;
    }
}

double GuidedLDA::GetWordTopicProb(int iWordId, int iTopic){
	//return (((double) wp[iWordId*T+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta));
	double dx, dFrac;
	int wioffset = iWordId*T;
	/*
	if(seedTopicWords[iWordId].size() > 0){
		dx = ((double)bSeedTot[2*iTopic]+dTau)/(bSeedTot[2*iTopic]+bSeedTot[2*iTopic+1]+d2Tau);
		dFrac = dx*(((double) wp[wioffset+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta));
		dFrac += (1-dx) * ((double) wpSeed[ wioffset+iTopic ] + (double) dMu)/( (double) ztotSeed[iTopic]+ (double) dWMu); 
	}else{
		dFrac = (((double) wp[wioffset+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta));
	}
	return dFrac;
	*/
	//dx = ((double)ztot[iTopic]+dTau)/(ztot[iTopic]+ztotSeed[iTopic]+d2Tau);
	dx = dTau;
	wioffset = iWordId*T;
	dFrac = dx*(((double) wp[wioffset+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta));
	dFrac += (1-dx) * ((double) wpSeed[ wioffset+iTopic ] + (double) dMu)/( (double) ztotSeed[iTopic]+ (double) dWMu); 
	return dFrac;
}

double GuidedLDA::GetTopicDocProb(int iTopic, int iDocId){
	return (((double) dp[iDocId*T+iTopic] + dAlpha)/((double)docLength[iDocId]+dKAlpha));
}

double GuidedLDA::GetWordDocProb(int iWordId, int iDocId){
    double dWordDocProb = 0;
    for(int j=0;j<T;j++){
        double dWordTopicProb = GetWordTopicProb(iWordId,j);
        double dTopicDocProb = GetTopicDocProb(j,iDocId);
        dWordDocProb += (dWordTopicProb * dTopicDocProb);
    }
    return dWordDocProb;
}

bool GuidedLDA::CheckDocDistribution(int iDocId){
    int dioffset = iDocId*T, iTemp=0;
	double dTempProb = 0;
    for(int i=dioffset;i<dioffset+T;i++){
        iTemp += dp[i];
    }
	for(int k=0;k<T;k++)
		dTempProb += GetTopicDocProb(k, iDocId);
    if(iTemp != docLength[iDocId]){
        cerr << "Tring Tring .... the total number of class assignments " << iTemp << " didn't match with doc length " << docLength[iDocId] << endl;
        return false;
    }
	if( !isfinite(dTempProb) || fabs(dTempProb-1) > 1e-8){
		fprintf(stderr, "Strange ... Though integers counted, sum p(.|d=%d) = %e didn't match\n", iDocId, dTempProb);
		return false;
	}
    return true;
}

bool GuidedLDA::CheckTopicDistribution(int iTopic){
	int iTemp = 0;
	double dTempProb = 0;
	for(int w=0;w<W;w++){
		iTemp += wp[w*T+iTopic];
		dTempProb += GetWordTopicProb(w,iTopic);
	}
	if(iTemp != ztot[iTopic]){
		fprintf(stderr, "Ooops ... No dict entries assigned to p(.|k=%d) didn't match\n", iTopic);
		return false;
	}
	/*
	dTempProb = 0;
	double dInside = 0, dOutside = 0;
	for(int iWordId=0;iWordId<W;iWordId++){
		double dx, dFrac;
		int wioffset = iWordId*T;
		if(seedTopicWords[iWordId].size() > 0){
			dx = ((double)bSeedTot[2*iTopic]+dTau)/(bSeedTot[2*iTopic]+bSeedTot[2*iTopic+1]+d2Tau);
			dFrac = dx*(((double) wp[wioffset+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta));
			dFrac += (1-dx) * ((double) wpSeed[ wioffset+iTopic ] + (double) dMu)/( (double) ztotSeed[iTopic]+ (double) dWMu); 
			dInside += dFrac;
		}else{
			dFrac = (((double) wp[wioffset+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta));
			dOutside += dFrac;
		}
		dTempProb += dFrac;
	}
	double dDocProb = 0, dSeedProb = 0;
	for(int w=0;w<W;w++){
		int wioffset = w*T;
		if(seedTopicWords[w].size() > 0){
			dSeedProb += (((double) wp[wioffset+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta));
		}else{
			dDocProb += (((double) wp[wioffset+iTopic] + dBeta)/((double)ztot[iTopic]+dWBeta)); 
		}
	}
	printf("Inside:%lf Outside:%lf delta(w,s)=0:%lf delta(w,s)=1:%lf\n", dInside, dOutside, dDocProb, dSeedProb);
	*/
	if( ! isfinite(dTempProb) || fabs(dTempProb-1) > 1e-8){
		fprintf(stderr, "Strange ... Though integers counted, sum p(.|k=%d) = %e didn't match\n", iTopic, dTempProb);
		return false;
	}
	/*
	iTemp = 0;
	for(map<int, vector<int> >::iterator iter=seedTopicWords.begin();iter != seedTopicWords.end();iter++){
		int w = iter->first;
		for(int i=0;i<seedTopicWords[w].size();i++){
			if(seedTopicWords[w][i] == iTopic){
				iTemp += wpSeed[w*T+iTopic];  
			}
		}
	}
	if( iTemp != ztotSeed[iTopic]){
		fprintf(stderr, "Ooops ... No seed words assigned to p(.|k=%d) didn't match\n", iTopic);
		return false;
	}
	*/

	if(iVerbose > 1)
		printf("Topic:%d Total seed tokens:%d (fromSeedTopic:%d fromDocTopic:%d)\n", iTopic, bSeedTot[2*iTopic]+bSeedTot[2*iTopic+1], bSeedTot[2*iTopic+1], bSeedTot[2*iTopic]);
	//printf("Topic:%d frmSeed:%d frmDoc:%d Total:%d\n", iTopic, ztotSeed[iTopic], ztot[iTopic], ztotSeed[iTopic]+ztot[iTopic]);
    return true;
}

void GuidedLDA::EntropyOfDistributions(){
	double dAvgEntropy = 0, dTmp;
	for(int di=0;di<D;di++){
		for(int k=0;k<T;k++){
			dTmp = GetTopicDocProb(k, di);
			dAvgEntropy += dTmp * log(dTmp);
		}
	}
	dAvgEntropy *= -1;
	dAvgEntropy /= (double)D;
	printf(" Entropy (k|d): %lf ", dAvgEntropy);
	dAvgEntropy = 0;
	for(int k=0;k<T;k++){
		for(int wi=0;wi<W;wi++){
			dTmp = GetWordTopicProb(wi, k);
			dAvgEntropy += dTmp * log(dTmp);
		}
	}
	dAvgEntropy *= -1;
	dAvgEntropy /= (double)T;
	printf("(w|k): %lf", dAvgEntropy);
}

bool GuidedLDA::CheckConsistency(){
    bool bReturn = true;

    for(int iDocId=0;iDocId < D; iDocId++){
        if( CheckDocDistribution(iDocId) == false)
            bReturn = false;
    }
    for(int topic=0;topic<T;topic++){
        if( CheckTopicDistribution(topic) == false)
            bReturn = false;
    }

	/*
    for(int iDocId=0;iDocId < D; iDocId++){
		double dTotProb = 0;
		for(int iWordId=0;iWordId<W;iWordId++)
			dTotProb += GetWordDocProb(iWordId, iDocId);
        if(fabs(dTotProb-1) > 1e-8){
            cerr << "Ooops ... Sum of word probabilities in doc " << iDocId << " is " << dTotProb << " != 1" << endl;
            return false;
        }
    }
	*/
    return bReturn;
}

double GuidedLDA::Perplexity(int n,const int *w,const int *d){
	int i,j,wi,di,wioffset,dioffset;
	double dWordDocProb,dprob,dPerp;
	int iSummedOver = 0;
	dPerp = 0;
	for(i=0;i<n;i++){
		wi = w[i];
		di = d[i];
		//if(dUnigramProbs[wi] > 4)
		//	continue;

		dWordDocProb = GetWordDocProb(wi, di);
		dPerp += log(dWordDocProb);
		iSummedOver+=1;
	}
	dPerp *= -1;
	dPerp /= (double)iSummedOver;
	return exp(dPerp);
}

void GuidedLDA::Iterate(bool bOnTestSet, int iNoSamples, const int *iOrder, int iNoExtraIter){
	int wi,di,i,ii,j,topic,iter, wioffset, dioffset, widx, iAllowTill;
	double totprob, r, max, dTemp;
	double *probs = (double *)calloc(2*T, sizeof(double));

	if( CheckConsistency() == false){
		fprintf(stderr, "Initial topic assignments are not consistent\n");
		exit(1234);
	}
	PauseToEvaluate(0, bOnTestSet, iNoSamples, iOrder);
	int iRunFor = (iNoExtraIter == 0 ? iNoIter : iNoExtraIter);
	bool bFrmSeedTopic;
	for (iter=1; iter<=iRunFor; iter++) {
		//dTemp = sqrt(iter);
		for (ii = 0; ii < iNoTokens; ii++) {
			i = order[ ii ]; // current word token to assess

			wi  = w[i]; // current word index
			di  = d[i]; // current document index  
			wioffset = wi*T;
			dioffset = di*T;

			bFrmSeedTopic = bSeed[i];
			topic = z[i]; // current topic assignment to word token
			dp[dioffset+topic]--;

			if(bFrmSeedTopic){
				ztotSeed[topic]--;
				wpSeed[wioffset+topic]--;
				bSeedTot[2*topic+1]--;
				if(ztotSeed[topic] < 0 || wpSeed[wioffset+topic] < 0 || bSeedTot[2*topic+1] < 0){
					fprintf(stderr, "frmSeedTopic for topic %d counts became %d, %d, %d\n", topic, ztotSeed[topic], wpSeed[wioffset+topic], bSeedTot[2*topic+1]);
					exit(1234);
				}
			}else{
				ztot[topic]--;  // substract this from counts
				wp[wioffset+topic]--;
				if(seedTopicWords[wi].size() > 0)
					bSeedTot[2*topic]--;
				if(ztot[topic] < 0 || wp[wioffset+topic] < 0 || bSeedTot[2*topic] < 0){
					fprintf(stderr, "doc counts became %d, %d, %d\n", ztot[topic], wp[wioffset+topic], bSeedTot[2*topic]);
					exit(1234);
				}
			}

			//printf( "(1) Working on ii=%d i=%d wi=%d di=%d topic=%d wp=%d dp=%d\n" , ii , i , wi , di , topic , wp[wi+topic*W] , dp[wi+topic*D] );

			totprob = (double) 0;
			if(seedTopicWords[wi].size() == 0)
			{
				// This word doesn't have any topic preference
				for (j = 0; j < T; j++) {
					probs[j] = ((double) wp[ wioffset+j ] + (double) dBeta)/( (double) ztot[j]+ (double) dWBeta)*( (double) dp[ dioffset+ j ] + (double) dAlpha);
					probs[j] *= (((double)ztot[j]+dTau)/(double)(ztot[j]+ztotSeed[j]+d2Tau));
					totprob += probs[j];
				}
				for(j=0;j<T;j++)
					probs[j+T] = 0;
			}else{
				// probabilities for xi = 0, i.e. choose from the doc
				for (j = 0; j < T; j++) {
					probs[j] = ((double) wp[ wioffset+j ] + (double) dBeta)/( (double) ztot[j]+ (double) dWBeta)*( (double) dp[ dioffset+ j ] + (double) dAlpha);
					//probs[j] *= (((double)ztot[j]+dTau)/(double)(ztot[j]+ztotSeed[j]+d2Tau));
					probs[j] *= (1-dTau);
					totprob += probs[j];
				}
				for(j=0;j<T;j++)
					probs[j+T] = 0;
				for(int jtmp=0;jtmp<seedTopicWords[wi].size();jtmp++){
					j = seedTopicWords[wi][jtmp];
					probs[j+T] = ((double) wpSeed[ wioffset+j ] + (double) dMu)/( (double) ztotSeed[j]+ (double) dWMu)*( (double) dp[ dioffset+ j ] + (double) dAlpha);
					//probs[j+T] *= (((double)ztotSeed[j]+dTau)/(double)(ztot[j]+ztotSeed[j]+d2Tau));
					probs[j+T] *= dTau;
					totprob += probs[j+T];
				}
			}

			// sample a topic from the distribution
			r = (double) totprob * (double) randomMT() / (double) 4294967296.0;
			max = probs[0];
			topic = 0;
			while (r>max) {
				topic++;
				max += probs[topic];
			}
			if(topic < 0 || topic >= 2*T){
				printf("%lf %lf\n", r, totprob);
				fprintf(stderr, "iteration %d: sampled incorrect topic %d\n", iter, topic);
				fprintf(stderr, "The word is %s is allowed in %d topics", Words[wi].c_str(), seedTopicWords[wi].size());
				exit(1234);
			}

			if(topic < T){
				bFrmSeedTopic = false;
			}else{
				bFrmSeedTopic = true;
				topic -= T;
			}
			bSeed[i] = bFrmSeedTopic;
			if(bFrmSeedTopic){
				bool bFound = false;
				for(j=0;j<seedTopicWords[wi].size();j++){
					if(seedTopicWords[wi][j] == topic){
						bFound = true;
						break;
					}
				}
				if(bFound == false){
					printf("%lf %lf\n", r, totprob);
					for(j=0;j<2*T;j++)
						printf("%lf ", probs[j]);
					printf("\n");
					fprintf(stderr, "Sampled topic %d, which is not allowed for word %s\n", topic, Words[wi].c_str());
					exit(1234);
				}
			}

			z[i] = topic; // assign current word token i to topic j
			dp[dioffset + topic ]++;
			if(bFrmSeedTopic){
				wpSeed[wioffset+topic]++;
				ztotSeed[topic]++;
				bSeedTot[2*topic+1]++;
			}else{
				wp[wioffset + topic ]++; // and update counts
				ztot[topic]++;
				if(seedTopicWords[wi].size() > 0)
					bSeedTot[2*topic]++;
			}
		}
		if ((iter % 100)==0){
			PauseToEvaluate(iter, bOnTestSet, iNoSamples, iOrder);
			//UpdateHyperParams();
			//exit(1234);
		}
	}

	if(iRunFor % 100 != 0)
		PauseToEvaluate(iter, bOnTestSet, iNoSamples, iOrder);
}

void GuidedLDA::PauseToEvaluate(int iter, bool bOnTestSet, int iNoSamples, const int *iOrder){
	double dPerp, dELLikelihood;
	if(CheckConsistency() == false){
		cerr << "Inconsistency in the probabilities " << endl;
		exit(1);
	}
	printf( "\tIter:");
	if(iter % 20 == 0)
		printf("20x:");
	if(iter % 50 == 0)
		printf("50x");
	if(iter % 100 == 0)
		printf("100x");
	printf( " %d /%d, perp..." , iter , iNoIter);
	dPerp = Perplexity(iNoTokens,w,d);
	printf(" (train) %lf",dPerp);
	if(bOnTestSet && iNoTestTokens != 0){   // If there is separate testing data
		dPerp = Perplexity(iNoTestTokens,wTest,dTest);
		printf(" (test) %lf",dPerp);

		/*
		if(iNoSamples != 0){
			double *dWordTopicLangProbs = GetWordTopicLangProbs();
			if(testWordDocFreq.size() > 0)
				dELLikelihood = EmpiricalLikelihood(dWordTopicLangProbs, testWordDocFreq, iNoSamples, dTopicDocDistributions);
			else
				dELLikelihood = EmpiricalLikelihood(dWordTopicLangProbs, iNoTestTokens, wTest, dTest, iNoSamples, dTopicDocDistributions);
			printf(" EL(m=%d) %lf", iNoSamples, dELLikelihood);
			if(dWordTopicLangProbs != NULL);
			free(dWordTopicLangProbs);
		}
		*/
	}
	//printf(" beta:%e", dBeta);
	EntropyOfDistributions();
	printf("\n");
}
/*
void GuidedLDA::UpdateHyperParams(){
	fprintf(stderr, "THIS FUNCTION PROBABLY NEEDS TO BE UPDATED\n"); exit(1234);
	// Update alpha
	double dNum, dDenom, dTotAlpha, dTotBeta;
	double dDGAlpha = D*GammaFunc::digamma(dAlpha), dDGKAlpha = D*GammaFunc::digamma(dKAlpha);
	dDenom = 0;
	for(int di=0;di<D;di++)
		dDenom += GammaFunc::digamma(docLength[di]+dKAlpha);
	dDenom -= dDGKAlpha;
	
	dTotAlpha = 0;
	for(int k=0;k<T;k++){
		dNum = 0;
		for(int di=0;di<D;di++)
			dNum += GammaFunc::digamma(dp[di*T+k] + dAlpha);
		dNum -= dDGAlpha;
		//dAlpha_k = dNum/dDenom;
		dTotAlpha += dNum;
	}
	dAlpha *= dTotAlpha / ((double)T*dDenom);	// Average of the dAlpha_k vec
	assert(dAlpha >= 0);
	dKAlpha = T*dAlpha;

	// Beta Update
	double dDGBeta = T*GammaFunc::digamma(dBeta), dDGWBeta = T*GammaFunc::digamma(dWBeta);
	dDenom = 0;
	for(int k=0;k<T;k++){
		dDenom += GammaFunc::digamma(ztot[k]+dWBeta);
	}
	dDenom -= dDGWBeta;

	dTotBeta = 0;
	for(int wi=0;wi<W;wi++){
		int wioffset = wi*T;
		dNum = 0;
		for(int k=0;k<T;k++)
			dNum += GammaFunc::digamma(wp[wioffset+k] + dBeta);
		dNum -= dDGBeta;
		dTotBeta += dNum;
	}
	dBeta *= dTotBeta / ((double)W*dDenom);
	assert(dBeta > 0);
	dWBeta = (double)W*dBeta;

	printf("Alpha:%lf Beta:%lf\n", dAlpha, dBeta);
}
*/

void GuidedLDA::PrintTopicalWords(int aNoWords){
	double dTotalProb=0, dTempProb=0;
	Pair<int,double>::Compare pDecreasingOrder(true);
	priority_queue<Pair<int,double>, vector<Pair<int,double> >, Pair<int,double>::Compare> pqTopicalWords(pDecreasingOrder);
	for(int k=0;k<T;k++){
		printf("Topic:%d\n", k);
		dTotalProb = 0;
		for(int w=0;w<W;w++){
			if(dUnigramProbs[w] >= 4)
			  continue;
			dTempProb = GetWordTopicProb(w, k);
			pqTopicalWords.push(Pair<int,double>(w, dTempProb));
			dTotalProb += dTempProb;
		}
		if(!isfinite(dTotalProb) || fabs(dTotalProb-1) > 1e-8){
			fprintf(stderr, "What ?? ... sum p(w|k=%d):%e != 1\n", k, dTotalProb);
			exit(1234);
		}
		for(int i=0;i<aNoWords && ! pqTopicalWords.empty();i++){
			Pair<int, double> p = pqTopicalWords.top();
			printf("%s\t-->\t%e\n", Words[p.First].c_str(), p.Second);
			pqTopicalWords.pop();
		}
		while(! pqTopicalWords.empty())
			pqTopicalWords.pop();
		printf("\n\n");
	}
}

void GuidedLDA::PrintBestTopic(string sFilePath){
	std::stringstream ss;
	ss << sFilePath;
	ofstream out(ss.str().c_str());
	int imax;
	double dmaxprob, dprob;
	for(int di=0;di<D;di++){
		imax = -1;
		for(int k=0;k<T;k++){
			dprob = GetTopicDocProb(k, di);
			if(imax == -1 || dmaxprob < dprob){
				dmaxprob = dprob;
				imax = k;
			}
		}
		out << imax << endl;
	}
}

void GuidedLDA::PrintDocTopicDist(string sFilePath){
	std::stringstream ss;
	ss << sFilePath;
	
	ofstream out(ss.str().c_str());
	//out << "# " << D << " " << T << " " << D*T << endl;
	for(int di=0;di<D;di++){
		double sumCourseTopic = 0.0;
		double sumGeneralTopic = 0.0;
		double sumLogisticTopic = 0.0;
		int maxTopic = 4;
		//out << di+1 ;
		for(int k = 0; k<T; k++) {
			out << GetTopicDocProb(k, di) << "\t";
			
		}
		out << endl;
		/* for(int k=3;k<T;k++){
			//if(dp[di*T+k])
				//out << " " << k+1 << ":" << dp[di*T+k];
				//out << " " << k+1 << ":" << GetTopicDocProb(k,di);
				sumCourseTopic += GetTopicDocProb(k,di);
		}
		sumGeneralTopic += GetTopicDocProb(2,di);
		//out << di+1 << "\t" << sumLogisticTopic;
		//out << endl;
		out << di+1 << "\t" << sumCourseTopic;
		out << endl;
		//out << di+1 << "\t" << sumGeneralTopic;
		//out << endl;	
		if(sumGeneralTopic > sumLogisticTopic) {
			if(sumCourseTopic > sumGeneralTopic)	
				maxTopic = 3;
			else
				maxTopic = 2;	
		}	
		else {
			if(sumCourseTopic > sumLogisticTopic)
				maxTopic = 3;
			else
				maxTopic = 1;
		}
		//out << di+1 << "\t"<<maxTopic;
		//for (int k =0; k < T; k++)
			//out << GetTopicDocProb(k,di) << "\t";
		*/
				
	}
}

int GuidedLDA::LeftOutTokens(){
	int wi,leftout=0, total = 0;
	for(int i=0;i<iNoTokens;i++){
		if(seedTopicWords[w[i]].size() > 0){
			total += 1;
			if(bSeed[i] == false)
				leftout += 1;
		}
	}
	fprintf(stderr, "Leftout %d tokens out of %d\n", leftout, total);
	return leftout;
}

int main(int argc,char *argv[])
{
	int iTopicalWords = 30;
	string sConfigPath, sAssignPath, sJob = "save";
	string sBestTopicFilePath = "/linqshomes/artir/Doctorate/projects/mooc/topic-models/SeededLDA/src/SeededLDA_bestTopic.txt";
	string sDocTopicDistFilePath = "/linqshomes/artir/Doctorate/projects/mooc/topic-models/SeededLDA/src/SeededLDA_docTopicDist.txt";
	if(argc < 2){
		printf("Usage: ./a.out config_file_path model_file_path [load/save]\n");
		return 0;
	}else if(argc == 2){
		sConfigPath = argv[1];
		printf("Running: %s %s\n", argv[0], sConfigPath.c_str());
	}else if(argc == 3){
		sConfigPath = argv[1];
		sAssignPath = argv[2];
		printf("Running: %s %s %s\n", argv[0], sConfigPath.c_str(), sAssignPath.c_str());
	}else if(argc == 4){
		sConfigPath = argv[1];
		sAssignPath = argv[2];
		sJob = argv[3];
		printf("Running: %s %s %s %s\n", argv[0], sConfigPath.c_str(), sAssignPath.c_str(), sJob.c_str());
	}

	GuidedLDA glda;
	glda.LoadData(sConfigPath);
	
	if(sJob == "save"){
		glda.RandomInitialize();
		glda.RandomOrder();
		glda.Iterate(true); //, 0, NULL, 100);
		glda.SaveTopicAssignments(sAssignPath);
	}else if(sJob == "load"){
		glda.LoadTopicAssignments(sAssignPath);
		glda.InitializeFrmPrevCounts();
	}
	//*/
	glda.PrintTopicalWords(iTopicalWords);
	glda.PrintBestTopic(sBestTopicFilePath);
	glda.PrintDocTopicDist(sDocTopicDistFilePath);
	//glda.LeftOutTokens();

}
